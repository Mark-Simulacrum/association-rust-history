#![feature(fs_read_write)]

extern crate chrono;
extern crate env_logger;
#[macro_use]
extern crate failure;
extern crate bincode;
extern crate git2;
extern crate petgraph;
extern crate serde;
#[macro_use]
extern crate serde_derive;

use std::collections::{HashMap, HashSet};
use std::fs;
use std::fmt;
use std::path::PathBuf;
use std::f64;

use chrono::{TimeZone, Utc};
use failure::Error;
use git2::{Diff, Repository, Tree};
use petgraph::{Graph, graph::NodeIndex};

mod git;

use git::Commit as CachedCommit;

fn load_repo() -> Result<(Repository, Vec<CachedDiff>, Vec<CachedCommit>), Error> {
    let buf = fs::read("cache.bincode").ok().unwrap_or_else(Vec::new);
    let mut diffs = bincode::deserialize(&buf[..]).ok().unwrap_or_else(|| {
        eprintln!("removing cache...");
        let _ = fs::remove_file("cache.bincode");
        HashMap::new()
    });
    let repo = git::get_repo()?;
    let mut checked = HashSet::new();

    {
        let master_rev = git::lookup_rev(&repo, "origin/master")?.clone();
        let mut queue = vec![master_rev];
        while let Some(mut cur) = queue.pop() {
            if !checked.insert(cur.id()) {
                continue;
            }
            let date = Utc.timestamp(cur.time().seconds(), 0);
            if date < Utc.ymd(2017, 1, 1).and_hms(0, 0, 0) {
                //if date >= Utc.ymd(2018, 3, 11).and_hms(0, 0, 0) {
                continue;
            }

            for commit in cur.parents() {
                queue.push(commit);
            }

            // not a merge commit
            if cur.parents().count() == 2 {
                let _merge_base = cur.parent(0)?; // not actually important
                let code_commit = cur.parent(1)?;
                let mut code_commit_cur = cur.parent(1)?;
                let merge_root = loop {
                    if code_commit_cur.parents().count() == 1 {
                        code_commit_cur = code_commit_cur.parent(0)?;
                    } else {
                        // Hit merge commit, stopping
                        break code_commit_cur;
                    }
                };
                diffs.entry(cur.id().to_string()).or_insert_with(|| {
                    let parent_tree = merge_root.tree().unwrap();
                    let cur_tree = code_commit.tree().unwrap();
                    let diff = repo.diff_tree_to_tree(Some(&parent_tree), Some(&cur_tree), None)
                        .unwrap();
                    CachedDiff::from(CachedCommit::from_git2_commit(&mut cur), &diff)
                });
            }
        }
    }

    eprintln!("commits: {}", checked.len());
    eprintln!("diffs: {}", diffs.len());

    let buf = bincode::serialize(&diffs).unwrap();
    fs::write("cache.bincode", &buf)?;

    Ok((
        repo,
        diffs.iter().map(|(_, v)| v.clone()).collect(),
        diffs.iter().map(|(_, v)| v.for_commit.clone()).collect(),
    ))
}

#[derive(Deserialize, Serialize, Debug, Clone)]
struct CachedDiff {
    for_commit: CachedCommit,
    deltas: Vec<CachedDelta>,
}

impl CachedDiff {
    fn from(commit: CachedCommit, diff: &Diff) -> Self {
        let deltas = diff.deltas()
            .map(|d| d.old_file().path().expect("path"))
            .map(|p| CachedDelta { path: p.to_owned() })
            .collect::<Vec<_>>();
        CachedDiff {
            for_commit: commit,
            deltas,
        }
    }
}

#[derive(Deserialize, Serialize, Debug, Clone)]
struct CachedDelta {
    path: PathBuf,
}

fn run() -> Result<(), Error> {
    env_logger::init();

    // Compute probability that when changing node A, node B will also change.
    // Basically, the number of times this undirected edge was touched over
    // the total number of diffs analyzed.
    let mut graph = Graph::<String, usize, petgraph::Undirected>::default();
    let mut nodes = HashMap::new();
    let mut edges = HashMap::new();

    let (repository, diffs, commits) = load_repo()?;
    let master_commit = git::lookup_rev(&repository, "origin/master")?;
    let master_tree = master_commit.tree()?;
    let dirs = diffs
        .iter()
        .map(|d| to_dirs(&master_tree, d))
        .collect::<Vec<_>>();

    let now = Utc::now();
    let last_week = commits
        .iter()
        .filter(|c| now.signed_duration_since(c.date).num_days() <= 7)
        .count();
    // we're allowed one cross commit per week
    let allowed_sync_points_per_week = 1.0;
    let probability = allowed_sync_points_per_week / (last_week as f64);

    eprintln!(
        "allowed breakage per week: {}",
        allowed_sync_points_per_week
    );
    eprintln!("permitted breakage probabilty: {}", probability);

    for dirs in &dirs {
        let mut seen_edges = HashSet::new();
        for a in dirs {
            for b in dirs {
                let a_idx = *nodes
                    .entry(a.clone())
                    .or_insert_with(|| graph.add_node(a.clone()));
                let b_idx = *nodes
                    .entry(b.clone())
                    .or_insert_with(|| graph.add_node(b.clone()));
                if a_idx != b_idx {
                    let mut indices = [a_idx, b_idx];
                    indices.sort();
                    let a_idx = indices[0];
                    let b_idx = indices[1];
                    let edge_idx = *edges.entry((a_idx, b_idx)).or_insert_with(|| {
                        assert!(!graph.contains_edge(a_idx, b_idx));
                        graph.add_edge(a_idx, b_idx, 0)
                    });
                    if seen_edges.insert(edge_idx) {
                        *graph.edge_weight_mut(edge_idx).unwrap() += 1;
                    }
                }
            }
        }
    }

    let root = graph.node_indices().next().expect(">0 nodes");
    let mut visited = HashSet::new();
    visited.insert(root);
    let mut clusters = Vec::new();

    let mut queue = vec![root];
    loop {
        let mut cluster = Vec::new();
        if let Some(idx) = graph.node_indices().find(|i| !visited.contains(i)) {
            queue.push(idx);
        } else {
            break;
        }
        while let Some(cur) = queue.pop() {
            if !visited.insert(cur) {
                continue;
            }
            cluster.push(cur);
            for neighbor in graph.neighbors(cur) {
                // opt?: lookup in edges map
                if let Some(edge_idx) = graph.find_edge(cur, neighbor) {
                    let weight = graph[edge_idx];
                    let prob = to_prob(diffs.len(), weight);
                    // The bound here is the minimum probability from which we keep things together.
                    // That means that n% of changes will have to sync across the two clusters.
                    // if false, then split into two
                    // if true, keep together
                    // if 100% of commits touched this edge, prob = 100%
                    // if 50% of commits did, then prob = 50%
                    // we want to follow this edge only if the edge is touched
                    // by more commits than our allowed probability
                    // (`probability`) allows.
                    if prob >= probability {
                        queue.push(neighbor);
                    }
                }
            }
        }

        let mut cluster = Cluster {
            min_prob: 0.0,
            nodes: cluster
                .iter()
                .map(|idx| (*idx, graph[*idx].clone()))
                .collect::<Vec<_>>(),
        };

        cluster.nodes.sort_by_key(|c| c.1.clone());

        clusters.push(cluster);
    }

    clusters.sort_by_key(|c| c.nodes[0].1.clone());

    eprintln!("clusters:");
    for cluster in &clusters {
        eprintln!("{}", cluster);
    }

    eprintln!("clusters: {}", clusters.len());
    eprintln!("");

    //let mut cluster_graph = Graph::<Cluster, f64, petgraph::Undirected>::default();
    //let mut cluster_graph_nodes = HashMap::new();
    for (i, cluster) in clusters.iter().take(1).enumerate() {
        //let cluster_idx = *cluster_graph_nodes
        //    .entry(i)
        //    .or_insert_with(|| cluster_graph.add_node(cluster.clone()));
        for (j, other) in clusters.iter().enumerate() {
            if i == j {
                continue;
            }

            // a = cluster, b = other
            let mut min = f64::INFINITY;
            for &(a, _) in cluster.nodes.iter() {
                for &(b, _) in other.nodes.iter() {
                    if let Some(edge_idx) = graph.find_edge(a, b) {
                        let weight = graph[edge_idx];
                        let prob = to_prob(diffs.len(), weight);
                        min = min.min(prob);
                    }
                }
            }
            eprintln!(
                "{} would get absorbed if limit was 1 sync in {:.01} days",
                other,
                1.0 / (min * (last_week as f64)) * 7.0
            );

            //let other_idx = *cluster_graph_nodes
            //    .entry(j)
            //    .or_insert_with(|| cluster_graph.add_node(other.clone()));
            //cluster_graph.update_edge(cluster_idx, other_idx, min);
        }
    }

    //fs::write(
    //    "graph.dot",
    //    format!("{}", petgraph::dot::Dot::new(&cluster_graph)),
    //)?;

    Ok(())
}

#[derive(Debug, Clone)]
struct Cluster {
    min_prob: f64,
    nodes: Vec<(NodeIndex, String)>,
}

impl fmt::Display for Cluster {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        for &(_, ref s) in &self.nodes[..self.nodes.len() - 1] {
            write!(f, "{}, ", s)?;
        }
        write!(f, "{}", self.nodes.last().unwrap().1)?;
        Ok(())
    }
}

fn to_prob(diffs: usize, weight: usize) -> f64 {
    (weight as f64) / (diffs as f64)
}

fn to_dirs(master_tree: &Tree, d: &CachedDiff) -> HashSet<String> {
    let mut dirs = HashSet::new();
    for delta in &d.deltas {
        let path = &delta.path;
        if !path.starts_with("src") {
            continue;
        }
        // ignore paths that no longer exist
        if master_tree.get_path(&path).is_err() {
            continue;
        }
        // We're only interested in things inside src/
        if let Some(component) = path.components().nth(1) {
            let dir = component.as_os_str().to_string_lossy().to_string();
            dirs.insert(dir);
        }
    }
    dirs
}

fn main() {
    run().unwrap();
}
