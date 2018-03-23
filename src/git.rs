// Copyright 2018 The Rust Project Developers
//
// Licensed under the Apache License, Version 2.0, <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! Get git commits with help of the libgit2 library

const RUST_SRC_URL: &str = "https://github.com/rust-lang/rust";
const RUST_SRC_REPO: Option<&str> = option_env!("RUST_SRC_REPO");

use std::path::Path;

use chrono::{DateTime, TimeZone, Utc};
use git2::{Commit as Git2Commit, Repository};
use git2::build::RepoBuilder;
use failure::Error;

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct Commit {
    pub sha: String,
    pub date: DateTime<Utc>,
    pub summary: String,
}

impl Commit {
    // Takes &mut because libgit2 internally caches summaries
    pub fn from_git2_commit(commit: &mut Git2Commit) -> Self {
        Commit {
            sha: commit.id().to_string(),
            date: Utc.timestamp(commit.time().seconds(), 0),
            summary: String::from_utf8_lossy(commit.summary_bytes().unwrap()).to_string(),
        }
    }
}

pub fn lookup_rev<'rev>(repo: &'rev Repository, rev: &str) -> Result<Git2Commit<'rev>, Error> {
    if let Ok(c) = repo.revparse_single(rev)?.into_commit() {
        return Ok(c);
    }
    bail!("Could not find a commit for revision specifier '{}'", rev)
}

pub fn get_repo() -> Result<Repository, Error> {
    let loc = Path::new("rust.git");
    match (RUST_SRC_REPO, loc.exists()) {
        (Some(_), _) | (_, true) => {
            let repo = Repository::open(RUST_SRC_REPO.map(Path::new).unwrap_or(loc))?;
            {
                let mut remote = repo.find_remote("origin")
                    .or_else(|_| repo.remote_anonymous("origin"))?;
                remote.fetch(&["master"], None, None)?;
            }
            Ok(repo)
        }
        (None, false) => Ok(RepoBuilder::new()
            .bare(true)
            .clone(RUST_SRC_URL, Path::new("rust.git"))?),
    }
}
