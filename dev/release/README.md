<!---
  Licensed to the Apache Software Foundation (ASF) under one
  or more contributor license agreements.  See the NOTICE file
  distributed with this work for additional information
  regarding copyright ownership.  The ASF licenses this file
  to you under the Apache License, Version 2.0 (the
  "License"); you may not use this file except in compliance
  with the License.  You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing,
  software distributed under the License is distributed on an
  "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
  KIND, either express or implied.  See the License for the
  specific language governing permissions and limitations
  under the License.
-->

# Releasing SedonaDB

## Verifying a release candidate

Release candidates are verified using the script `verify-release-candidate.sh <version> <rc_num>`.
For example, to verify SedonaDB 0.2.0 RC0, run:

```shell
# git clone https://github.com/apache/sedona-db.git && cd sedona-db
# or
# cd existing/sedona-db && git fetch upstream && git switch main && git pull upstream main
dev/release/verify-release-candidate.sh 0.2.0 0
```

Release verification requires a recent Rust toolchain. This toolchain can be installed
by following instructions from <https://rustup.rs/>.

MacOS users can use [Homebrew](https://brew.sh) to install the required dependencies.

```shell
brew install geos proj openssl abseil
```

Linux users (e.g., `docker run --rm -it condaforge/mambaforge`) can use `conda` to
install the required dependencies:

```shell
conda create -y --name verify-sedona-db
conda activate verify-sedona-db
conda install -y compilers curl gnupg geos proj openssl libabseil cmake make pkg-config
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$CONDA_PREFIX/lib"
```

When verifying via Docker or on a smaller machine it may be necessary to limit the
number of parallel jobs to avoid running out of memory:

```shell
export CARGO_BUILD_JOBS=4
```

Verifiers may opt in to additional features normally tested in CI with dedicated tooling.
For example, to verify with specific Python build-time features, the `MATURIN_PEP517_ARGS`
environment variable may be set.

```shell
export MATURIN_PEP517_ARGS="--features s2geography"
```

## Pre-release

Approaching the release date, create a draft pull request into Apache Sedona
that contains a draft of the release post for the
[Apache Sedona Blog](https://sedona.apache.org/latest/blog). Ensure that each
release highlight has an example and that the example runs. Blog posts are
rendered from the [docs/blog/posts](https://github.com/apache/sedona/tree/master/docs/blog/posts)
subdirectory of the apache/sedona repository.

Before the release branch is created, the GitHub milestone should also be updated.
Items that should be included in the milestone are:

- Issues that were closed as completed (e.g., a PR closed them or some other task
  resolved them in a different way).
- Pull Requests that did not close an issue (i.e., a PR can both represent a discussion
  of a problem and the code change that solves it).

The milestone issue/pull request count is used in the blog post and the email send to the
mailing list to start the vote.

## Creating a release

Create a release branch on the corresponding remote pointing to the official Apache
repository (i.e., <https://github.com/apache/sedona-db>). This step must be done by
a committer.

```shell
git pull upstream main
git checkout -b branch-0.2.0
git push upstream -u branch-0.2.0:branch-0.2.0
```

This push should cause two CI runs to begin:

- <https://github.com/apache/sedona-db/actions/workflows/packaging.yml>
- <https://github.com/apache/sedona-db/actions/workflows/python-wheels.yml>

The verification run will create the release source tarball and documentation,
which from the Apache release's perspective are the only artifacts that are
being verified. The Python wheels (and the tests that are run as they are created)
are considered a "packaging" step (i.e., the artifacts aren't uploaded to the
release or voted on), although those CI jobs are important to ensuring
the release is ready for a vote.

Before creating a tag, download the tarball from the latest packaging run and
check it locally:

```shell
dev/release/verify-release-candidate.sh path/to/tarball.tar.gz
```

When the state of the `branch-x.x.x` branch is clean and checks are complete,
the release candidate tag can be created:

```shell
git tag -a apache-sedona-db-0.2.0-rc0 -m "Tag Apache SedonaDB 0.2.0-rc0"
git push upstream apache-sedona-db-0.2.0-rc0
```

This will trigger another packaging CI run that, if successful, will create a
pre-release at <https://github.com/apache/sedona-db/releases> with the release
artifacts uploaded from the CI run.

After the release has been created with the appropriate artifacts, the assets
need to be signed with signatures uploaded as release assets. Please create
dev/release/.env from dev/release/.env.example and set the GPG_KEY_ID variable.
The GPG_KEY_ID in dev/release/.env must have its public component listed in the
[Apache Sedona KEYS file](https://dist.apache.org/repos/dist/dev/sedona/KEYS).

```shell
# sign-assets.sh <version> <rc_number>
dev/release/sign-assets.sh 0.2.0 0
```

After the assets are signed, they can be committed and uploaded to the
dev/sedona directory of the Apache distribution SVN. A helper script
is provided:

```shell
# upload-candidate.sh <version> <rc_number>
APACHE_USERNAME=your_apache_username dev/release/upload-candidate.sh 0.2.0 0
```

## Vote

An email must now be sent to `dev@sedona.apache.org` calling on developers to follow
the release verification instructions and vote appropriately on the source release.
The following may be used as a template:

```
[VOTE] Release Apache SedonaDB 0.2.0-rc0

Hello,

I would like to propose the following release candidate (rc0) of Apache SedonaDB [0] version 0.2.0. This is a release consisting of 138 resolved GitHub issues from 17 contributors [1].

This release candidate is based on commit: 69f89a9a0cced55a3792eb6dfe1f76e9dd01033c [2]

The source release rc0 is hosted at [3].

Please download, verify checksums and signatures, run the unit tests, and vote on the release. See [4] for how to validate a release candidate. See [5] for a draft release post with release highlights.

The vote will be open for at least 72 hours.

[ ] +1 Release this as Apache SedonaDB 0.2.0
[ ] +0
[ ] -1 Do not release this as Apache SedonaDB 0.2.0 because...

[0] https://github.com/apache/sedona-db
[1] https://github.com/apache/sedona-db/milestone/1?closed=1
[2] https://github.com/apache/sedona-db/tree/apache-sedona-db-0.2.0-rc1
[3] https://dist.apache.org/repos/dist/dev/sedona/apache-sedona-db-0.2.0-rc1
[4] https://github.com/apache/arrow-nanoarrow/blob/main/dev/release/README.md
[5] https://github.com/apache/sedona/pull/2540
```

Ensure the links point to the correct assets/blog post pull request, the issue count has been updated
from the latest GitHub milestone, and the contributor count has been updated from the
`git shortlog` command on the release blog post.

## Publish

### Upload/tag source release

After a successful release vote, the tarball needs to be uploaded to the official
Apache release repository. A helper script is provided:

```shell
# upload-release.sh <version> <rc_number>
dev/release/upload-release.sh 0.2.0 0
```

An official Git tag must also be created and uploaded to the Apache remote:

```shell
git tag -a apache-sedona-db-0.2.0 -m "SedonaDB 0.2.0" apache-sedona-db-0.2.0-rc0^{}
git push upstream apache-sedona-db-0.2.0
```

The prerelease located at <https://github.com/apache/sedona-db/releases/tag/apache-sedona-db-0.2.0-rc0>
can now be edited to point to the official release tag and the GitHub release published
from the UI. The release notes may be automatically generated by selecting
`apache-sedona-db-0.2.0.dev` as the previous release.

### Publish Python package

Locate the latest run identifier for the appropriate run of the python-wheels workflow
that was run on the release branch:
<https://github.com/apache/sedona-db/actions/workflows/python-wheels.yml>. The
artifacts can be downloaded and extracted with `gh run download`.

```shell
# Clear the wheels directory
rm -rf wheels
mkdir wheels

# Download assets from the latest `branch-x.x.x` branch run,
# remove the pyodide wheels (which will be rejected by PyPI)
pushd wheels
gh run download 15963020465
popd
```

Use `twine` to upload the release to PyPI. This will require a token created
in the PyPI UI.

```shell
# pip install twine
twine upload wheels/**/*.whl
rm -rf wheels
```

### Upload to `crates.io`

A script is provided to upload the crates to <https://crates.io>.

```shell
cargo login
dev/release/upload-crates-io.sh
```

### Publish release blog post

Mark the [pull request into the Apache Sedona Blog](https://github.com/apache/sedona/pulls)
as ready for review, solicit final reviews, and merge to publish the post to the
[Apache Sedona Blog](https://sedona.apache.org/latest/blog).

### Send email to `announce@apache.org` mailing list

The following template may be used:

```
[ANNOUNCE] Apache SedonaDB 0.2.0 released

Dear all,

We are happy to report that we have released Apache SedonaDB 0.2.0.
Thank you again for your help.

SedonaDB is the first open-source, single-node analytical database engine that treats spatial data as a first-class citizen. It is developed as a subproject of Apache Sedona.

Website:
http://sedona.apache.org/sedonadb

Release notes:
https://sedona.apache.org/latest/blog/2025/12/01/sedonadb-020-release/

Download link:
https://www.apache.org/dyn/closer.cgi/sedona/apache-sedona-db-0.2.0

Additional resources:
Mailing list: dev@sedona.apache.org
Discord: https://discord.gg/9A3k5dEBsY

Regards,
Apache Sedona Team
```

Ensure that:

- The link to the release notes is correct/update to the current release
- The email is sent from your Apache email account (i.e., `xxx@apache.org`)
  via the official Apache outgoing mail server.
- The email is plain text and contains no HTML formatting (or it will be rejected
  from the announce list).
- The email contains a download link (or it will be rejected from the announce list)

## Bump versions

After a successful release, versions on the `main` branch need to be updated. These
are currently all derived from `Cargo.toml`, which can be updated to:

```
[workspace.package]
version = "0.3.0"
```

The R package must also be updated. R Packages use a different convention for development
versions such that in preparation for 0.3.0 the development version should be
`0.2.0.9000`. This is set the DESCRIPTION of the requisite package.

Development versions and the changelog are derived from the presence of a development
tag on the main branch signifying where development of that version "started". After
the version bump PR merges, that commit should be tagged with the appropriate
development tag:

```shell
git tag -a apache-sedona-db-0.3.0.dev -m "tag dev 0.3.0"
git push upstream apache-sedona-db-0.3.0.dev
```
