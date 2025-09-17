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

## Creating a release

Create a release branch on the corresponding remote pointing to the official Apache
repository (i.e., <https://github.com/apache/sedona-db>). This step must be done by
a committer.

```shell
git pull upstream main
git branch -b branch-0.1.0
git push upstream -u branch-0.1.0:branch-0.1.0
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

When the state of the `branch-x.x.x` branch is clean and checks are complete,
the release candidate tag can be created:

```shell
git tag -a apache-sedona-db-0.1.0-rc0 -m "Tag Apache SedonaDB 0.1.0-rc0"
git push upstream apache-sedona-db-0.1.0-rc0
```

This will trigger another packaging CI run that, if successful, will create a
pre-release at <https://github.com/apache/sedona-db/releases> with the release
artifacts uploaded from the CI run.

After the release has been created with the appropriate artifacts, the assets
need to be signed with signatures uploaded as release assets. The GPG_KEY_ID
must have its public component listed in the
[Apache Sedona KEYS file](https://dist.apache.org/repos/dist/dev/sedona/KEYS).

```shell
# sign-assets.sh <version> <rc_number>
GPG_KEY_ID=your_gpg_key_id dev/release/sign-assets.sh 0.1.0 0
```

After the assets are signed, they can be committed and uploaded to the
dev/sedona directory of the Apache distribution SVN. A helper script
is provided:

```shell
# upload-candidate.sh <version> <rc_number>
APACHE_USERNAME=your_apache_username dev/release/upload-candidate.sh 0.1.0 0
```

## Vote

An email must now be sent to `dev@sedona.apache.org` calling on developers to follow
the release verification instructions and vote appropriately on the source release.

## Publish

### Upload/tag source release

After a successful release vote, the tarball needs to be uploaded to the official
Apache release repository. A helper script is provided:

```shell
# upload-release.sh <version> <rc_number>
APACHE_USERNAME=your_apache_username dev/release/upload-release.sh 0.1.0 0
```

An official GitHub tag must also be created:

```shell
git tag -a apache-sedona-db-0.1.0 -m "SedonaDB 0.1.0" apache-sedona-db-0.1.0-rc0
git push upstream apache-sedona-db-0.1.0
```

The prerelease located at <https://github.com/apache/sedona-db/releases/tag/apache-sedona-db-0.1.0-rc0>
can now be edited to point to the official release tag and the GitHub release published
from the UI.

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
