# Releasing

Both SDKs are published from GitHub Actions with trusted publishing: no PyPI or npm token is stored anywhere, and each release carries a provenance attestation linking it to the workflow run and commit that built it.

## One-time setup

1. **PyPI:** on pypi.org, project `agentfences` → Manage → Publishing → add a trusted publisher: GitHub, owner `bhaviksardar`, repository `fences`, workflow `publish-pypi.yml`, environment `pypi`. On GitHub: Settings → Environments → New environment `pypi` (optionally with required reviewers, so a release waits for your click).
2. **npm:** on npmjs.com, package `agentfences` → Settings → Trusted publisher → GitHub Actions, repository `bhaviksardar/fences`, workflow `publish-npm.yml`.
3. **Signed tags** (SSH signing, no GPG needed):
   ```sh
   git config --global gpg.format ssh
   git config --global user.signingkey ~/.ssh/id_ed25519.pub
   git config --global tag.gpgSign true
   ```
   Then add the same public key on GitHub (Settings → SSH and GPG keys → New SSH key, type **Signing key**), so tags show as Verified.

## Each release

1. Bump the version in `sdk/agentfences/__init__.py`, and in `sdk-js/package.json`, `sdk-js/package-lock.json` and `sdk-js/src/version.ts`. Update the changelog in `PROTOCOL.md`.
2. Commit and push to `main`, and wait for CI to pass.
3. Tag and push:
   ```sh
   git tag -s v0.9.0 -m "agentfences 0.9.0 (Python)"
   git tag -s sdk-js-v0.9.0 -m "agentfences 0.9.0 (TypeScript)"
   git push origin v0.9.0 sdk-js-v0.9.0
   ```
   `v*` tags publish the Python SDK (`publish-pypi.yml`), `sdk-js-v*` tags the TypeScript SDK (`publish-npm.yml`). Each workflow checks that the tag matches the version in the code and runs the tests before publishing.
4. Tell the platform's `SDK_LATEST` about the new versions.

Dependabot opens weekly PRs for the Actions the workflows use and for the SDKs' dependencies.
