# Private and public checkouts

The public repository has its own root commit and main branch. The exporter reads committed blobs from the configured private branch. The selector in `paths.py` includes current Blender, Python, viewer, recipe and skill source, including skill references and scripts. List additional reviewed configuration and documentation paths in `files.txt`. The public ignore file has its own template. Files under `assets/public/` are published byte-for-byte. Private inputs, run records and memory directories stay in the development checkout. Public-only files such as LICENSE may live in public main.

## Configure a private checkout

Run at the private repository root with Git 2.40+, Python 3.11+ and uv. These examples assume there is no other post-commit hook. If one exists, call the supplied hook from it.

```sh
git config --local charloom.publicRepo /absolute/path/to/charloom-public
git config --local charloom.publicBranch "$(git branch --show-current)"
git config --local charloom.publicUv "$(command -v uv)"
uv run --script scripts/public-source/sync.py
hook=$(git rev-parse --git-path hooks/post-commit)
test ! -e "$hook" && cp scripts/public-source/post-commit "$hook"
```

The hook updates the local public checkout on a private commit. After a fast-forward pull, or if a hook reports an error, rerun the same sync command. An independent public commit pauses export until integration. To publish later, set origin in the public checkout. Set `charloom.publicPush=true` in the private checkout only if commits should also push to origin/main. A rejected push is reported and may be retried after integrating upstream.

## Bring public updates back

```sh
scripts/public-source/merge-public --check
scripts/public-source/merge-public
git diff --cached
git commit -m "Integrate public source updates"
```

Add `--pull` to fetch origin/main and integrate it in the public checkout first. Git fast-forwards where possible and merges divergent public commits. Resolve any conflict from that operation in the public checkout, then rerun the importer. Review what is staged before committing, including any pre-existing staged changes. Conflicts appear in `git status`; edit each file, stage the resolution and commit. The public ignore file maps to its template. Source and README changes map to their original paths. Disjoint private code and comments remain intact. New public source and shared assets are selected by the same rules used for export. Files reported as public-only stay in public main. An empty private merge can be acknowledged with `git commit --allow-empty`. To cancel, use the paths reported by the preview to restore any imported work you want to discard, then run `merge-public --abort` to clear the tracking record.

The private branch's commit triggers export of the integrated tree. The importer tracks its base in the public checkout's local Git metadata. Keep that checkout for subsequent syncs. Only push the public repository's main branch to its public remote.
