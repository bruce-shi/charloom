# Charloom

Charloom fits humanoid scans to a rigged MakeHuman prior, separates and skins garments, and retargets animation. The workspace includes a Rust CLI, a local service, an MCP server and a React character viewer.

## Build

```sh
cargo build --locked --workspace --profile iterate
cd viewer
bun install --frozen-lockfile
bun run build
cd ..
target/iterate/charloom --help
```

## Fit a scan

Supply a scan and a rigged MakeHuman GLB. To prepare annotated views, run `charloom mh-guide --help`. To fit the body, skin the garments and retarget a motion, run the following command from the repository root.

```sh
target/iterate/charloom mh-transfer scan.glb --prior prior.glb --motion walk.fbx --out out/my-character
python3 scripts/mh-ide-sync.py out/my-character
target/iterate/charloom ide
```

## Verify

```sh
cargo test --locked --workspace
cd viewer
bun run test:local
```
