# Charloom Gallery

English · [简体中文](README.zh-CN.md)

The local Gallery displays registered character results, compares the original scan with bare and dressed exports, and plays their animation clips. Open the session URL printed by Vite. Direct character links support refresh and browser navigation.

```sh
bun install --frozen-lockfile
bun run dev
```

For the production build, use `bun run build` and `bun run preview`. The read-only catalog server runs in both modes. Register results with `scripts/mh-ide-sync.py` in the checkout. The [Blender extension and Python scripts](../README.md) handle authoring.
