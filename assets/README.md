# Asset directories

English · [简体中文](README.zh-CN.md)

Store assets cleared for redistribution and their attribution or license files in `assets/public/`. After you commit them in the development repository, the mirror publishes them at the same paths.

Keep private assets, character inputs and their annotations in `assets/private/`. Existing scans, licensed reference models, generated test models, motion sources and run outputs retain their original paths. The public mirror keeps its published fixtures. Choose the appropriate directory for each new asset.

Skills and scripts read local GLBs, motion files, templates and registered references supplied by the caller. Prepare the inputs specified by the skill for the character you intend to build.

Batch tools read a local JSON file through `CHARLOOM_BATCH_CONFIG`. In `characters`, map each character name to the source model path, template path and annotation directory name, in that order. Use `smoke` to list names for a quick check. The shell batch runner reads your zsh input table through `CHARLOOM_BATCH_TABLE`. Character-specific replays require the corresponding bound model, motion and configuration.
