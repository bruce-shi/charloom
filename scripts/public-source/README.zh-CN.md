# 私库与公库工作副本

[English](README.md) · 简体中文

公库有独立的根提交和 main 分支。导出工具从指定私库分支读取已提交的文件。`paths.py` 选择当前 Blender、Python、Viewer、配方、skills 及配套文档和脚本。额外审核过的配置可列在 `files.txt` 中。公库使用自己的忽略模板；`assets/public/` 按原字节发布。私有输入、运行记录和记忆留在开发仓库。公库 main 也可保留仅在公库维护的文件，例如 LICENSE。

工具会自动导出 `docs/` 中的 Markdown 指南。运行历史放在 `docs/history/`，检查点放在 `docs/checkpoints/`。英文文件用 `topic.md`，简体中文用 `topic.zh-CN.md`。文内引用链接到同一语言的页面和章节。技能的英文入口保留 `SKILL.md` 文件名，供 agent 发现。

## 配置私库工作副本

在私库根目录运行以下命令，需要 Git 2.40+、Python 3.11+ 和 uv。示例假定没有其他 post-commit 钩子。若已有钩子，请从现有钩子调用项目提供的钩子。

```sh
git config --local charloom.publicRepo /absolute/path/to/charloom-public
git config --local charloom.publicBranch "$(git branch --show-current)"
git config --local charloom.publicUv "$(command -v uv)"
uv run --script scripts/public-source/sync.py
hook=$(git rev-parse --git-path hooks/post-commit)
test ! -e "$hook" && cp scripts/public-source/post-commit "$hook"
```

在私库提交时，钩子会更新本地公库工作副本。快进拉取后或钩子报错时，重新执行同一同步命令。若公库有独立提交，先整合再继续导出。发布前应在公库配置 origin。若希望每次提交同时推送至 origin/main，可在私库设置 `charloom.publicPush=true`。推送被拒时会报告错误，整合上游后可重试。

## 将公库更新合回私库

```sh
scripts/public-source/merge-public --check
scripts/public-source/merge-public
git diff --cached
git commit -m "Integrate public source updates"
```

加 `--pull` 会先取回 origin/main 并在公库工作副本整合。Git 能快进时会快进，分叉时会合并。在公库工作副本解决冲突后重新运行导入命令。提交前复核暂存区，包括先前暂存的改动。通过 `git status` 找到冲突文件，编辑、暂存并提交。公库的忽略配置映射回模板，源码和 README 映射回原路径。互不重叠的私有代码和注释会保留。新增源码和共享资产沿用导出规则，标为仅公库的文件留在公库 main。可用 `git commit --allow-empty` 确认空合并。需要取消时，先按预览列出的路径撤销本次想放弃的修改，再运行 `merge-public --abort` 清除跟踪记录。

私库分支的提交会触发整合后文件树的导出。导入器在公库的本地 Git 元数据中记录基准，保留该工作副本供后续同步。往公开远端推送时，只推送公库的 main 分支。
