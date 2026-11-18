# /// script
# requires-python = ">=3.11"
# dependencies = ["tree-sitter==0.25.2", "tree-sitter-rust==0.24.0", "tree-sitter-typescript==0.23.2"]
# ///
"""Project one committed source tree into an independent local Git repository."""

import argparse
import io
import json
from pathlib import Path
import re
import subprocess
import tokenize

from tree_sitter import Language, Parser
import tree_sitter_rust
import tree_sitter_typescript

from paths import public_path

from common import (
    SOURCE,
    OVERLAYS,
    PREFIX,
    config,
    configured_destination,
    files_at,
    git,
    locked_public,
    regular_blob,
    updated_tree,
    write_json,
)

PARSERS = {
    ext: Parser(Language(grammar()))
    for ext, grammar in [
        (".rs", tree_sitter_rust.language),
        (".ts", tree_sitter_typescript.language_typescript),
        (".tsx", tree_sitter_typescript.language_tsx),
    ]
}
PRIVATE = re.compile(
    r"/(?:Users|Volumes|home)/[^/\s]+/|\.codex/(?:memories|sessions)|BEGIN [A-Z ]*PRIVATE KEY|gh[pousr]_[A-Za-z0-9]{24,}|AKIA[A-Z0-9]{16}|sk-[A-Za-z0-9_-]{32,}",
    re.I,
)
PRIVATE_BYTES = re.compile(PRIVATE.pattern.encode(), re.I)
HISTORICAL = re.compile(
    r"\b(?:old|formerly|previous(?:ly)?|originally|used to|earlier|history|experiment\w*|workaround|lesson|checkpoint|regression|rejected|prototype|reviewed|baseline|bug|broken|instead|unlike|was|were|fixed|prior run|agent|codex|brute|goblin|fox|orc|troll|hiker|starbow|nujiao|dimlight|herbalist|kawaii|divine.guard|swordswoman|mage|idol|x\d{2,}|r\d{2,}|v\d+|chui)\b|out/(examples|review|benchmarks)|skills/|kr-|202[0-9][-0-9]{4}|→",
    re.I,
)


def public_comments(path, data):
    tree = PARSERS[path.suffix].parse(data)
    # Treat consecutive documentation lines as one comment. A removed account
    # of a private experiment must not leave its continuation lines behind.
    stack = [tree.root_node]
    groups = []
    while stack:
        node = stack.pop()
        if node.type in ("line_comment", "block_comment", "comment"):
            a, b = node.start_byte, node.end_byte
            if (
                groups
                and not data[groups[-1][1] : a].strip()
                and data[groups[-1][1] : a].count(b"\n") <= 1
            ):
                groups[-1] = (groups[-1][0], b)
            else:
                groups.append((a, b))
        else:
            stack.extend(reversed(node.children))
    for a, b in reversed(groups):
        comment = data[a:b].decode()
        if HISTORICAL.search(comment) or PRIVATE.search(comment):
            if re.search(
                r"copyright|spdx-license|@license|@ts-|vite-ignore|__PURE__",
                comment,
                re.I,
            ):
                raise ValueError(
                    f"mixed notice/directive and private comment in {path}"
                )
            data = data[:a] + b" " + data[b:]
    return data


def remove_section(s, start, end):
    if s.count(start) != 1 or s.count(end) != 1:
        raise ValueError(f"review changed public boundary: {start.strip()}")
    a = s.index(start)
    return s[:a] + s[s.index(end, a) :]


def project(name, data):
    p = Path(name)
    if name.startswith("assets/public/"):
        if PRIVATE_BYTES.search(data):
            raise ValueError(f"Private content marker in {name}")
        return data
    s = data.decode()
    if name == "crates/charloom-cli/src/main.rs":
        s = remove_section(
            s, "    /// Rebuild the recorded bare", "    /// Apply sculpted mouth"
        )
        s = remove_section(
            s, "        Command::MhRebuildRoster", "        Command::MhFitSculptMouth"
        )
    elif name.endswith("/roster/mod.rs"):
        s = s.replace("pub mod rebuild;\n", "")
    elif name == "Cargo.toml":
        s = remove_section(s, "[workspace.metadata.blaze]", "[workspace.dependencies]")
    elif name == "scripts/mh-ide-sync.py":
        s = remove_section(s, "CHANGES = [", "GENERIC = ")
        s = remove_section(s, "    if group in GROUP_CHANGES:", '    return ""\n')
        s = remove_section(
            s,
            '            if character_of(Path(r.get("source", ""))) in {',
            "            digest = ",
        )
        s = remove_section(
            s, "        sheet = out /", "        return standard, closeups"
        )
        # Drop the private workflow narrative, preserving any future legal notices.
        a = s.index('"""')
        b = s.index('"""', a + 3)
        s = (
            s[:a]
            + '"""Register local transfer results in the character viewer."""'
            + s[b + 3 :]
        )
        s = tokenize.untokenize(
            [
                t
                for t in tokenize.generate_tokens(io.StringIO(s).readline)
                if t.type != tokenize.COMMENT
                or re.search(r"^#!|copyright|license", t.string, re.I)
            ]
        )
    # Keep the public defaults independent of internal run directories.
    if name in {
        "scripts/roster/rebuild.py", "scripts/roster/verify-lips.ts",
        "recipes/goblin-fixed-rig/replay.py",
    }:
        s = re.sub(r"out/review/full-roster-\d{8}", "out/transfers", s)
    if p.suffix in PARSERS:
        s = public_comments(p, s.encode()).decode()
    if PRIVATE.search(s):
        raise ValueError(f"private content marker in {name}")
    return s.encode()


def sync(dest, branch):
    if dest == SOURCE or dest in SOURCE.parents or SOURCE in dest.parents:
        raise ValueError("Choose a separate directory for public source")
    if not dest.exists():
        dest.mkdir()
        git(dest, "init", "--quiet", "--initial-branch=main", "--object-format=sha1")
        git(dest, "config", "--local", "charloom.sourceMirror", "true")
    with locked_public(dest):
        state_path = dest / ".git/source-sync.json"
        state = json.loads(state_path.read_text()) if state_path.exists() else {}
        head = next(
            iter(
                git(dest, "for-each-ref", "--format=%(objectname)", "refs/heads/main")
                .decode()
                .splitlines()
            ),
            None,
        )
        rev = git(
            SOURCE, "rev-parse", "--verify", f"refs/heads/{branch}^{{commit}}"
        ).strip()
        marker = Path(
            git(
                SOURCE,
                "rev-parse",
                "--path-format=absolute",
                "--git-path",
                "charloom-public-merge.json",
            )
            .decode()
            .strip()
        )
        accepted = None
        if marker.exists():
            pending = json.loads(marker.read_text())
            if rev.decode() == pending["private"]:
                raise ValueError("Commit the pending public merge first")
            git(SOURCE, "merge-base", "--is-ancestor", pending["private"], rev)
            if state.get("base") != pending["base"]:
                raise ValueError(
                    "Public merge base changed, check the pending integration"
                )
            uncommitted = git(SOURCE, "diff", "--cached", "--name-only", "-z", rev)
            if set(uncommitted.decode().split("\0")) & set(pending["paths"]):
                raise ValueError(
                    "Private commit omitted staged public changes. Commit the complete merge."
                )
            accepted = pending["public"]
        expected = accepted or state.get("base")
        if head != expected:
            raise ValueError(
                "Public main has independent commits. Run scripts/public-source/merge-public first."
            )
        files = files_at(SOURCE, rev)

        def contents(path):
            return regular_blob(SOURCE, files[path])[1]

        listed = set(contents(PREFIX + "files.txt").decode().splitlines())
        tree = {}
        for name in files:
            if name not in OVERLAYS and public_path(name, listed):
                mode, blob = regular_blob(SOURCE, files[name])
                tree[name] = mode, project(name, blob)
        for name, template in OVERLAYS.items():
            path = PREFIX + template
            if path in files:
                blob = contents(path)
                if PRIVATE.search(blob.decode()):
                    raise ValueError(f"Private marker in public template {template}")
                tree[name] = (b"100644", blob)
        managed = set(state.get("paths", []))
        existing = set(files_at(dest, head)) if head else set()
        if not accepted and (existing - managed) & tree.keys():
            raise ValueError("A public-only path would be overwritten")
        updates = {name: None for name in managed - tree.keys()}
        oid = updated_tree(dest, head, {**updates, **tree})
        if not head or oid != git(dest, "rev-parse", "HEAD^{tree}").strip():
            args = ["-p", head] if head else []
            title = "Update public source" if head else "Initial public source"
            new = git(dest, "commit-tree", oid, *args, "-m", title).decode().strip()
            old_tree = head or git(dest, "mktree", data=b"").strip()
            git(dest, "read-tree", "-u", "-m", old_tree, new)
            git(dest, "update-ref", "refs/heads/main", new, head or "0" * 40)
            head = new
            print("Public source commit", head[:12])
        write_json(state_path, {"base": head, "paths": sorted(tree)})
        if accepted:
            marker.unlink()
        if config(SOURCE, "charloom.publicPush") == "true":
            git(dest, "push", "origin", "refs/heads/main:refs/heads/main")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path, nargs="?")
    parser.add_argument("--branch")
    opts = parser.parse_args()
    try:
        sync(
            (opts.destination or configured_destination()).resolve(),
            opts.branch or config(SOURCE, "charloom.publicBranch") or "mh-registration",
        )
    except (KeyError, OSError, ValueError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"Public sync: {error}\n")
