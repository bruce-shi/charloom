"""Shared Git operations for the private/public source bridge."""

from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import subprocess
import tempfile

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parents[1]
ENV = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
IDENTITY = [
    "-c",
    "user.name=Charloom",
    "-c",
    "user.email=charloom@users.noreply.invalid",
    "-c",
    "commit.gpgSign=false",
    "-c",
    "core.hooksPath=/dev/null",
]

OVERLAYS = {".gitignore": "gitignore.public"}
PREFIX = "scripts/public-source/"


def git(repo, *args, data=None, env=None):
    return subprocess.run(
        ["git", "-C", str(repo), *IDENTITY, *args],
        env=env or ENV,
        input=data,
        stdout=subprocess.PIPE,
        check=True,
    ).stdout


def config(repo, name):
    result = subprocess.run(
        ["git", "-C", str(repo), "config", "--get", name],
        env=ENV,
        stdout=subprocess.PIPE,
    )
    if result.returncode not in (0, 1):
        result.check_returncode()
    return result.stdout.decode().strip()


def configured_destination():
    value = config(SOURCE, "charloom.publicRepo")
    if not value:
        raise ValueError(
            "Set git config --local charloom.publicRepo to the public checkout."
        )
    return Path(value).resolve()


def files_at(repo, commit):
    entries = git(repo, "ls-tree", "-rz", commit).split(b"\0")
    result = {}
    for entry in filter(None, entries):
        fields, name = entry.split(b"\t", 1)
        mode, kind, oid = fields.split()
        result[name.decode()] = (mode, kind, oid)
    return result


def regular_blob(repo, entry):
    mode, kind, oid = entry
    if kind != b"blob" or mode not in (b"100644", b"100755"):
        raise ValueError("Only regular source files can cross the public boundary")
    return mode, git(repo, "cat-file", "blob", oid)


def write_json(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


@contextmanager
def locked_public(dest):
    if not (dest / ".git").is_dir() or config(dest, "charloom.sourceMirror") != "true":
        raise ValueError(
            "The destination must be a public checkout initialized by sync.py"
        )
    with (dest / ".git/source-sync.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if git(dest, "symbolic-ref", "HEAD").strip() != b"refs/heads/main":
            raise ValueError("Check out main in the public repository")
        if git(dest, "status", "--porcelain", "--untracked-files=normal"):
            raise ValueError("Commit or stash the public checkout changes first")
        yield


def updated_tree(repo, base, updates):
    """Build a tree without touching the checkout or its index."""
    with tempfile.TemporaryDirectory() as temp:
        env = dict(ENV, GIT_INDEX_FILE=str(Path(temp) / "index"))
        if base:
            git(repo, "read-tree", base, env=env)
        lines = []
        for name, entry in sorted(updates.items()):
            if entry is None:
                fields = b"0 " + b"0" * 40
            else:
                mode, blob = entry
                oid = git(repo, "hash-object", "-w", "--stdin", data=blob).strip()
                fields = mode + b" " + oid
            lines.append(fields + b"\t" + name.encode() + b"\0")
        git(repo, "update-index", "-z", "--index-info", env=env, data=b"".join(lines))
        return git(repo, "write-tree", env=env).strip()
