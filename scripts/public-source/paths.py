"""Select committed source files and explicitly public assets for the mirror."""
from pathlib import PurePosixPath
import re


SOURCE_EXTENSIONS = {
    ".py", ".ts", ".tsx", ".js", ".mjs", ".css", ".html", ".sh",
    ".toml", ".md", ".yaml", ".yml",
}
SOURCE_ROOTS = {"blender", "scripts", "skills", "viewer", "recipes"}
PRIVATE_PARTS = {
    "private", "history", "histories", "sessions", "rollouts", "rollout_summaries",
    "memories", "memory", "checkpoints", "transcripts", "logs", "notes",
    ".codex", ".claude", ".git", ".cache", "__pycache__", "out", "target",
    "node_modules", "dist",
}


def restricted(name):
    path = PurePosixPath(name.lower())
    return (not path.parts or path.is_absolute() or ".." in path.parts
            or bool(PRIVATE_PARTS.intersection(path.parts))
            or path.name.startswith(".env") or path.name in {"id_rsa", "id_ed25519"}
            or path.suffix in {".key", ".pem", ".p12", ".pfx", ".log"}
            or any(re.search(r"(?:^|[-_.])(memory|memories|history|transcript|checkpoint|rollout|session-log)(?:$|[-_.])", part)
                   for part in path.parts))


def public_path(name, listed):
    if restricted(name):
        return False
    path = PurePosixPath(name)
    english_page = name.removesuffix(".zh-CN.md") + ".md"
    translated_entry = name.endswith(".zh-CN.md") and english_page in listed
    if name.startswith("assets/public/") or name in listed or translated_entry:
        return True
    if "assets" in path.parts or "fixtures" in path.parts or "inputs" in path.parts:
        return False
    if name.startswith("scripts/public-source/"):
        # Configuration and public templates have explicit entries.
        return path.suffix == ".py" or path.name in {"merge-public", "post-commit"}
    if name.startswith("docs/"):
        return path.suffix == ".md"
    if name.startswith("schemas/"):
        return path.suffix == ".json"
    if path.parts[0] in SOURCE_ROOTS:
        return path.suffix in SOURCE_EXTENSIONS
    return False
