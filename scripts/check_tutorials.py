"""Validate repository-local links in Markdown documentation."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from urllib.parse import unquote


MARKDOWN_LINK = re.compile(r"(?<!!)\[[^\]]*\]\(([^)]+)\)")
HTML_SOURCE = re.compile(r"(?:src|href)=[\"']([^\"']+)[\"']")
EXTERNAL_PREFIXES = ("http://", "https://", "mailto:", "data:")


def documentation_files(root: Path) -> list[Path]:
    files = list(root.glob("*.md"))
    files.extend((root / "docs").rglob("*.md"))
    files.extend((root / "examples").rglob("*.md"))
    return sorted(set(path.resolve() for path in files))


def local_targets(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    return MARKDOWN_LINK.findall(text) + HTML_SOURCE.findall(text)


def validate_links(root: Path) -> tuple[int, list[str]]:
    checked = 0
    errors: list[str] = []
    for document in documentation_files(root):
        for raw_target in local_targets(document):
            target = raw_target.strip().split(maxsplit=1)[0].strip("<>")
            if not target or target.startswith("#") or target.startswith(EXTERNAL_PREFIXES):
                continue
            target_path = unquote(target.split("#", 1)[0])
            if not target_path:
                continue
            checked += 1
            resolved = (document.parent / target_path).resolve()
            try:
                resolved.relative_to(root)
            except ValueError:
                errors.append(f"{document.relative_to(root)}: link escapes repository: {target}")
                continue
            if not resolved.exists():
                errors.append(f"{document.relative_to(root)}: missing target: {target}")
    return checked, errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    checked, errors = validate_links(root)
    if errors:
        raise SystemExit("\n".join(errors))
    print(f"Documentation links OK: {checked} local targets checked")


if __name__ == "__main__":
    main()
