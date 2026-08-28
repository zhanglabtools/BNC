from pathlib import Path

from scripts.check_tutorials import validate_links


def test_repository_markdown_links_resolve() -> None:
    root = Path(__file__).resolve().parents[1]
    checked, errors = validate_links(root)
    assert checked >= 40
    assert errors == []
