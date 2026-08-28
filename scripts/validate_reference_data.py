from __future__ import annotations

import json
from pathlib import Path

from bnc_repro.validation import validate_all


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    result = validate_all(root / "paper_data")
    output = root / "paper_data" / "validation_report.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

