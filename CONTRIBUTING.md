# Contributing

This repository is an audited reproducibility package. Changes should improve
usability without weakening the distinction between supplied evidence, fresh
runs, pilots, and blocked targets.

## Before changing code

1. Read `REPRO_STATUS.md`, `docs/claims_to_code.md`, and the relevant metric or
   experiment contract.
2. Identify whether the target is reusable package code, a formal config,
   supplied paper data, or a historical audit script.
3. Keep formal configuration validation strict. New exploratory settings must
   default off or live in a clearly non-formal config.

## Required checks

```bash
python -m compileall -q src scripts tests examples
pytest -q
python scripts/validate_reference_data.py
python scripts/check_tutorials.py
```

Run the CPU smoke when changing models, protocols, checkpointing, aggregation,
or plotting. A full formal sweep is not required for documentation-only changes.

## Data and result rules

- Do not overwrite files under `paper_data/` with fresh or synthetic outputs.
- Keep custom outputs under `outputs/` and custom plots outside
  `figures/paper_data/`.
- Never fill a blocked target with fabricated or inferred numerical data.
- Preserve negative controls and label smoke/pilot results explicitly.
- Do not commit private checkpoints, credentials, machine-specific cache paths,
  or large unreviewed run directories.

## Documentation

Commands must run from the repository root. State expected outputs and the
boundary of what each check proves. Run `python scripts/check_tutorials.py`
after changing local links.

## License status

The supplied source archives did not contain an approved software license.
Follow `LICENSE_PENDING.md`; do not assume that contribution or redistribution
permission has been granted.
