# Independent environment verification

Audit date: 2026-08-27.

- The latest restricted project-server source was inspected before consolidation; no experiment implementation newer than the July 31 centered-alignment package was found.
- The source-only audit snapshot contained 138 files and had SHA-256 `B565ED44D9378D7486C1E76F3525A452C0D286E9CE7418EA2E458529686019FC`.
- Relevant source hashes matched the supplied packages for Figure 2, S1, S3, S4, and spectral-rank utilities.
- The S2 difference was limited to default-path handling; the formal algorithm and metadata were unchanged.
- Selected scripts are retained under `experiments/server_latest_audit/` as provenance and are not imported by the core package.

The clean repository was independently checked in a Linux environment with Python 3.12.7. Editable installation, bytecode compilation, 23 tests, paper-data validation, five data-only figure builds, and the four-architecture CPU smoke test all completed successfully. Generated smoke outputs, virtual environments, hostnames, credentials, and internal absolute paths are excluded from version control.
