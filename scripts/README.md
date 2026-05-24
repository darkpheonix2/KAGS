# Development scripts

One-off utilities used during the credentials refactor. **Not required** to run KAGS pipelines.

| Script | Purpose |
|--------|---------|
| `refactor_secrets.py` | Replaced hardcoded credentials with `db_config` imports (already applied) |
| `fix_imports.py` | Fixed import order after refactor (already applied) |

Safe to delete from a clone if you only need runtime code (`kgr.py`, `kge.py`, etc.).
