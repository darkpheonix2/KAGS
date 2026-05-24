#!/usr/bin/env python3
"""Fix import order and indentation after secret refactor."""

from pathlib import Path

RAGS_ROOT = Path(__file__).resolve().parents[1]

PATH_BLOCK = """import sys
from pathlib import Path

_RAGS_ROOT = Path(__file__).resolve().parents[1]
if str(_RAGS_ROOT) not in sys.path:
    sys.path.insert(0, str(_RAGS_ROOT))
"""

WRONG_ORDER = """from db_config import setup_hf_token

import sys
from pathlib import Path
_RAGS_ROOT = Path(__file__).resolve().parents[1]
if str(_RAGS_ROOT) not in sys.path:
    sys.path.insert(0, str(_RAGS_ROOT))

setup_hf_token()
"""

RIGHT_ORDER = PATH_BLOCK + """
from db_config import setup_hf_token

setup_hf_token()
"""


def fix_file(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    original = text

    if WRONG_ORDER in text:
        text = text.replace(WRONG_ORDER, RIGHT_ORDER)

    # Fix dedented setup_hf_token inside if __name__ blocks
    text = text.replace(
        "    from db_config import setup_hf_token\n\nsetup_hf_token()\n",
        "    from db_config import setup_hf_token\n    setup_hf_token()\n",
    )

    # kge.py main block: setup + clients
    text = text.replace(
        "    from db_config import setup_hf_token\n    setup_hf_token()\n"
        "    from db_config import get_weaviate_client, get_neo4j_driver\n",
        "    from db_config import setup_hf_token, get_weaviate_client, get_neo4j_driver\n"
        "    setup_hf_token()\n",
    )

    # NaturalQA/TriviaQA kgr stray comment
    text = text.replace(
        'triplet_database_name="Triplets_phi4" \n\n    # Instance 1\nfrom db_config',
        'triplet_database_name="Triplets_phi4"\n\nfrom db_config',
    )
    text = text.replace(
        "driver = get_neo4j_driver()\n# Best practice: store your credentials in environment variables\nfrom db_config",
        "driver = get_neo4j_driver()\n\nfrom db_config",
    )

    if text != original:
        path.write_text(text, encoding="utf-8")
        return True
    return False


def main():
    fixed = []
    for path in RAGS_ROOT.rglob("*.py"):
        if path.parts[-2] == "scripts":
            continue
        if fix_file(path):
            fixed.append(path.relative_to(RAGS_ROOT))
    print(f"Fixed {len(fixed)} files")
    for p in sorted(fixed):
        print(f"  - {p}")


if __name__ == "__main__":
    main()
