"""
Load Weaviate, Neo4j, and Hugging Face credentials from environment variables.

Copy .env.example to .env in this directory (RAGs/) and set your values.
Scripts in dataset subfolders should add the RAGs root to sys.path before importing:

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from db_config import get_weaviate_client, get_neo4j_driver
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

_RAGS_ROOT = Path(__file__).resolve().parent
load_dotenv(_RAGS_ROOT / ".env")
load_dotenv()


def _require(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise EnvironmentError(
            f"Missing required environment variable '{name}'. "
            f"Create {_RAGS_ROOT / '.env'} from .env.example and set your credentials."
        )
    return value


def get_weaviate_url() -> str:
    return _require("WEAVIATE_URL")


def get_weaviate_api_key() -> str:
    return _require("WEAVIATE_API_KEY")


def get_neo4j_uri() -> str:
    return _require("NEO4J_URI")


def get_neo4j_user() -> str:
    return os.getenv("NEO4J_USER", "neo4j")


def get_neo4j_password() -> str:
    return _require("NEO4J_PASSWORD")


def get_neo4j_auth() -> tuple[str, str]:
    return get_neo4j_user(), get_neo4j_password()


def get_hf_token() -> str:
    """Hugging Face token (HF_TOKEN or HUGGINGFACEHUB_API_TOKEN)."""
    return os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN") or _require("HF_TOKEN")


def setup_hf_token() -> str:
    """Set HUGGINGFACEHUB_API_TOKEN from HF_TOKEN and return it."""
    token = get_hf_token()
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = token
    os.environ.setdefault("HF_TOKEN", token)
    return token


def get_weaviate_client():
    import weaviate
    from weaviate.auth import Auth

    return weaviate.connect_to_weaviate_cloud(
        cluster_url=get_weaviate_url(),
        auth_credentials=Auth.api_key(get_weaviate_api_key()),
    )


def get_neo4j_driver():
    from neo4j import GraphDatabase

    return GraphDatabase.driver(get_neo4j_uri(), auth=get_neo4j_auth())
