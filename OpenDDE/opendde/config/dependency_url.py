# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
import os
import posixpath
from urllib.parse import urlsplit, urlunsplit

from opendde.config.model_registry import DEFAULT_MODEL_NAME

DEFAULT_CHECKPOINT_FILE = "opendde.pt"
CHECKPOINT_FILES = {DEFAULT_MODEL_NAME: DEFAULT_CHECKPOINT_FILE}

DEPENDENCY_URL_ROOT = os.environ.get(
    "OPENDDE_DEPENDENCY_URL",
    "https://huggingface.co/aurekaresearch/OpenDDE/resolve/main",
).rstrip("/")

COMMON_URL_ROOT = os.environ.get(
    "OPENDDE_COMMON_URL",
    os.environ.get(
        "OPENDDE_DEPENDENCY_URL",
        "https://huggingface.co/aurekaresearch/OpenDDE/resolve/main/common",
    ),
).rstrip("/")


def _root_url(root: str, *parts: str) -> str:
    parsed = urlsplit(root)
    clean_parts = [part.strip("/") for part in parts if part]
    if parsed.scheme and parsed.netloc:
        path = posixpath.join(parsed.path.rstrip("/"), *clean_parts)
        return urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))
    return os.path.join(root, *clean_parts)


def dependency_url(*parts: str) -> str:
    return _root_url(DEPENDENCY_URL_ROOT, *parts)


def common_url(*parts: str) -> str:
    return _root_url(COMMON_URL_ROOT, *parts)


URL = {
    DEFAULT_MODEL_NAME: dependency_url(CHECKPOINT_FILES[DEFAULT_MODEL_NAME]),
    "ccd_components_file": common_url("components.cif"),
    "ccd_components_rdkit_mol_file": common_url("components.cif.rdkit_mol.pkl"),
    # the following files will be used if enable_template is True
    "obsolete_pdbs_path": common_url("obsolete_to_successor.json"),
    "release_dates_path": common_url("release_date_cache.json"),
}


# Sequence databases for local MSA/template search (hmmsearch / nhmmer).
#
# These default to the AlphaFold database v3.0 archives so that local
# protein-template and RNA-MSA search work out of the box. The small databases
# (pdb_seqres ~220MB and rfam ~220MB) can be re-hosted by setting the
# per-database env vars below; the large databases (nt_rna ~75GB,
# rnacentral ~13GB) are best left on a reliable mirror.
# Set OPENDDE_SEARCH_DATABASE_URL to relocate all four under a single root.
SEARCH_DATABASE_URL_ROOT = os.environ.get(
    "OPENDDE_SEARCH_DATABASE_URL",
    "https://storage.googleapis.com/alphafold-databases/v3.0",
).rstrip("/")


def search_database_url(filename: str, env_var: str = "") -> str:
    """Resolve a search-database download URL, honoring a per-database override."""
    if env_var:
        override = os.environ.get(env_var)
        if override:
            return override
    return f"{SEARCH_DATABASE_URL_ROOT}/{filename}"


SEARCH_DATABASE_URL = {
    "pdb_seqres": search_database_url(
        "pdb_seqres_2022_09_28.fasta.zst", "OPENDDE_PDB_SEQRES_URL"
    ),
    "rfam": search_database_url(
        "rfam_14_9_clust_seq_id_90_cov_80_rep_seq.fasta.zst",
        "OPENDDE_RFAM_DB_URL",
    ),
    "nt_rna": search_database_url(
        "nt_rna_2023_02_23_clust_seq_id_90_cov_80_rep_seq.fasta.zst",
        "OPENDDE_NT_RNA_DB_URL",
    ),
    "rnacentral": search_database_url(
        "rnacentral_active_seq_id_90_cov_80_linclust.fasta.zst",
        "OPENDDE_RNACENTRAL_DB_URL",
    ),
}
