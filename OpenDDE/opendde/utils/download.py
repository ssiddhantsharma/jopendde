# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
"""Download utilities for OpenDDE."""

import logging
import os
import shutil
import subprocess
import tempfile
import urllib.request
from os.path import exists as opexists
from typing import Any
from urllib.parse import urlsplit

import torch

from opendde.config.dependency_url import CHECKPOINT_FILES, URL

logger = logging.getLogger(__name__)


def progress_callback(block_num: int, block_size: int, total_size: int) -> None:
    """
    Callback for tracking download progress.

    Args:
        block_num: Current block number.
        block_size: Size of each block in bytes.
        total_size: Total file size in bytes.
    """
    downloaded = block_num * block_size
    if total_size <= 0:
        print(f"\rDownloaded {downloaded} bytes", end="", flush=True)
        return

    percent = min(100, downloaded * 100 / total_size)
    bar_length = 30
    filled_length = int(bar_length * percent // 100)
    bar = "=" * filled_length + "-" * (bar_length - filled_length)

    status = f"\r[{bar}] {percent:.1f}%"
    print(status, end="", flush=True)

    if downloaded >= total_size:
        print()


def _decompress_zst(zst_path: str, output_path: str, source_url: str) -> None:
    """Decompress a .zst archive using python-zstandard or the zstd binary."""
    try:
        import zstandard as zstd
    except ImportError:
        zstd = None

    if zstd is not None:
        try:
            with (
                open(zst_path, "rb") as compressed,
                open(output_path, "wb") as output,
            ):
                zstd.ZstdDecompressor().copy_stream(compressed, output)
            return
        except Exception as e:
            if opexists(output_path):
                os.remove(output_path)
            raise RuntimeError(
                f"Failed to decompress .zst archive downloaded from {source_url} "
                f"to {output_path}: {e}"
            ) from e

    zstd_binary = shutil.which("zstd")
    if zstd_binary is None:
        raise RuntimeError(
            f"Downloaded {source_url} is a .zst archive. Install the `zstd` "
            "command or the optional Python `zstandard` package, or download "
            f"and decompress it manually to {output_path}."
        )

    try:
        subprocess.run(
            [zstd_binary, "-d", "-f", "-o", output_path, zst_path], check=True
        )
    except Exception as e:
        if opexists(output_path):
            os.remove(output_path)
        raise RuntimeError(
            f"Failed to decompress .zst archive downloaded from {source_url} "
            f"to {output_path}: {e}"
        ) from e


def _should_decompress_zst(tos_url: str, checkpoint_path: str) -> bool:
    return urlsplit(tos_url).path.endswith(".zst") and not checkpoint_path.endswith(
        ".zst"
    )


def download_from_url(
    tos_url: str, checkpoint_path: str, check_weight: bool = True
) -> None:
    """
    Download a file from URL and optionally verify if it's a valid checkpoint.

    Args:
        tos_url: URL to download from.
        checkpoint_path: Local path to save the downloaded file.
        check_weight: Whether to verify the downloaded file as a valid PyTorch checkpoint.

    Raises:
        RuntimeError: If download or verification fails.
    """
    if _should_decompress_zst(tos_url, checkpoint_path):
        tmp_dir = os.path.dirname(os.path.abspath(checkpoint_path))
        fd, compressed_path = tempfile.mkstemp(suffix=".zst", dir=tmp_dir)
        os.close(fd)
        try:
            urllib.request.urlretrieve(
                tos_url, compressed_path, reporthook=progress_callback
            )
            _decompress_zst(compressed_path, checkpoint_path, tos_url)
        finally:
            if opexists(compressed_path):
                os.remove(compressed_path)
    else:
        urllib.request.urlretrieve(
            tos_url, checkpoint_path, reporthook=progress_callback
        )

    if check_weight:
        try:
            ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            del ckpt
        except Exception as e:
            if opexists(checkpoint_path):
                os.remove(checkpoint_path)
            raise RuntimeError(
                f"Download model checkpoint failed: {e}. Please download "
                f"manually with: wget {tos_url} -O {checkpoint_path}"
            ) from e


def resolve_checkpoint_path(configs: Any) -> str:
    """
    Resolve the checkpoint path from configuration.

    Args:
        configs: Configuration object.

    Returns:
        Full path to the checkpoint file.
    """
    checkpoint_path = configs.get("load_checkpoint_path", "")
    if checkpoint_path:
        return checkpoint_path
    checkpoint_file = CHECKPOINT_FILES.get(
        configs.model_name, f"{configs.model_name}.pt"
    )
    return os.path.join(configs.load_checkpoint_dir, checkpoint_file)


def download_inference_cache(configs: Any) -> None:
    """
    Download necessary data and model checkpoints for inference.

    Args:
        configs: Configuration object containing paths and model names.
    """
    # Download common data cache files
    for cache_name in (
        "ccd_components_file",
        "ccd_components_rdkit_mol_file",
    ):
        cur_cache_fpath = configs["data"][cache_name]
        if not opexists(cur_cache_fpath):
            os.makedirs(os.path.dirname(cur_cache_fpath), exist_ok=True)
            tos_url = URL[cache_name]
            assert os.path.basename(tos_url) == os.path.basename(cur_cache_fpath), (
                f"{cache_name} file name is incorrect, `{tos_url}` and "
                f"`{cur_cache_fpath}`. Please check and try again."
            )
            logger.info(
                f"Downloading data cache from\n {tos_url}...\n to {cur_cache_fpath}"
            )
            download_from_url(tos_url, cur_cache_fpath, check_weight=False)

    # Download template-related cache files if templates are enabled
    if configs.use_template:
        for cache_name in (
            "obsolete_pdbs_path",
            "release_dates_path",
        ):
            cur_cache_fpath = configs["data"]["template"][cache_name]
            if not opexists(cur_cache_fpath):
                os.makedirs(os.path.dirname(cur_cache_fpath), exist_ok=True)
                tos_url = URL[cache_name]
                assert os.path.basename(tos_url) == os.path.basename(cur_cache_fpath), (
                    f"{cache_name} file name is incorrect, `{tos_url}` and "
                    f"`{cur_cache_fpath}`. Please check and try again."
                )
                logger.info(
                    f"Downloading data cache from\n {tos_url}...\n to {cur_cache_fpath}"
                )
                download_from_url(tos_url, cur_cache_fpath, check_weight=False)
            else:
                logger.info(f"{cache_name} already exists at {cur_cache_fpath}")

    # Download model checkpoint if not present
    checkpoint_path = resolve_checkpoint_path(configs)
    checkpoint_dir = os.path.dirname(checkpoint_path) or configs.load_checkpoint_dir

    if not opexists(checkpoint_path):
        if configs.get("load_checkpoint_path", ""):
            raise FileNotFoundError(
                f"Given checkpoint path not exist [{checkpoint_path}]"
            )
        os.makedirs(checkpoint_dir, exist_ok=True)
        tos_url = URL[configs.model_name]
        logger.info(
            f"Downloading model checkpoint from\n {tos_url}...\n to {checkpoint_path}"
        )
        download_from_url(tos_url, checkpoint_path)
