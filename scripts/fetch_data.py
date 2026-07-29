#!/usr/bin/env python3
"""Download, verify, and safely install the external PVplotHub data archive."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import sys
import tempfile
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from zipfile import BadZipFile, ZipFile


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
CACHE_DIR = ROOT / ".cache" / "data"
DEFAULT_CONFIG = ROOT / "config" / "data-source.json"
CHUNK_SIZE = 8 * 1024 * 1024


class DataError(RuntimeError):
    """Raised when an archive or its configuration is not usable."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download the configured external data archive into ./data."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Path to data-source.json (default: config/data-source.json).",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Only check that the installed data contains the required paths.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing data/ directory after a successful extraction.",
    )
    parser.add_argument(
        "--keep-archive",
        action="store_true",
        help="Keep the verified ZIP in .cache/data after installation.",
    )
    return parser.parse_args()


def format_bytes(size: int | None) -> str:
    if size is None:
        return "unknown size"
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.1f} {unit}"
        value /= 1024
    raise AssertionError("unreachable")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def load_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise DataError(f"data configuration not found: {path}")
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DataError(f"invalid JSON in {path}: {exc}") from exc

    if not isinstance(config, dict):
        raise DataError("the data configuration must be a JSON object")

    required_strings = (
        "dataset_version",
        "doi",
        "archive_url",
        "archive_filename",
        "archive_sha256",
        "archive_format",
        "archive_root",
    )
    for key in required_strings:
        if not isinstance(config.get(key), str) or not config[key].strip():
            raise DataError(f"configuration field {key!r} must be a non-empty string")
    if not isinstance(config.get("archive_size_bytes"), int) or config["archive_size_bytes"] < 0:
        raise DataError("configuration field 'archive_size_bytes' must be a non-negative integer")
    if config["archive_format"] != "zip":
        raise DataError("only ZIP archives are supported; set archive_format to 'zip'")
    if Path(config["archive_filename"]).name != config["archive_filename"]:
        raise DataError("archive_filename must be a filename, not a path")

    archive_root = safe_relative_path(config["archive_root"], "archive_root")
    if len(archive_root.parts) != 1:
        raise DataError("archive_root must be one top-level directory, normally 'data'")
    config["archive_root"] = archive_root.as_posix()

    required_paths = config.get("required_paths")
    if not isinstance(required_paths, list) or not required_paths:
        raise DataError("configuration field 'required_paths' must be a non-empty list")
    checked_paths: list[str] = []
    for value in required_paths:
        if not isinstance(value, str):
            raise DataError("every required_paths entry must be a string")
        checked_paths.append(safe_relative_path(value, "required_paths entry").as_posix())
    config["required_paths"] = checked_paths
    return config


def safe_relative_path(value: str, label: str) -> PurePosixPath:
    normalized = value.replace("\\", "/")
    path = PurePosixPath(normalized)
    if not normalized or path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise DataError(f"{label} must be a safe relative path: {value!r}")
    return path


def assert_download_configured(config: dict[str, Any]) -> None:
    doi = config["doi"].strip()
    url = config["archive_url"]
    digest = config["archive_sha256"].lower()
    normalized_doi = doi.removeprefix("https://doi.org/").removeprefix("doi:")
    if "REPLACE_" in doi or not re.fullmatch(r"10\.\d{4,9}/\S+", normalized_doi, flags=re.IGNORECASE):
        raise DataError("doi must be the published version DOI, for example 10.5281/zenodo.1234567")
    if "REPLACE_" in url or not url.startswith(("https://", "http://")):
        raise DataError(
            "data hosting is not configured. Publish the ZIP, then set archive_url and "
            "archive_sha256 in config/data-source.json. See README.md."
        )
    if "REPLACE_" in digest or len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise DataError("archive_sha256 must be the 64-character SHA-256 printed by package_data.py")
    if config["archive_size_bytes"] <= 0:
        raise DataError("archive_size_bytes must be the positive ZIP size printed by package_data.py")
    config["archive_sha256"] = digest


def missing_required_paths(base_dir: Path, config: dict[str, Any]) -> list[str]:
    return [path for path in config["required_paths"] if not (base_dir / path).exists()]


def check_installed_data(config: dict[str, Any]) -> int:
    if not DATA_DIR.is_dir():
        print(f"Data directory is missing: {DATA_DIR}", file=sys.stderr)
        return 1
    missing = missing_required_paths(DATA_DIR, config)
    if missing:
        print("Data directory is incomplete. Missing required paths:", file=sys.stderr)
        for path in missing:
            print(f"  - data/{path}", file=sys.stderr)
        return 1
    print(f"Data check passed: {DATA_DIR}")
    return 0


def download_archive(config: dict[str, Any]) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    archive_path = CACHE_DIR / config["archive_filename"]
    partial_path = archive_path.with_name(f"{archive_path.name}.part")
    expected_digest = config["archive_sha256"]

    if archive_path.exists():
        print(f"Verifying cached archive: {archive_path}")
        if sha256_file(archive_path) == expected_digest:
            return archive_path
        print("Cached archive does not match the configured SHA-256; downloading it again.")
        archive_path.unlink()

    offset = partial_path.stat().st_size if partial_path.exists() else 0
    headers = {"User-Agent": "PVplotHub-data-fetcher/1.0"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
        print(f"Resuming download at {format_bytes(offset)}...")
    else:
        print("Downloading data archive...")

    try:
        request = Request(config["archive_url"], headers=headers)
        with urlopen(request, timeout=60) as response:
            status = response.getcode()
            mode = "ab" if offset and status == 206 else "wb"
            if offset and status != 206:
                print("The host did not accept a ranged request; restarting the download.")
                offset = 0

            content_length = response.headers.get("Content-Length")
            response_size = int(content_length) if content_length and content_length.isdigit() else None
            total_size = offset + response_size if response_size is not None else None
            downloaded = offset
            last_report = downloaded
            with partial_path.open(mode) as handle:
                while chunk := response.read(CHUNK_SIZE):
                    handle.write(chunk)
                    downloaded += len(chunk)
                    if downloaded - last_report >= 32 * 1024 * 1024:
                        if total_size is None:
                            progress = format_bytes(downloaded)
                        else:
                            progress = f"{format_bytes(downloaded)} / {format_bytes(total_size)}"
                        print(f"\rDownloaded {progress}", end="", flush=True)
                        last_report = downloaded
            if downloaded != offset:
                print()
    except HTTPError as exc:
        if exc.code == 416 and partial_path.exists():
            if sha256_file(partial_path) == expected_digest:
                partial_path.replace(archive_path)
                return archive_path
            partial_path.unlink()
            print("The partial download is stale; restarting it from byte zero.")
            return download_archive(config)
        raise DataError(f"download failed with HTTP {exc.code}: {exc.reason}") from exc
    except URLError as exc:
        raise DataError(f"download failed: {exc.reason}") from exc

    print("Verifying SHA-256...")
    if partial_path.stat().st_size != config["archive_size_bytes"]:
        partial_path.unlink(missing_ok=True)
        raise DataError(
            "downloaded archive size does not match archive_size_bytes; the remote file or configuration may be wrong"
        )
    if sha256_file(partial_path) != expected_digest:
        partial_path.unlink(missing_ok=True)
        raise DataError(
            "downloaded archive does not match archive_sha256; the remote file or configuration may be wrong"
        )
    partial_path.replace(archive_path)
    return archive_path


def zip_member_path(name: str, archive_root: str) -> PurePosixPath:
    path = safe_relative_path(name, "archive member")
    if path.parts[0] != archive_root:
        raise DataError(
            f"archive member is outside the expected {archive_root!r} directory: {name!r}"
        )
    return path


def is_zip_symlink(info: Any) -> bool:
    mode = info.external_attr >> 16
    return stat.S_ISLNK(mode)


def extract_archive(archive_path: Path, config: dict[str, Any], force: bool) -> None:
    temporary_root = Path(tempfile.mkdtemp(prefix=".data-extract-", dir=ROOT))
    try:
        with ZipFile(archive_path) as archive:
            for info in archive.infolist():
                target_path = zip_member_path(info.filename, config["archive_root"])
                if is_zip_symlink(info):
                    raise DataError(f"archive contains a symbolic link, which is not allowed: {info.filename!r}")
                target = temporary_root.joinpath(*target_path.parts)
                if info.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(info) as source, target.open("wb") as destination:
                    shutil.copyfileobj(source, destination, length=CHUNK_SIZE)

        extracted_data = temporary_root / config["archive_root"]
        if not extracted_data.is_dir():
            raise DataError(f"archive does not contain the expected {config['archive_root']!r} directory")
        missing = missing_required_paths(extracted_data, config)
        if missing:
            formatted = ", ".join(missing)
            raise DataError(f"archive extraction is incomplete; missing: {formatted}")

        if DATA_DIR.exists() or DATA_DIR.is_symlink():
            if not force:
                raise DataError("data/ already exists; use --force only when it is safe to replace it")
            print("Replacing existing data/ directory after successful verification...")
            if DATA_DIR.is_symlink() or DATA_DIR.is_file():
                DATA_DIR.unlink()
            else:
                shutil.rmtree(DATA_DIR)

        extracted_data.replace(DATA_DIR)
    except BadZipFile as exc:
        raise DataError(f"invalid ZIP archive: {archive_path}") from exc
    finally:
        shutil.rmtree(temporary_root, ignore_errors=True)


def main() -> int:
    args = parse_args()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    try:
        config = load_config(config_path)
        if args.check:
            return check_installed_data(config)

        if DATA_DIR.exists() or DATA_DIR.is_symlink():
            installed_status = check_installed_data(config)
            if installed_status == 0 and not args.force:
                print("Nothing to download: data/ is already installed. Use --force to replace it.")
                return 0
            if not args.force:
                raise DataError("data/ already exists but is incomplete; inspect it or rerun with --force")

        assert_download_configured(config)
        archive_path = download_archive(config)
        extract_archive(archive_path, config, force=args.force)
        if not args.keep_archive:
            archive_path.unlink(missing_ok=True)
        return check_installed_data(config)
    except (DataError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
