#!/usr/bin/env python3
"""Create the distributable ZIP archive for PVplotHub's external data."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import sys
from zipfile import ZIP_DEFLATED, ZipFile


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config" / "data-source.json"


class PackageError(RuntimeError):
    """Raised when the local data directory cannot produce a valid release."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Package the local data/ directory into a portable ZIP archive."
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output ZIP path, relative to the repository root unless absolute.",
    )
    parser.add_argument(
        "--version",
        default="v1.0.0",
        help="Dataset version written into the configuration snippet.",
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=ROOT / "data",
        help="Data directory to package (default: ./data).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Data configuration that defines required_paths (default: config/data-source.json).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output archive.",
    )
    parser.add_argument(
        "--compression-level",
        type=int,
        choices=range(0, 10),
        default=6,
        metavar="0-9",
        help="ZIP DEFLATE compression level (default: 6).",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def format_bytes(size: int) -> str:
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.1f} {unit}"
        value /= 1024
    raise AssertionError("unreachable")


def iter_source_files(source: Path) -> list[Path]:
    symlinks = [path for path in source.rglob("*") if path.is_symlink()]
    if symlinks:
        example = symlinks[0].relative_to(source)
        raise PackageError(f"refusing to package symbolic links (first found: data/{example})")
    files = [
        path
        for path in source.rglob("*")
        if path.is_file() and path.name != ".DS_Store"
    ]
    return sorted(files, key=lambda path: path.relative_to(source).as_posix())


def safe_relative_path(value: str) -> PurePosixPath:
    normalized = value.replace("\\", "/")
    path = PurePosixPath(normalized)
    if not normalized or path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise PackageError(f"required_paths contains an unsafe path: {value!r}")
    return path


def load_required_paths(config_path: Path) -> list[str]:
    if not config_path.is_file():
        raise PackageError(f"data configuration not found: {config_path}")
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise PackageError(f"invalid JSON in {config_path}: {exc}") from exc
    required_paths = config.get("required_paths") if isinstance(config, dict) else None
    if not isinstance(required_paths, list) or not required_paths:
        raise PackageError("config required_paths must be a non-empty list")
    paths: list[str] = []
    for value in required_paths:
        if not isinstance(value, str):
            raise PackageError("every config required_paths entry must be a string")
        paths.append(safe_relative_path(value).as_posix())
    return paths


def validate_required_paths(source: Path, required_paths: list[str]) -> None:
    missing = [path for path in required_paths if not (source / path).is_file()]
    if missing:
        preview = "\n  - ".join(missing[:10])
        suffix = "\n  ..." if len(missing) > 10 else ""
        raise PackageError(f"data is incomplete; required files are missing:\n  - {preview}{suffix}")


def main() -> int:
    args = parse_args()
    source = args.source if args.source.is_absolute() else ROOT / args.source
    output = args.output if args.output.is_absolute() else ROOT / args.output
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    source = source.resolve()
    output = output.resolve()

    if source.name != "data":
        print("error: --source must name a directory called data so the archive layout stays stable.", file=sys.stderr)
        return 2
    if not source.is_dir():
        print(f"error: data directory not found: {source}", file=sys.stderr)
        return 2
    if output.suffix.lower() != ".zip":
        print("error: --output must end in .zip", file=sys.stderr)
        return 2
    if output.exists() and not args.overwrite:
        print(f"error: output already exists: {output} (use --overwrite to replace it)", file=sys.stderr)
        return 2
    try:
        output.relative_to(source)
    except ValueError:
        pass
    else:
        print("error: --output must be outside data/", file=sys.stderr)
        return 2

    try:
        required_paths = load_required_paths(config_path)
        validate_required_paths(source, required_paths)
        files = iter_source_files(source)
    except PackageError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if not files:
        print(f"error: no files found in {source}", file=sys.stderr)
        return 2

    total_source_size = sum(path.stat().st_size for path in files)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output.with_name(f".{output.name}.tmp")
    if temporary_output.exists():
        temporary_output.unlink()

    print(f"Packaging {len(files)} files ({format_bytes(total_source_size)})...")
    try:
        with ZipFile(
            temporary_output,
            mode="w",
            compression=ZIP_DEFLATED,
            compresslevel=args.compression_level,
            allowZip64=True,
        ) as archive:
            processed_size = 0
            for index, path in enumerate(files, start=1):
                archive_name = path.relative_to(source.parent).as_posix()
                file_size = path.stat().st_size
                progress = (processed_size + file_size) / total_source_size * 100
                print(f"[{index:>3}/{len(files)}] {progress:5.1f}% {archive_name}")
                archive.write(path, arcname=archive_name)
                processed_size += file_size
        shutil.move(temporary_output, output)
    except BaseException:
        temporary_output.unlink(missing_ok=True)
        raise

    archive_size = output.stat().st_size
    print("Calculating SHA-256...")
    digest = sha256_file(output)
    config_snippet = {
        "dataset_version": args.version,
        "doi": "REPLACE_WITH_VERSION_DOI",
        "archive_url": "REPLACE_WITH_ZENODO_DIRECT_FILE_URL",
        "archive_filename": output.name,
        "archive_sha256": digest,
        "archive_size_bytes": archive_size,
        "archive_format": "zip",
        "archive_root": "data",
        "required_paths": required_paths,
    }

    print(f"Created {output} ({format_bytes(archive_size)})")
    print("\nPaste this into config/data-source.json after publishing the ZIP:")
    print(json.dumps(config_snippet, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
