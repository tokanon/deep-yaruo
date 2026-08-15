from __future__ import annotations

import argparse
import json
from pathlib import Path

from .yaruyomi import export_entries, index_archive, read_database_stats, recategorize_database


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARCHIVE = ROOT / "datasets" / "incoming" / "yaruyomi" / "v32.1" / "source.zip"
DEFAULT_DATABASE = ROOT / "datasets" / "incoming" / "yaruyomi" / "v32.1" / "index.sqlite3"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Index and selectively export the Yaruyomi MLT archive."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    index = subparsers.add_parser("index", help="Build a compact SQLite index without extracting all AA files.")
    index.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    index.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    index.add_argument("--version", default="v32.1")
    index.add_argument("--max-files", type=int)
    index.add_argument("--force", action="store_true")

    stats = subparsers.add_parser("stats", help="Print index counts and category distribution.")
    stats.add_argument("--database", type=Path, default=DEFAULT_DATABASE)

    recategorize = subparsers.add_parser(
        "recategorize", help="Reapply the current lightweight category rules without rebuilding hashes."
    )
    recategorize.add_argument("--database", type=Path, default=DEFAULT_DATABASE)

    export = subparsers.add_parser(
        "export", help="Export selected records as one text file per AA."
    )
    export.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    export.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    export.add_argument(
        "--output",
        type=Path,
        default=ROOT / "datasets" / "incoming" / "yaruyomi" / "v32.1" / "export",
    )
    export.add_argument("--category")
    export.add_argument("--record-type", default="aa", choices=("aa", "section", "metadata", "text"))
    export.add_argument("--limit", type=int, default=100, help="0 exports every matching record.")
    export.add_argument(
        "--output-encoding",
        choices=("utf-8", "cp932-ncr"),
        default="utf-8",
        help="UTF-8 Unicode, or CP932 with unsupported characters serialized as numeric references.",
    )
    export.add_argument("--include-duplicates", action="store_true")
    sensitivity = export.add_mutually_exclusive_group()
    sensitivity.add_argument(
        "--exclude-sensitive",
        action="store_true",
        help="Optional filter; sensitive records are included by default.",
    )
    sensitivity.add_argument(
        "--sensitive-only",
        action="store_true",
        help="Export only records carrying the provisional sensitive tag.",
    )
    export.add_argument(
        "--thumbnails",
        action="store_true",
        help="Render a Saitamaar PNG preview next to every exported text file.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "index":
        result = index_archive(
            args.archive,
            args.database,
            version=args.version,
            max_files=args.max_files,
            force=args.force,
        )
    elif args.command == "stats":
        result = read_database_stats(args.database)
    elif args.command == "recategorize":
        result = recategorize_database(args.database)
    else:
        result = export_entries(
            args.database,
            args.archive,
            args.output,
            category=args.category,
            record_type=args.record_type,
            include_duplicates=args.include_duplicates,
            exclude_sensitive=args.exclude_sensitive,
            sensitive_only=args.sensitive_only,
            generate_thumbnails=args.thumbnails,
            output_encoding=args.output_encoding,
            limit=args.limit,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
