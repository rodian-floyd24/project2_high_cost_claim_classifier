"""Download or stage CMS synthetic claims files into bronze object storage."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import urljoin
from urllib.request import urlretrieve


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INVENTORY = PROJECT_ROOT / "planning" / "cms_source_inventory.json"
DEFAULT_STORAGE_ROOT = PROJECT_ROOT / "object_storage"


@dataclass
class DownloadLink:
    text: str
    href: str


class LinkExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[DownloadLink] = []
        self._current_href: str | None = None
        self._current_text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "a":
            return
        attr_map = dict(attrs)
        self._current_href = attr_map.get("href")
        self._current_text = []

    def handle_data(self, data: str) -> None:
        if self._current_href is not None:
            self._current_text.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag != "a" or self._current_href is None:
            return
        text = "".join(self._current_text).strip()
        if text and self._current_href:
            self.links.append(DownloadLink(text=text, href=self._current_href))
        self._current_href = None
        self._current_text = []


def load_source_inventory(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def resolve_download_links(sample_page_url: str) -> dict[str, str]:
    import requests

    response = requests.get(sample_page_url, timeout=60)
    response.raise_for_status()
    parser = LinkExtractor()
    parser.feed(response.text)
    resolved: dict[str, str] = {}
    for link in parser.links:
        resolved[link.text] = urljoin(sample_page_url, link.href)
    return resolved


def resolve_download_target(entry: dict[str, Any], links: dict[str, str]) -> tuple[str, str]:
    href_hint = entry.get("preferred_href_substring")
    if href_hint:
        href_matches = [(text, href) for text, href in links.items() if href_hint in href]
        if len(href_matches) == 1:
            return href_matches[0]
        if len(href_matches) > 1:
            raise ValueError(
                f"Multiple href matches found for {entry['logical_name']} using hint '{href_hint}'"
            )

    display_name = entry["display_name"]
    if display_name in links:
        return display_name, links[display_name]

    raise KeyError(f"Could not resolve download link for '{display_name}'")


def download_file(url: str, local_tmp_path: Path) -> None:
    local_tmp_path.parent.mkdir(parents=True, exist_ok=True)
    urlretrieve(url, local_tmp_path)


def compute_checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def upload_to_object_storage(local_path: Path, bronze_key: Path) -> Path:
    bronze_key.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(local_path, bronze_key)
    return bronze_key


def extract_if_zip(source_path: Path, extraction_root: Path) -> list[Path]:
    if not zipfile.is_zipfile(source_path):
        return []
    extraction_root.mkdir(parents=True, exist_ok=True)
    extracted_paths: list[Path] = []
    with zipfile.ZipFile(source_path) as archive:
        for member in archive.namelist():
            if member.endswith("/"):
                continue
            archive.extract(member, extraction_root)
            extracted_paths.append(extraction_root / member)
    return extracted_paths


def write_ingestion_manifest(records: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(records, indent=2))


def write_exception_report(records: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(records, indent=2))


def build_manifest_record(
    entry: dict[str, Any],
    matched_link_text: str,
    source_url: str,
    target_bronze_path: Path,
    checksum: str,
    extracted_paths: list[Path],
) -> dict[str, Any]:
    return {
        "entity": entry["entity"],
        "logical_name": entry["logical_name"],
        "matched_link_text": matched_link_text,
        "source_file_name": target_bronze_path.name,
        "source_url": source_url,
        "target_bronze_path": str(target_bronze_path),
        "ingested_at_utc": datetime.now(timezone.utc).isoformat(),
        "checksum": checksum,
        "source_dataset_version": entry.get("source_dataset_version", "DE1.0 Sample 1"),
        "expected_row_count": entry.get("expected_row_count"),
        "expected_join_key": entry.get("expected_join_key"),
        "expected_date_column": entry.get("expected_date_column"),
        "expected_cost_column": entry.get("expected_cost_column"),
        "expected_extracted_name": entry.get("expected_extracted_name"),
        "extracted_paths": [str(path) for path in extracted_paths],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--storage-root", type=Path, default=DEFAULT_STORAGE_ROOT)
    parser.add_argument("--manifest-name", default="cms_ingestion_manifest.json")
    parser.add_argument("--exceptions-name", default="cms_ingestion_exceptions.json")
    args = parser.parse_args()

    inventory = load_source_inventory(args.inventory)
    download_links = resolve_download_links(inventory["sample_page_url"])
    manifest_records: list[dict[str, Any]] = []
    exception_records: list[dict[str, Any]] = []

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_root = Path(tmp_dir)
        for entry in inventory["files"]:
            try:
                matched_link_text, source_url = resolve_download_target(entry, download_links)
                inferred_name = Path(source_url).name or f"{entry['logical_name']}.dat"
                local_tmp_path = tmp_root / inferred_name
                download_file(source_url, local_tmp_path)

                entity_root = args.storage_root / "bronze" / entry["entity"]
                raw_path = upload_to_object_storage(local_tmp_path, entity_root / "raw" / inferred_name)
                extracted_paths = extract_if_zip(raw_path, entity_root / "extracted")

                manifest_records.append(
                    build_manifest_record(
                        entry=entry,
                        matched_link_text=matched_link_text,
                        source_url=source_url,
                        target_bronze_path=raw_path,
                        checksum=compute_checksum(raw_path),
                        extracted_paths=extracted_paths,
                    )
                )
            except Exception as exc:
                exception_records.append(
                    {
                        "logical_name": entry["logical_name"],
                        "display_name": entry["display_name"],
                        "preferred_href_substring": entry.get("preferred_href_substring"),
                        "error": str(exc),
                    }
                )

    write_ingestion_manifest(manifest_records, args.storage_root / "bronze" / args.manifest_name)
    write_exception_report(exception_records, args.storage_root / "bronze" / args.exceptions_name)
    print(f"Wrote ingestion manifest with {len(manifest_records)} records to bronze storage.")
    if exception_records:
        print(f"Wrote exception report with {len(exception_records)} records to bronze storage.")


if __name__ == "__main__":
    main()
