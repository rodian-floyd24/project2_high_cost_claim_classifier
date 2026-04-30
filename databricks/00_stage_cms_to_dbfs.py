"""Stage CMS DE-SynPUF Sample 1 files into a Unity Catalog Volume."""

from __future__ import annotations

import hashlib
import json
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import urljoin
from urllib.request import urlretrieve

import requests


VOLUME_PROJECT_ROOT = "/Volumes/workspace/default/project2_high_cost_claim_classifier"
VOLUME_PLANNING_ROOT = f"{VOLUME_PROJECT_ROOT}/planning"
VOLUME_STORAGE_ROOT = f"{VOLUME_PROJECT_ROOT}/object_storage"


SOURCE_INVENTORY = {
    "dataset_name": "CMS DE-SynPUF Sample 1",
    "source_dataset_version": "DE1.0 Sample 1",
    "sample_page_url": "https://www.cms.gov/Research-Statistics-Data-and-Systems/Downloadable-Public-Use-Files/SynPUFs/DESample01.html",
    "default_read_options": {"format": "csv", "header": True, "sep": ","},
    "files": [
        {
            "entity": "beneficiary_summary",
            "logical_name": "beneficiary_2008",
            "display_name": "DE1.0 Sample 1 2008 Beneficiary Summary File (ZIP)",
            "preferred_href_substring": "de1_0_2008_beneficiary_summary_file_sample_1.zip",
            "expected_extracted_name": "DE1_0_2008_Beneficiary_Summary_File_Sample_1.csv",
            "expected_row_count": 116352,
            "expected_join_key": "DESYNPUF_ID",
            "expected_date_column": None,
            "expected_cost_column": "MEDREIMB_IP",
        },
        {
            "entity": "beneficiary_summary",
            "logical_name": "beneficiary_2009",
            "display_name": "DE1.0 Sample 1 2009 Beneficiary Summary File (ZIP)",
            "preferred_href_substring": "de1_0_2009_beneficiary_summary_file_sample_1.zip",
            "expected_extracted_name": "DE1_0_2009_Beneficiary_Summary_File_Sample_1.csv",
            "expected_row_count": 114538,
            "expected_join_key": "DESYNPUF_ID",
            "expected_date_column": None,
            "expected_cost_column": "MEDREIMB_IP",
        },
        {
            "entity": "beneficiary_summary",
            "logical_name": "beneficiary_2010",
            "display_name": "DE1.0 Sample 1 2010 Beneficiary Summary File (ZIP)",
            "preferred_href_substring": "de1_0_2010_beneficiary_summary_file_sample_20.zip",
            "expected_extracted_name": "DE1_0_2010_Beneficiary_Summary_File_Sample_1.csv",
            "expected_row_count": 112754,
            "expected_join_key": "DESYNPUF_ID",
            "expected_date_column": None,
            "expected_cost_column": "MEDREIMB_IP",
        },
        {
            "entity": "inpatient_claims",
            "logical_name": "inpatient_2008_2010",
            "display_name": "DE1.0 Sample 1 2008-2010 Inpatient Claims (ZIP)",
            "preferred_href_substring": "de1_0_2008_to_2010_inpatient_claims_sample_1.zip",
            "expected_extracted_name": "DE1_0_2008_to_2010_Inpatient_Claims_Sample_1.csv",
            "expected_row_count": 66773,
            "expected_join_key": "DESYNPUF_ID",
            "expected_date_column": "CLM_THRU_DT",
            "expected_cost_column": "CLM_PMT_AMT",
        },
        {
            "entity": "outpatient_claims",
            "logical_name": "outpatient_2008_2010",
            "display_name": "DE1.0 Sample 1 2008-2010 Outpatient Claims (ZIP)",
            "preferred_href_substring": "de1_0_2008_to_2010_outpatient_claims_sample_1.zip",
            "expected_extracted_name": "DE1_0_2008_to_2010_Outpatient_Claims_Sample_1.csv",
            "expected_row_count": 790790,
            "expected_join_key": "DESYNPUF_ID",
            "expected_date_column": "CLM_THRU_DT",
            "expected_cost_column": "CLM_PMT_AMT",
        },
        {
            "entity": "carrier_claims",
            "logical_name": "carrier_2008_2010_a",
            "display_name": "DE1.0 Sample 1 2008-2010 Carrier Claims 1",
            "preferred_href_substring": "DE1_0_2008_to_2010_Carrier_Claims_Sample_1A.zip",
            "expected_extracted_name": "DE1_0_2008_to_2010_Carrier_Claims_Sample_1A.csv",
            "expected_row_count": 2370667,
            "expected_join_key": "DESYNPUF_ID",
            "expected_date_column": "CLM_THRU_DT",
            "expected_cost_column": "LINE_NCH_PMT_AMT",
        },
        {
            "entity": "carrier_claims",
            "logical_name": "carrier_2008_2010_b",
            "display_name": "DE1.0 Sample 1 2008-2010 Carrier Claims 2",
            "preferred_href_substring": "DE1_0_2008_to_2010_Carrier_Claims_Sample_1B.zip",
            "expected_extracted_name": "DE1_0_2008_to_2010_Carrier_Claims_Sample_1B.csv",
            "expected_row_count": 2370668,
            "expected_join_key": "DESYNPUF_ID",
            "expected_date_column": "CLM_THRU_DT",
            "expected_cost_column": "LINE_NCH_PMT_AMT",
        },
        {
            "entity": "prescription_drug_events",
            "logical_name": "pde_2008_2010",
            "display_name": "DE1.0 Sample 1 2008-2010 Prescription Drug Events",
            "preferred_href_substring": "DE1_0_2008_to_2010_Prescription_Drug_Events_Sample_1.zip",
            "expected_extracted_name": "DE1_0_2008_to_2010_Prescription_Drug_Events_Sample_1.csv",
            "expected_row_count": 5552421,
            "expected_join_key": "DESYNPUF_ID",
            "expected_date_column": "SRVC_DT",
            "expected_cost_column": "TOT_RX_CST_AMT",
        },
    ],
}


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
        if tag == "a":
            self._current_href = dict(attrs).get("href")
            self._current_text = []

    def handle_data(self, data: str) -> None:
        if self._current_href is not None:
            self._current_text.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._current_href is not None:
            text = "".join(self._current_text).strip()
            if text:
                self.links.append(DownloadLink(text=text, href=self._current_href))
            self._current_href = None
            self._current_text = []


def resolve_download_links(sample_page_url: str) -> dict[str, str]:
    response = requests.get(sample_page_url, timeout=60)
    response.raise_for_status()
    parser = LinkExtractor()
    parser.feed(response.text)
    return {link.text: urljoin(sample_page_url, link.href) for link in parser.links}


def resolve_download_target(entry: dict[str, Any], links: dict[str, str]) -> tuple[str, str]:
    href_hint = entry.get("preferred_href_substring")
    if href_hint:
        matches = [(text, href) for text, href in links.items() if href_hint in href]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise ValueError(f"Multiple href matches for {entry['logical_name']}")
    display_name = entry["display_name"]
    if display_name in links:
        return display_name, links[display_name]
    raise KeyError(f"Could not resolve download link for {display_name}")


def compute_checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_zip_to_volume(source_path: Path, extraction_root: Path) -> list[Path]:
    extraction_root.mkdir(parents=True, exist_ok=True)
    extracted_paths: list[Path] = []
    with zipfile.ZipFile(source_path) as archive:
        for member in archive.namelist():
            if member.endswith("/"):
                continue
            archive.extract(member, extraction_root)
            extracted_paths.append(extraction_root / member)
    return extracted_paths


def main() -> None:
    Path(VOLUME_PLANNING_ROOT).mkdir(parents=True, exist_ok=True)
    Path(f"{VOLUME_STORAGE_ROOT}/bronze").mkdir(parents=True, exist_ok=True)

    inventory_path = Path(VOLUME_PLANNING_ROOT) / "cms_source_inventory.json"
    inventory_path.write_text(json.dumps(SOURCE_INVENTORY, indent=2))

    links = resolve_download_links(SOURCE_INVENTORY["sample_page_url"])
    manifest_records = []
    exception_records = []

    for entry in SOURCE_INVENTORY["files"]:
        try:
            matched_link_text, source_url = resolve_download_target(entry, links)
            zip_name = Path(source_url).name or f"{entry['logical_name']}.zip"
            raw_dir = Path(f"{VOLUME_STORAGE_ROOT}/bronze/{entry['entity']}/raw")
            extracted_dir = Path(f"{VOLUME_STORAGE_ROOT}/bronze/{entry['entity']}/extracted")
            raw_dir.mkdir(parents=True, exist_ok=True)
            extracted_dir.mkdir(parents=True, exist_ok=True)

            volume_zip_path = raw_dir / zip_name
            response = requests.get(source_url, timeout=120, stream=True)
            response.raise_for_status()
            with volume_zip_path.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        handle.write(chunk)

            extracted_volume_paths = extract_zip_to_volume(volume_zip_path, extracted_dir)
            extracted_dbfs_paths = [str(path) for path in extracted_volume_paths]

            manifest_records.append(
                {
                    "entity": entry["entity"],
                    "logical_name": entry["logical_name"],
                    "matched_link_text": matched_link_text,
                    "source_file_name": zip_name,
                    "source_url": source_url,
                    "target_bronze_path": str(volume_zip_path),
                    "ingested_at_utc": datetime.now(timezone.utc).isoformat(),
                    "checksum": compute_checksum(volume_zip_path),
                    "source_dataset_version": SOURCE_INVENTORY["source_dataset_version"],
                    "expected_row_count": entry.get("expected_row_count"),
                    "expected_join_key": entry.get("expected_join_key"),
                    "expected_date_column": entry.get("expected_date_column"),
                    "expected_cost_column": entry.get("expected_cost_column"),
                    "expected_extracted_name": entry.get("expected_extracted_name"),
                    "extracted_paths": extracted_dbfs_paths,
                }
            )
            print(f"staged {entry['logical_name']} -> {volume_zip_path}")
        except Exception as exc:
            exception_records.append(
                {
                    "logical_name": entry["logical_name"],
                    "display_name": entry["display_name"],
                    "preferred_href_substring": entry.get("preferred_href_substring"),
                    "error": str(exc),
                }
            )
            print(f"failed {entry['logical_name']}: {exc}")

    (Path(f"{VOLUME_STORAGE_ROOT}/bronze") / "cms_ingestion_manifest.json").write_text(
        json.dumps(manifest_records, indent=2)
    )
    (Path(f"{VOLUME_STORAGE_ROOT}/bronze") / "cms_ingestion_exceptions.json").write_text(
        json.dumps(exception_records, indent=2)
    )
    print(f"manifest_records={len(manifest_records)} exceptions={len(exception_records)}")


if __name__ == "__main__":
    main()
