import logging
import subprocess
from pathlib import Path

import pandas as pd
import pytest
import numpy as np
from pydicom import Dataset

from liverct.bids import CTBIDSConverter, convert_dicom_directory_to_bids
from liverct.ingestion import build_manifest, inventory_archive, load_config, score_inventory, stage_sourcedata
from liverct.ingestion.review import _build_slice_index, _display_pixels, _generate_montages, _make_thumbnail, generate_review_reports
from liverct.ingestion.tiering import _classify_kernel, _classify_phase, _score_reconstruction_diameter, _score_z_coverage


def _write_dicom(path, series_uid, instance_number, study_date="20200722"):
    import numpy as np
    from pydicom import Dataset, FileDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid

    file_meta = Dataset()
    file_meta.MediaStorageSOPClassUID = generate_uid()
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    dataset = FileDataset(str(path), {}, file_meta=file_meta, preamble=b"\0" * 128)
    dataset.PatientID = "0119838"
    dataset.StudyInstanceUID = "study1"
    dataset.SeriesInstanceUID = series_uid
    dataset.StudyDate = study_date
    dataset.Modality = "CT"
    dataset.SeriesDescription = "ABD"
    dataset.ImageType = ["ORIGINAL", "PRIMARY", "AXIAL"]
    dataset.SeriesNumber = 2
    dataset.InstanceNumber = instance_number
    dataset.Rows = 2
    dataset.Columns = 2
    dataset.BitsAllocated = 16
    dataset.BitsStored = 16
    dataset.HighBit = 15
    dataset.PixelRepresentation = 0
    dataset.SamplesPerPixel = 1
    dataset.PhotometricInterpretation = "MONOCHROME2"
    dataset.ImagePositionPatient = [0, 0, float(instance_number)]
    dataset.PixelData = np.zeros((2, 2), dtype=np.uint16).tobytes()
    dataset.save_as(path)


def _create_archive_subject(archive, series_uid="series1"):
    archive.mkdir(parents=True)
    for index in range(1, 4):
        _write_dicom(archive / "image{}.dcm".format(index), series_uid, index)


def test_bids_conversion_uses_implicit_sourcedata_and_logs_session(tmp_path, monkeypatch, caplog):
    caplog.set_level(logging.INFO)
    bids_root = tmp_path / "bids"
    series = bids_root / "sourcedata" / "sub-001" / "ses-01" / "series-1"
    _create_archive_subject(series)
    converted = []

    def fake_convert(self, dicom_dir, bids_root, subject_id, session_id, config_file, **kwargs):
        converted.append((dicom_dir, bids_root, subject_id, session_id))
        return True

    monkeypatch.setattr("liverct.bids.CTBIDSConverter.convert", fake_convert)

    results = convert_dicom_directory_to_bids(bids_root=bids_root)

    assert results == {"successful": 1, "failed": 0, "skipped": 0}
    assert converted == [(str(series), str(bids_root), "001", "01")]
    assert "Processing subject sub-001, session ses-01" in caplog.text


def test_converter_uses_gdcm_fallback_after_failed_conversion(tmp_path, monkeypatch, caplog):
    caplog.set_level(logging.INFO)
    series = tmp_path / "series-1"
    _create_archive_subject(series)
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        if command[0] == "dcm2bids4ct" and len(calls) == 1:
            raise subprocess.CalledProcessError(1, command, stderr="JPEG decode failed")
        if command[0] == "gdcmconv":
            output_path = Path(command[-1])
            output_path.write_bytes(b"decompressed")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = CTBIDSConverter().convert(
        dicom_dir=str(series),
        bids_root=str(tmp_path / "bids"),
        subject_id="001",
        session_id="01",
    )

    assert result is True
    assert calls[0][0] == "dcm2bids4ct"
    assert sum(command[0] == "gdcmconv" for command in calls) == 3
    assert calls[-1][0] == "dcm2bids4ct"
    assert "GDCM fallback conversion succeeded" in caplog.text


def test_converter_skips_complete_outputs_when_overwrite_disabled(tmp_path, monkeypatch):
    output_dir = tmp_path / "bids" / "sub-001" / "ses-01" / "ct"
    output_dir.mkdir(parents=True)
    (output_dir / "sub-001_ses-01_ct_1.nii.gz").write_bytes(b"nifti")
    (output_dir / "sub-001_ses-01_ct_1.json").write_text("{}", encoding="utf-8")

    def fail_if_called(*args, **kwargs):
        raise AssertionError("dcm2bids4ct should not run for complete outputs")

    monkeypatch.setattr(subprocess, "run", fail_if_called)

    result = CTBIDSConverter().convert(
        dicom_dir=str(tmp_path),
        bids_root=str(tmp_path / "bids"),
        subject_id="001",
        session_id="01",
        overwrite=False,
    )

    assert result is True


def test_inventory_test_mode_limits_top_level_search(tmp_path):
    _create_archive_subject(tmp_path / "0119838" / "GRANULAR" / "CT" / "20200722")
    _create_archive_subject(tmp_path / "0999999" / "GRANULAR" / "CT" / "20200722", "series2")

    inventory = inventory_archive(tmp_path, test_mode=True)
    assert set(inventory["subject_folder"]) == {"0119838"}


def test_inventory_and_staging_filter_series_uid(tmp_path):
    archive = tmp_path / "0119838" / "GRANULAR" / "CT" / "20200722"
    _create_archive_subject(archive)
    _write_dicom(archive / "other.dcm", "series2", 1)

    inventory = inventory_archive(tmp_path)
    assert set(inventory["series_instance_uid"]) == {"series1", "series2"}
    series = inventory[inventory["series_instance_uid"] == "series1"].iloc[0]
    manifest = tmp_path / "manifest.tsv"
    pd.DataFrame([{
        "subject_id": "folder-subject", "patient_id": "patient-0119838", "session_id": "20200722", "series_uid": "series1",
        "source_directory": series["source_directory"],
    }]).to_csv(manifest, sep="\t", index=False)

    sourcedata = stage_sourcedata(manifest, tmp_path, tmp_path / "bids")
    staged = list(sourcedata.rglob("*.dcm"))
    assert len(staged) == 3
    assert all("sub-patient-0119838" in str(path) for path in staged)
    assert all("folder-subject" not in str(path) for path in staged)


def test_yaml_config_overrides_defaults(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("tiering:\n  min_z_extent_mm: 250\n", encoding="utf-8")
    config = load_config(config_path)
    assert config.tiering["min_z_extent_mm"] == 250
    assert config.tiering["min_num_slices"] == 50


def test_scoring_assigns_four_tiers(tmp_path):
    inventory = pd.DataFrame([
        {"subject_folder": "011", "study_instance_uid": "study1", "series_instance_uid": "s1", "modality": "CT", "image_type": "ORIGINAL\\PRIMARY\\AXIAL", "series_description": "ABD", "study_description": "ABDOMEN", "z_extent_mm": "250", "num_slices": "100"},
        {"subject_folder": "011", "study_instance_uid": "study1", "series_instance_uid": "s2", "modality": "CT", "image_type": "ORIGINAL\\PRIMARY", "series_description": "ABD", "study_description": "", "z_extent_mm": "250", "num_slices": "100"},
        {"subject_folder": "011", "study_instance_uid": "study1", "series_instance_uid": "s3", "modality": "CT", "image_type": "LOCALIZER", "series_description": "SCOUT", "study_description": "", "z_extent_mm": "", "num_slices": "1"},
        {"subject_folder": "011", "study_instance_uid": "study1", "series_instance_uid": "s4", "modality": "MR", "image_type": "ORIGINAL", "series_description": "ABD", "study_description": "", "z_extent_mm": "250", "num_slices": "100"},
        {"subject_folder": "011", "study_instance_uid": "study1", "series_instance_uid": "s5", "modality": "CT", "image_type": "ORIGINAL\\PRIMARY\\AXIAL", "series_description": "ABD", "study_description": "ABDOMEN", "z_extent_mm": "250", "num_slices": "2"},
    ])
    source = tmp_path / "inventory.tsv"
    output = tmp_path / "scored.tsv"
    inventory.to_csv(source, sep="\t", index=False)
    scored = score_inventory(source, output)
    assert list(scored["tier"]) == ["Tier 1", "Tier 2", "Tier 4", "Tier 4", "Tier 4"]
    assert scored.loc[scored["series_instance_uid"] == "s5", "reject_short_series"].iloc[0] == 1
    assert scored.loc[scored["series_instance_uid"] == "s5", "tier_reason"].iloc[0] == "series contains only 1 or 2 slices"


def test_scoring_recommends_one_primary_per_study(tmp_path):
    inventory = pd.DataFrame([
        {"subject_folder": "011", "study_instance_uid": "study1", "series_instance_uid": "s1", "study_date": "20200722", "modality": "CT", "image_type": "ORIGINAL\\PRIMARY\\AXIAL", "series_description": "ABD", "study_description": "ABDOMEN", "z_extent_mm": "350", "num_slices": "100"},
        {"subject_folder": "011", "study_instance_uid": "study1", "series_instance_uid": "s2", "study_date": "20200722", "modality": "CT", "image_type": "ORIGINAL\\PRIMARY", "series_description": "VENOUS", "study_description": "ABDOMEN", "z_extent_mm": "250", "num_slices": "100"},
        {"subject_folder": "011", "study_instance_uid": "study2", "series_instance_uid": "s3", "study_date": "20210830", "modality": "CT", "image_type": "LOCALIZER", "series_description": "SCOUT", "study_description": "ABDOMEN", "z_extent_mm": "", "num_slices": "1"},
    ])
    source = tmp_path / "inventory.tsv"
    inventory.to_csv(source, sep="\t", index=False)
    scored = score_inventory(source)
    assert list(scored["recommendation"]) == ["PRIMARY", "SECONDARY", "REJECT"]
    assert scored.loc[0, "study_group_key"] == "|study1"


def test_scoring_selects_one_primary_per_subject_date_across_studies(tmp_path):
    inventory = pd.DataFrame([
        {"subject_folder": "011", "study_instance_uid": "study1", "series_instance_uid": "s1", "study_date": "20200722", "modality": "CT", "image_type": "ORIGINAL\\PRIMARY\\AXIAL", "series_description": "ABD", "study_description": "ABDOMEN", "z_extent_mm": "350", "num_slices": "100"},
        {"subject_folder": "011", "study_instance_uid": "study2", "series_instance_uid": "s2", "study_date": "20200722", "modality": "CT", "image_type": "ORIGINAL\\PRIMARY\\AXIAL", "series_description": "ABD", "study_description": "ABDOMEN", "z_extent_mm": "250", "num_slices": "100"},
        {"subject_folder": "011", "study_instance_uid": "study2", "series_instance_uid": "head", "study_date": "20200722", "modality": "CT", "image_type": "ORIGINAL\\PRIMARY\\AXIAL", "series_description": "HEAD", "study_description": "HEAD", "z_extent_mm": "250", "num_slices": "100"},
    ])
    source = tmp_path / "inventory.tsv"
    inventory.to_csv(source, sep="\t", index=False)
    scored = score_inventory(source)
    assert scored["scan_group_key"].nunique() == 1
    assert list(scored.loc[scored["recommendation"] == "PRIMARY", "series_instance_uid"]) == ["s1"]
    assert set(scored.loc[scored["series_instance_uid"] == "head", "recommendation"]) == {"REJECT"}
    assert set(scored.loc[scored["series_instance_uid"] == "s1", "candidate_status"]) == {"AUTO_PRIMARY"}
    assert scored["is_auto_primary"].sum() == 1


def test_scoring_affirms_no_candidate_for_non_abdominal_scan(tmp_path):
    inventory = pd.DataFrame([
        {"subject_folder": "011", "study_instance_uid": "study1", "series_instance_uid": "head", "study_date": "20200722", "modality": "CT", "image_type": "ORIGINAL\\PRIMARY\\AXIAL", "series_description": "HEAD", "study_description": "HEAD", "z_extent_mm": "250", "num_slices": "100"},
        {"subject_folder": "011", "study_instance_uid": "study1", "series_instance_uid": "scout", "study_date": "20200722", "modality": "CT", "image_type": "LOCALIZER", "series_description": "SCOUT", "study_description": "ABDOMEN", "z_extent_mm": "", "num_slices": "1"},
    ])
    source = tmp_path / "inventory.tsv"
    inventory.to_csv(source, sep="\t", index=False)
    scored = score_inventory(source)
    assert set(scored["candidate_status"]) == {"NO_CANDIDATE"}
    assert set(scored["recommendation"]) == {"REJECT"}
    assert scored["is_auto_primary"].sum() == 0


def test_scoring_uses_study_description_for_anatomy_exclusion(tmp_path):
    inventory = pd.DataFrame([{
        "subject_folder": "011", "study_instance_uid": "study1", "series_instance_uid": "lower-extrem",
        "study_date": "20200722", "modality": "CT", "image_type": "ORIGINAL\\PRIMARY\\AXIAL",
        "series_description": "ROUTINE", "study_description": "CT LOWER EXTREM",
        "z_extent_mm": "350", "num_slices": "100",
    }])
    source = tmp_path / "inventory.tsv"
    inventory.to_csv(source, sep="\t", index=False)

    scored = score_inventory(source)

    row = scored.iloc[0]
    assert row["reject_anatomy"] == 1
    assert row["automatic_candidate"] == 0
    assert row["recommendation"] == "REJECT"


def test_scoring_prefers_standard_reconstruction_and_exposes_components(tmp_path):
    inventory = pd.DataFrame([
        {"subject_folder": "011", "study_instance_uid": "study1", "series_instance_uid": "standard", "study_date": "20200722", "modality": "CT", "image_type": "ORIGINAL\\PRIMARY\\AXIAL", "series_description": "PE CHEST ABDOMEN PELVIS STANDARD VENOUS", "study_description": "ABDOMEN", "z_extent_mm": "650", "num_slices": "130", "reconstruction_diameter": "400", "convolution_kernel": "B30", "slice_thickness": "5"},
        {"subject_folder": "011", "study_instance_uid": "study1", "series_instance_uid": "thin", "study_date": "20200722", "modality": "CT", "image_type": "ORIGINAL\\PRIMARY\\AXIAL", "series_description": "PE CHEST ABDOMEN PELVIS VENOUS", "study_description": "ABDOMEN", "z_extent_mm": "650", "num_slices": "520", "reconstruction_diameter": "400", "convolution_kernel": "B31", "slice_thickness": "1"},
        {"subject_folder": "011", "study_instance_uid": "study1", "series_instance_uid": "bone", "study_date": "20200722", "modality": "CT", "image_type": "ORIGINAL\\PRIMARY\\AXIAL", "series_description": "PE CHEST ABDOMEN PELVIS BONE DELAYED", "study_description": "ABDOMEN", "z_extent_mm": "650", "num_slices": "520", "reconstruction_diameter": "400", "convolution_kernel": "B70", "slice_thickness": "1"},
        {"subject_folder": "011", "study_instance_uid": "study2", "series_instance_uid": "spect", "study_date": "20200722", "modality": "CT", "image_type": "ORIGINAL\\PRIMARY\\AXIAL", "series_description": "SPECT ATTENUATION CT", "study_description": "ABDOMEN", "z_extent_mm": "650", "num_slices": "130", "reconstruction_diameter": "400", "convolution_kernel": "B30", "slice_thickness": "5"},
    ])
    source = tmp_path / "inventory.tsv"
    inventory.to_csv(source, sep="\t", index=False)

    scored = score_inventory(source)

    standard = scored.loc[scored["series_instance_uid"] == "standard"].iloc[0]
    thin = scored.loc[scored["series_instance_uid"] == "thin"].iloc[0]
    bone = scored.loc[scored["series_instance_uid"] == "bone"].iloc[0]
    spect = scored.loc[scored["series_instance_uid"] == "spect"].iloc[0]
    assert standard["recommendation"] == "PRIMARY"
    assert standard["candidate_status"] == "AUTO_PRIMARY"
    assert standard["thickness_score"] == 20
    assert thin["thickness_score"] == 0
    assert bone["kernel_score"] == -50
    assert bone["recommendation"] == "SECONDARY"
    assert spect["nuclear_penalty"] == -40
    assert {"kernel_class", "phase_type", "anatomy_class", "thickness_score", "kernel_score", "phase_score", "coverage_score", "fov_score", "candidate_score"}.issubset(scored.columns)


def test_scoring_excludes_spine_only_studies(tmp_path):
    inventory = pd.DataFrame([{
        "subject_folder": "011", "study_instance_uid": "study1", "series_instance_uid": "spine", "study_date": "20200722", "modality": "CT", "image_type": "ORIGINAL\\PRIMARY\\AXIAL", "series_description": "T SPINE", "study_description": "THORACIC SPINE", "z_extent_mm": "400", "num_slices": "100",
    }])
    source = tmp_path / "inventory.tsv"
    inventory.to_csv(source, sep="\t", index=False)

    scored = score_inventory(source)

    row = scored.iloc[0]
    assert row["anatomy_class"] == "non_torso"
    assert row["automatic_candidate"] == 0
    assert row["candidate_status"] == "NO_CANDIDATE"


def test_scoring_auto_approves_single_candidate_below_score_threshold(tmp_path):
    inventory = pd.DataFrame([{
        "subject_folder": "011", "study_instance_uid": "study1", "series_instance_uid": "only",
        "study_date": "20200722", "modality": "CT", "image_type": "ORIGINAL\\PRIMARY\\AXIAL",
        "series_description": "ABD", "study_description": "ABDOMEN",
        "z_extent_mm": "200", "num_slices": "50",
    }])
    source = tmp_path / "inventory.tsv"
    inventory.to_csv(source, sep="\t", index=False)
    config = load_config()
    config.values["tiering"]["auto_min_score"] = 999

    scored = score_inventory(source, config=config)

    row = scored.iloc[0]
    assert row["candidate_score"] < 999
    assert row["candidate_status"] == "AUTO_PRIMARY"
    assert row["recommendation"] == "PRIMARY"
    assert row["is_auto_primary"] == 1


def test_scoring_prefers_geometry_without_naming_conventions(tmp_path):
    inventory = pd.DataFrame([
        {"subject_folder": "011", "study_instance_uid": "study1", "series_instance_uid": "vendor", "study_date": "20200722", "modality": "CT", "image_type": "", "series_description": "ROUTINE", "study_description": "", "z_extent_mm": "500", "num_slices": "100", "reconstruction_diameter": "400", "convolution_kernel": "B30", "slice_thickness": "5"},
        {"subject_folder": "011", "study_instance_uid": "study1", "series_instance_uid": "bone", "study_date": "20200722", "modality": "CT", "image_type": "ORIGINAL\\PRIMARY\\AXIAL", "series_description": "BONE", "study_description": "", "z_extent_mm": "450", "num_slices": "100", "reconstruction_diameter": "400", "convolution_kernel": "B70", "slice_thickness": "1"},
    ])
    source = tmp_path / "inventory.tsv"
    inventory.to_csv(source, sep="\t", index=False)

    scored = score_inventory(source)

    row = scored.loc[scored["series_instance_uid"] == "vendor"].iloc[0]
    assert row["automatic_candidate"] == 1
    assert row["recommendation"] == "PRIMARY"
    assert row["coverage_score"] == 60
    assert row["fov_score"] == 40
    assert row["kernel_score"] == 25
    assert row["is_soft_kernel"] == 1
    assert row["candidate_score"] > scored.loc[scored["series_instance_uid"] == "bone", "candidate_score"].iloc[0]


def test_scoring_feature_helpers_are_explainable():
    assert _score_z_coverage("500") == 60
    assert _score_z_coverage("250") == 10
    assert _score_reconstruction_diameter("400") == 40
    assert _score_reconstruction_diameter("200") == -20
    assert _classify_kernel({"convolution_kernel": "B31", "series_description": "routine"}) == "soft"
    assert _classify_kernel({"convolution_kernel": "B70", "series_description": "routine"}) == "bone"
    assert _classify_phase({"series_description": "portal venous", "study_description": ""}) == "venous"
    assert _classify_phase({"series_description": "5 min delayed", "study_description": ""}) == "delayed"


def test_manifest_includes_secondary_only_when_requested(tmp_path):
    inventory = pd.DataFrame([
        {"subject_folder": "011", "study_instance_uid": "study1", "series_instance_uid": "s1", "study_date": "20200722", "series_number": "2", "series_description": "ABD", "study_description": "ABDOMEN", "source_directory": str(tmp_path), "representative_file": "x1", "tier": "Tier 1", "tier_reason": "primary", "recommendation": "PRIMARY", "study_group_key": "011|study1", "rule_version": "1"},
        {"subject_folder": "011", "study_instance_uid": "study1", "series_instance_uid": "s2", "study_date": "20200722", "series_number": "3", "series_description": "VENOUS", "study_description": "ABDOMEN", "source_directory": str(tmp_path), "representative_file": "x2", "tier": "Tier 2", "tier_reason": "secondary", "recommendation": "SECONDARY", "study_group_key": "011|study1", "rule_version": "1"},
    ])
    scored = tmp_path / "scored.tsv"
    review = tmp_path / "review.tsv"
    inventory.to_csv(scored, sep="\t", index=False)
    pd.DataFrame([
        {"series_key": "011|study1|s1", "reviewer_decision": "PRIMARY"},
        {"series_key": "011|study1|s2", "reviewer_decision": "SECONDARY"},
    ]).to_csv(review, sep="\t", index=False)

    primary_only = build_manifest(scored, review, tmp_path / "primary.tsv")
    with_secondary = build_manifest(scored, review, tmp_path / "all.tsv", include_secondary=True)
    assert list(primary_only["series_uid"]) == ["s1"]
    assert list(with_secondary["series_uid"]) == ["s1", "s2"]


def test_manifest_requires_ambiguous_review(tmp_path):
    inventory = pd.DataFrame([{
        "subject_folder": "sub-011", "study_instance_uid": "study1", "series_instance_uid": "s1",
        "study_date": "20200722", "series_number": "2", "series_description": "ABD",
        "study_description": "ABDOMEN", "source_directory": str(tmp_path), "representative_file": "x",
        "tier": "Tier 2", "tier_reason": "review", "rule_version": "1",
    }])
    scored = tmp_path / "scored.tsv"
    review = tmp_path / "review.tsv"
    output = tmp_path / "manifest.tsv"
    inventory.to_csv(scored, sep="\t", index=False)
    pd.DataFrame(columns=["series_key", "decision"]).to_csv(review, sep="\t", index=False)
    with pytest.raises(ValueError, match="Missing review decision"):
        build_manifest(scored, review, output)


def _pixel_dataset(values, slope=None, intercept=None, center=None, width=None):
    dataset = Dataset()
    from pydicom.dataset import FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian

    dataset.file_meta = FileMetaDataset()
    dataset.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    dataset.is_little_endian = True
    dataset.is_implicit_VR = False
    pixels = np.asarray(values, dtype=np.int16)
    dataset.Rows, dataset.Columns = pixels.shape
    dataset.BitsAllocated = 16
    dataset.BitsStored = 16
    dataset.HighBit = 15
    dataset.PixelRepresentation = 1
    dataset.SamplesPerPixel = 1
    dataset.PhotometricInterpretation = "MONOCHROME2"
    dataset.PixelData = pixels.tobytes()
    if slope is not None:
        dataset.RescaleSlope = slope
    if intercept is not None:
        dataset.RescaleIntercept = intercept
    if center is not None:
        dataset.WindowCenter = center
    if width is not None:
        dataset.WindowWidth = width
    return dataset


def test_review_scaling_converts_hu_and_applies_dicom_window():
    dataset = _pixel_dataset([[0, 1000, 2000]], slope=2, intercept=-1000, center=1000, width=2000)
    display = _display_pixels(dataset)
    assert display.tolist() == [[0, 127, 255]]


def test_review_scaling_uses_percentiles_without_window():
    dataset = _pixel_dataset([list(range(100)) + [10000]])
    display = _display_pixels(dataset)
    assert display.dtype == np.uint8
    assert display[0, 0] == 0
    assert display[0, -1] == 255
    assert display[0, 10] < display[0, 50] < display[0, 90]


def test_review_montage_uses_fixed_height_and_variable_width(tmp_path):
    archive = tmp_path / "series"
    archive.mkdir()
    for index in range(1, 4):
        _write_dicom(archive / "image{}.dcm".format(index), "series1", index)
    row = {
        "source_directory": str(archive),
        "series_instance_uid": "series1",
    }
    from liverct.ingestion.config import IngestionConfig
    thumbnail = _make_thumbnail(row, tmp_path / "assets", IngestionConfig())
    from PIL import Image
    image = Image.open(tmp_path / "assets" / "series1.png")
    assert thumbnail == "review_assets/series1.png"
    assert image.height == 260
    assert image.width == 3 * 240


def test_review_reuses_existing_montage_without_dicom_reads(tmp_path, monkeypatch):
    from liverct.ingestion.config import IngestionConfig

    assets = tmp_path / "assets"
    assets.mkdir()
    expected = assets / "series1.png"
    expected.write_bytes(b"cached")
    row = {"source_directory": str(tmp_path / "missing"), "series_instance_uid": "series1"}
    monkeypatch.setattr("liverct.ingestion.review._build_slice_index", lambda pending: (_ for _ in ()).throw(AssertionError("cache miss")))

    result = _generate_montages([row], assets, IngestionConfig())

    assert result["||series1"] == "review_assets/series1.png"


def test_review_slice_index_reads_metadata_without_pixels(tmp_path, monkeypatch):
    archive = tmp_path / "series"
    archive.mkdir()
    _write_dicom(archive / "image1.dcm", "series1", 1)
    calls = []
    import pydicom
    original_read = pydicom.dcmread

    def tracked_read(*args, **kwargs):
        calls.append(kwargs.get("stop_before_pixels"))
        return original_read(*args, **kwargs)

    monkeypatch.setattr(pydicom, "dcmread", tracked_read)
    row = {"source_directory": str(archive), "series_instance_uid": "series1"}
    index = _build_slice_index([("||series1", row, "", "")])

    assert index["||series1"]
    assert calls == [True]


def test_review_montage_generation_supports_process_workers(tmp_path):
    archive = tmp_path / "series"
    archive.mkdir()
    for index in range(1, 4):
        _write_dicom(archive / "image{}.dcm".format(index), "series1", index)
    from liverct.ingestion.config import IngestionConfig
    config = IngestionConfig()
    config.values["review"]["montage_workers"] = 2
    row = {"source_directory": str(archive), "series_instance_uid": "series1"}

    result = _generate_montages([row], tmp_path / "assets", config)

    assert result["||series1"] == "review_assets/series1.png"
    assert (tmp_path / "assets" / "series1.png").exists()


def test_review_template_prepopulates_candidates_and_preserves_decisions(tmp_path):
    scored = pd.DataFrame([
        {
            "subject_folder": "011", "study_instance_uid": "study1", "series_instance_uid": "s1",
            "study_date": "20200722", "study_description": "ABDOMEN", "series_number": "2",
            "series_description": "ABD", "modality": "CT", "image_type": "ORIGINAL\\PRIMARY",
            "num_slices": "100", "z_extent_mm": "250", "source_directory": str(tmp_path),
            "tier": "Tier 2", "tier_reason": "requires review",
        },
        {
            "subject_folder": "011", "study_instance_uid": "study1", "series_instance_uid": "s2",
            "study_date": "20200722", "study_description": "ABDOMEN", "series_number": "3",
            "series_description": "SCOUT", "modality": "CT", "image_type": "LOCALIZER",
            "num_slices": "1", "z_extent_mm": "", "source_directory": str(tmp_path),
            "tier": "Tier 4", "tier_reason": "series contains only 1 or 2 slices",
        },
    ])
    scored_path = tmp_path / "scored.tsv"
    output_dir = tmp_path / "review"
    scored.to_csv(scored_path, sep="\t", index=False)

    review_path = generate_review_reports(scored_path, output_dir)
    review = pd.read_csv(review_path, sep="\t", dtype=str).fillna("")
    assert list(review.loc[review["is_data"] == "1", "series_instance_uid"]) == ["s1"]
    assert review.loc[0, "series_key"] == "011|study1|s1"
    assert review.loc[0, "reviewer_decision"] == "PRIMARY"

    review.loc[0, "reviewer_decision"] = "SECONDARY"
    review.loc[0, "notes"] = "reviewer1"
    review.to_csv(review_path, sep="\t", index=False)
    generate_review_reports(scored_path, output_dir)
    rerun = pd.read_csv(review_path, sep="\t", dtype=str).fillna("")
    assert rerun.loc[0, "reviewer_decision"] == "SECONDARY"
    assert rerun.loc[0, "notes"] == "reviewer1"


def test_review_template_is_compact_by_scan_date(tmp_path):
    scored = pd.DataFrame([
        {"subject_folder": "011", "study_instance_uid": "study1", "series_instance_uid": "primary", "study_date": "20200722", "series_number": "2", "series_description": "ABD", "study_description": "ABDOMEN", "image_type": "ORIGINAL\\PRIMARY\\AXIAL", "num_slices": "100", "z_extent_mm": "300", "tier": "Tier 1", "recommendation": "PRIMARY", "candidate_status": "AUTO_PRIMARY", "automatic_candidate": "1", "is_auto_primary": "1", "candidate_score": "150", "source_directory": str(tmp_path)},
        {"subject_folder": "011", "study_instance_uid": "study1", "series_instance_uid": "scout", "study_date": "20200722", "series_number": "1", "series_description": "SCOUT", "study_description": "ABDOMEN", "image_type": "LOCALIZER", "num_slices": "1", "tier": "Tier 4", "recommendation": "REJECT", "candidate_status": "AUTO_PRIMARY", "automatic_candidate": "0", "source_directory": str(tmp_path)},
        {"subject_folder": "011", "study_instance_uid": "study2", "series_instance_uid": "none", "study_date": "20210830", "series_number": "1", "series_description": "HEAD", "study_description": "HEAD", "image_type": "ORIGINAL\\PRIMARY\\AXIAL", "num_slices": "100", "tier": "Tier 4", "recommendation": "REJECT", "candidate_status": "NO_CANDIDATE", "automatic_candidate": "0", "candidate_reason": "no eligible candidate", "source_directory": str(tmp_path)},
    ])
    scored_path = tmp_path / "scored.tsv"
    scored.to_csv(scored_path, sep="\t", index=False)
    review = pd.read_csv(generate_review_reports(scored_path, tmp_path / "review"), sep="\t", dtype=str, keep_default_na=False)
    assert review.empty


def test_review_html_handles_automatic_scan_without_review_rows(tmp_path):
    scored = pd.DataFrame([{
        "subject_folder": "011", "patient_id": "patient-1", "study_instance_uid": "study1",
        "series_instance_uid": "primary", "study_date": "20200722", "series_number": "1",
        "series_description": "ABD", "study_description": "ABDOMEN", "image_type": "ORIGINAL\\PRIMARY\\AXIAL",
        "num_slices": "100", "z_extent_mm": "500", "recommendation": "PRIMARY",
        "candidate_status": "AUTO_PRIMARY", "automatic_candidate": "0", "is_auto_primary": "0",
        "candidate_reason": "only strict abdominal axial candidate", "source_directory": str(tmp_path),
    }])
    scored_path = tmp_path / "scored.tsv"
    output_dir = tmp_path / "review"
    scored.to_csv(scored_path, sep="\t", index=False)

    generate_review_reports(scored_path, output_dir)

    report = (output_dir / "sub-patient-1.html").read_text(encoding="utf-8")
    assert "only strict abdominal axial candidate" in report


def test_review_template_contains_only_ambiguous_candidates(tmp_path):
    scored = pd.DataFrame([
        {"subject_folder": "011", "study_instance_uid": "study1", "series_instance_uid": "a", "study_date": "20200722", "series_number": "1", "series_description": "ABD", "study_description": "ABDOMEN", "image_type": "ORIGINAL\\PRIMARY\\AXIAL", "num_slices": "100", "z_extent_mm": "300", "recommendation": "PRIMARY", "candidate_status": "REVIEW_REQUIRED", "automatic_candidate": "1", "candidate_score": "80", "source_directory": str(tmp_path)},
        {"subject_folder": "011", "study_instance_uid": "study1", "series_instance_uid": "b", "study_date": "20200722", "series_number": "2", "series_description": "VENOUS", "study_description": "ABDOMEN", "image_type": "ORIGINAL\\PRIMARY\\AXIAL", "num_slices": "100", "z_extent_mm": "300", "recommendation": "SECONDARY", "candidate_status": "REVIEW_REQUIRED", "automatic_candidate": "1", "candidate_score": "78", "source_directory": str(tmp_path)},
        {"subject_folder": "011", "study_instance_uid": "study2", "series_instance_uid": "auto", "study_date": "20210830", "series_number": "1", "series_description": "ABD", "study_description": "ABDOMEN", "image_type": "ORIGINAL\\PRIMARY\\AXIAL", "num_slices": "100", "z_extent_mm": "500", "recommendation": "PRIMARY", "candidate_status": "AUTO_PRIMARY", "automatic_candidate": "1", "is_auto_primary": "1", "candidate_score": "150", "source_directory": str(tmp_path)},
    ])
    scored_path = tmp_path / "scored.tsv"
    scored.to_csv(scored_path, sep="\t", index=False)

    review = pd.read_csv(generate_review_reports(scored_path, tmp_path / "review"), sep="\t", dtype=str, keep_default_na=False)

    assert list(review.loc[review["is_data"] == "1", "series_instance_uid"]) == ["a", "b"]
    assert set(review["candidate_status"]) == {"REVIEW_REQUIRED"}


def test_review_artifacts_sort_by_date_and_numeric_series(tmp_path):
    rows = []
    for study_date, series_number, series_uid in [
        ("20210102", "10", "s10"), ("20210102", "2", "s2"),
        ("20200722", "11", "s11"), ("20200722", "1", "s1"),
    ]:
        rows.append({
            "subject_folder": "011", "patient_id": "011", "study_instance_uid": "study-{}".format(study_date),
            "series_instance_uid": series_uid, "study_date": study_date,
            "study_description": "ABDOMEN", "series_number": series_number,
            "series_description": "ABD", "modality": "CT", "image_type": "ORIGINAL\\PRIMARY\\AXIAL",
            "num_slices": "100", "z_extent_mm": "250", "source_directory": str(tmp_path),
            "tier": "Tier 2", "tier_reason": "review", "recommendation": "PRIMARY",
        })
    scored_path = tmp_path / "scored.tsv"
    output_dir = tmp_path / "review"
    pd.DataFrame(rows).to_csv(scored_path, sep="\t", index=False)

    review_path = generate_review_reports(scored_path, output_dir)
    review = pd.read_csv(review_path, sep="\t", dtype=str, keep_default_na=False)
    data = review[review["is_data"] == "1"]
    assert list(zip(data["study_date"], data["series_number"])) == [
        ("20200722", "1"), ("20200722", "11"), ("20210102", "2"), ("20210102", "10")
    ]
    assert list(review.loc[review["is_data"] == "0", "index"]) == ["3"]
    report = (output_dir / "sub-011.html").read_text(encoding="utf-8")
    assert report.index("Study Date: 2020-07-22") < report.index("Study Date: 2021-01-02")
    assert report.index(">1</td>") < report.index(">11</td>")


def test_review_html_groups_by_patient_id_not_subject_folder(tmp_path):
    scored = pd.DataFrame([
        {"subject_folder": "011", "patient_id": "patient-1", "study_instance_uid": "study1", "series_instance_uid": "s1", "study_date": "20200722", "study_description": "ABDOMEN", "series_number": "1", "series_description": "ABD", "modality": "CT", "image_type": "ORIGINAL\\PRIMARY\\AXIAL", "num_slices": "100", "z_extent_mm": "250", "source_directory": str(tmp_path), "tier": "Tier 2", "recommendation": "PRIMARY"},
        {"subject_folder": "012", "patient_id": "patient-1", "study_instance_uid": "study2", "series_instance_uid": "s2", "study_date": "20210830", "study_description": "ABDOMEN", "series_number": "1", "series_description": "ABD", "modality": "CT", "image_type": "ORIGINAL\\PRIMARY\\AXIAL", "num_slices": "100", "z_extent_mm": "250", "source_directory": str(tmp_path), "tier": "Tier 2", "recommendation": "PRIMARY"},
    ])
    scored_path = tmp_path / "scored.tsv"
    output_dir = tmp_path / "review"
    scored.to_csv(scored_path, sep="\t", index=False)

    generate_review_reports(scored_path, output_dir)

    report = (output_dir / "sub-patient-1.html").read_text(encoding="utf-8")
    assert "Study Date: 2020-07-22" in report
    assert "Study Date: 2021-08-30" in report
    assert "<th>Series Key</th>" not in report
    first_series = report.index("011|study1|s1")
    second_series = report.index("012|study2|s2")
    assert report.index("<strong>Series Key:</strong>", report.index(">ABD</td>")) < first_series
    assert report.index("<strong>Series Key:</strong>", report.index(">ABD</td>", report.index(">ABD</td>") + 1)) < second_series
    assert not (output_dir / "sub-011.html").exists()
    assert not (output_dir / "sub-012.html").exists()


def test_same_folder_date_different_patients_are_separate_scan_groups(tmp_path):
    scored = pd.DataFrame([
        {"subject_folder": "shared-folder", "patient_id": "patient-1", "study_instance_uid": "study1", "series_instance_uid": "s1", "study_date": "20200722", "study_description": "ABDOMEN", "series_number": "1", "series_description": "ABD", "modality": "CT", "image_type": "ORIGINAL\\PRIMARY\\AXIAL", "num_slices": "100", "z_extent_mm": "250", "source_directory": str(tmp_path), "tier": "Tier 2", "recommendation": "PRIMARY", "candidate_status": "REVIEW_REQUIRED", "automatic_candidate": "1", "candidate_score": "80", "scan_group_key": "shared-folder|20200722"},
        {"subject_folder": "shared-folder", "patient_id": "patient-2", "study_instance_uid": "study2", "series_instance_uid": "s2", "study_date": "20200722", "study_description": "ABDOMEN", "series_number": "1", "series_description": "ABD", "modality": "CT", "image_type": "ORIGINAL\\PRIMARY\\AXIAL", "num_slices": "100", "z_extent_mm": "80", "source_directory": str(tmp_path), "tier": "Tier 2", "recommendation": "PRIMARY", "candidate_status": "REVIEW_REQUIRED", "automatic_candidate": "1", "candidate_score": "80", "scan_group_key": "shared-folder|20200722"},
    ])
    scored_path = tmp_path / "scored.tsv"
    output_dir = tmp_path / "review"
    scored.to_csv(scored_path, sep="\t", index=False)

    generate_review_reports(scored_path, output_dir)

    review = pd.read_csv(output_dir / "review.tsv", sep="\t", dtype=str, keep_default_na=False)
    data = review[review["is_data"] == "1"]
    assert set(data["patient_id"]) == {"patient-1", "patient-2"}
    assert (output_dir / "sub-patient-1.html").exists()
    assert (output_dir / "sub-patient-2.html").exists()
