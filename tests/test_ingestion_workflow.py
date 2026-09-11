import pandas as pd
import pytest

from liverct.ingestion import build_manifest, inventory_archive, load_config, score_inventory, stage_sourcedata


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
        "subject_id": "0119838", "session_id": "20200722", "series_uid": "series1",
        "source_directory": series["source_directory"],
    }]).to_csv(manifest, sep="\t", index=False)

    sourcedata = stage_sourcedata(manifest, tmp_path, tmp_path / "bids")
    staged = list(sourcedata.rglob("*.dcm"))
    assert len(staged) == 3


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
    ])
    source = tmp_path / "inventory.tsv"
    output = tmp_path / "scored.tsv"
    inventory.to_csv(source, sep="\t", index=False)
    scored = score_inventory(source, output)
    assert list(scored["tier"]) == ["Tier 1", "Tier 2", "Tier 4", "Tier 4"]


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
