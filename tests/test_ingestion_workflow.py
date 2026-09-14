import pandas as pd
import pytest
import numpy as np
from pydicom import Dataset

from liverct.ingestion import build_manifest, inventory_archive, load_config, score_inventory, stage_sourcedata
from liverct.ingestion.review import _display_pixels, _make_thumbnail, generate_review_reports


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
        {"subject_folder": "011", "study_instance_uid": "study1", "series_instance_uid": "s1", "study_date": "20200722", "modality": "CT", "image_type": "ORIGINAL\\PRIMARY\\AXIAL", "series_description": "ABD", "study_description": "ABDOMEN", "z_extent_mm": "250", "num_slices": "100"},
        {"subject_folder": "011", "study_instance_uid": "study1", "series_instance_uid": "s2", "study_date": "20200722", "modality": "CT", "image_type": "ORIGINAL\\PRIMARY", "series_description": "VENOUS", "study_description": "ABDOMEN", "z_extent_mm": "250", "num_slices": "100"},
        {"subject_folder": "011", "study_instance_uid": "study2", "series_instance_uid": "s3", "study_date": "20210830", "modality": "CT", "image_type": "LOCALIZER", "series_description": "SCOUT", "study_description": "ABDOMEN", "z_extent_mm": "", "num_slices": "1"},
    ])
    source = tmp_path / "inventory.tsv"
    inventory.to_csv(source, sep="\t", index=False)
    scored = score_inventory(source)
    assert list(scored["recommendation"]) == ["PRIMARY", "SECONDARY", "REJECT"]
    assert scored.loc[0, "study_group_key"] == "011|study1"


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
    assert list(review["series_instance_uid"]) == ["s1", "s2"]
    assert review.loc[0, "series_key"] == "011|study1|s1"
    assert review.loc[0, "reviewer_decision"] == "PRIMARY"
    assert review.loc[1, "recommendation"] == "REJECT"

    review.loc[0, "reviewer_decision"] = "SECONDARY"
    review.loc[0, "notes"] = "reviewer1"
    review.to_csv(review_path, sep="\t", index=False)
    generate_review_reports(scored_path, output_dir)
    rerun = pd.read_csv(review_path, sep="\t", dtype=str).fillna("")
    assert rerun.loc[0, "reviewer_decision"] == "SECONDARY"
    assert rerun.loc[0, "notes"] == "reviewer1"


def test_review_artifacts_sort_by_date_and_numeric_series(tmp_path):
    rows = []
    for study_date, series_number, series_uid in [
        ("20210102", "10", "s10"), ("20210102", "2", "s2"),
        ("20200722", "11", "s11"), ("20200722", "1", "s1"),
    ]:
        rows.append({
            "subject_folder": "011", "study_instance_uid": "study-{}".format(study_date),
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
    assert report.index(">2</td>") < report.index(">10</td>")
