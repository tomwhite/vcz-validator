from pathlib import Path

import numpy as np
import pytest
import zarr

from vcz_validator.validator import validate


@pytest.fixture()
def example_vcz_path(tmp_path):
    path = Path(tmp_path) / "example.vcz"
    path.mkdir()
    root = zarr.create_group(path, zarr_format=2)
    root.attrs["vcf_zarr_version"] = "0.4"

    def create_array(name, dimension_names, data):
        arr = root.create_array(name, data=data)
        arr.attrs["_ARRAY_DIMENSIONS"] = dimension_names

    create_array("variant_contig", ["variants"], np.array([0, 1], dtype=np.int32))
    create_array("variant_position", ["variants"], np.array([10, 20], dtype=np.int32))
    create_array("variant_id", ["variants"], np.array(["a", "b"]))
    create_array(
        "variant_allele", ["variants", "alleles"], np.array([["A", "."], ["C", "G"]])
    )
    create_array(
        "variant_quality", ["variants"], np.array([1.0, 2.0], dtype=np.float32)
    )
    create_array(
        "variant_filter",
        ["variants", "filters"],
        np.array([[True], [False]], dtype=np.bool),
    )

    create_array("contig_id", ["contigs"], np.array(["chr1", "chr2"]))
    create_array("filter_id", ["filters"], np.array(["PASS"]))
    create_array("sample_id", ["samples"], np.array(["S1", "S2", "S3"]))

    return path


def expect_validate_failure(path, expected_message):
    failures = validate(path)
    assert len(failures) == 1
    assert failures[0].message == expected_message


def test_failure__path_does_not_exist(tmp_path):
    non_existent_path = Path(tmp_path) / "non-existent"
    expect_validate_failure(
        non_existent_path,
        f"Path '{non_existent_path}' does not exist",
    )


def test_failure__path_is_not_zarr_group(tmp_path):
    path = Path(tmp_path) / "path"
    path.mkdir()
    expect_validate_failure(
        path,
        f"Path '{path}' is not a Zarr group",
    )


def test_failure__path_is_not_zarr_group_v3(tmp_path):
    path = Path(tmp_path) / "path"
    path.mkdir()
    zarr.create_group(path)

    expect_validate_failure(path, "Zarr format must be 2, but was 3")


def test_failure__path_vcf_zarr_version_not_present(tmp_path):
    path = Path(tmp_path) / "path"
    path.mkdir()
    zarr.create_group(path, zarr_format=2)

    expect_validate_failure(path, "'vcf_zarr_version' group attribute must be present")


def test_failure__path_vcf_zarr_version_not_supported(tmp_path):
    path = Path(tmp_path) / "path"
    path.mkdir()
    root = zarr.create_group(path, zarr_format=2)
    root.attrs["vcf_zarr_version"] = "0.3"

    expect_validate_failure(path, "'vcf_zarr_version' must be '0.4', but was '0.3'")


def test_failure__array_dimension_names_missing(tmp_path):
    path = Path(tmp_path) / "path"
    path.mkdir()
    root = zarr.create_group(path, zarr_format=2)
    root.attrs["vcf_zarr_version"] = "0.4"
    root.create_array("variant_position", data=np.array([1, 2]))

    expect_validate_failure(
        path,
        "Arrays must have dimension names, but they were missing for: variant_position",
    )


def test_failure__dimension_names_len_mismatches_array_ndim(tmp_path):
    path = Path(tmp_path) / "path"
    path.mkdir()
    root = zarr.create_group(path, zarr_format=2)
    root.attrs["vcf_zarr_version"] = "0.4"
    variant_position = root.create_array(
        "variant_position", data=np.array([1, 2], dtype=np.int32)
    )
    variant_position.attrs["_ARRAY_DIMENSIONS"] = ["variants", "extra"]

    expect_validate_failure(
        path,
        "Number of dimension names must match array ndim, "
        "but they were mismatched for: variant_position.\n"
        "The dimension name counts and ndims were:\n"
        "{'variant_position': {'dimension_names': 2, 'ndim': 1}}",
    )


def test_failure__array_dimension_names_with_inconsistent_sizes(tmp_path):
    path = Path(tmp_path) / "path"
    path.mkdir()
    root = zarr.create_group(path, zarr_format=2)
    root.attrs["vcf_zarr_version"] = "0.4"
    variant_position = root.create_array("variant_position", data=np.array([1, 2]))
    variant_position.attrs["_ARRAY_DIMENSIONS"] = ["variants"]
    variant_id = root.create_array("variant_id", data=np.array(["a", "b", "c"]))
    variant_id.attrs["_ARRAY_DIMENSIONS"] = ["variants"]

    expect_validate_failure(
        path,
        "Dimension names must have consistent sizes, but they were inconsistent "
        "for: variants.\nThe array dimensions and sizes were:\n"
        "{'variant_id': {'variants': 3}, 'variant_position': {'variants': 2}}",
    )


def test_failure__required_fields_missing(tmp_path):
    path = Path(tmp_path) / "path"
    path.mkdir()
    root = zarr.create_group(path, zarr_format=2)
    root.attrs["vcf_zarr_version"] = "0.4"

    expect_validate_failure(
        path,
        "Missing required fields: contig_id,filter_id,sample_id,variant_allele,"
        "variant_contig,variant_filter,variant_id,variant_position,variant_quality",
    )


def test_failure__field_dimension_names_incorrect(example_vcz_path):
    root = zarr.open(example_vcz_path, mode="r+")
    root["variant_position"].attrs["_ARRAY_DIMENSIONS"] = ["contigs"]

    expect_validate_failure(
        example_vcz_path,
        "Incorrect dimension names for 'variant_position': "
        "expected ['variants'] but was ['contigs']",
    )


def test_success(example_vcz_path):
    failures = validate(example_vcz_path)
    assert len(failures) == 0
