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
    create_array("variant_id", ["variants"], np.array(["a", "b"], dtype="T"))
    create_array(
        "variant_allele",
        ["variants", "alleles"],
        np.array([["A", "."], ["C", "G"]], dtype="T"),
    )
    create_array(
        "variant_quality", ["variants"], np.array([1.0, 2.0], dtype=np.float32)
    )
    create_array(
        "variant_filter",
        ["variants", "filters"],
        np.array([[True], [False]], dtype=np.bool),
    )

    create_array("contig_id", ["contigs"], np.array(["chr1", "chr2"], dtype="T"))
    create_array("filter_id", ["filters"], np.array(["PASS"], dtype="T"))
    create_array(
        "filter_description", ["filters"], np.array(["All filters passed"], dtype="T")
    )
    create_array("sample_id", ["samples"], np.array(["S1", "S2", "S3"], dtype="T"))

    return path


def expect_validate_failures(path, *expected_messages):
    failures = validate(path)
    assert len(failures) == len(expected_messages)
    for failure, expected_message in zip(failures, expected_messages):
        assert failure.message == expected_message


def test_failure__path_does_not_exist(tmp_path):
    non_existent_path = Path(tmp_path) / "non-existent"
    expect_validate_failures(
        non_existent_path,
        f"Path '{non_existent_path}' does not exist",
    )


def test_failure__path_is_not_zarr_group(tmp_path):
    path = Path(tmp_path) / "path"
    path.mkdir()
    expect_validate_failures(
        path,
        f"Path '{path}' is not a Zarr group",
    )


def test_failure__path_is_not_zarr_group_v3(tmp_path):
    path = Path(tmp_path) / "path"
    path.mkdir()
    zarr.create_group(path)

    expect_validate_failures(path, "Zarr format must be 2, but was 3")


def test_failure__path_vcf_zarr_version_not_present(tmp_path):
    path = Path(tmp_path) / "path"
    path.mkdir()
    zarr.create_group(path, zarr_format=2)

    expect_validate_failures(path, "'vcf_zarr_version' group attribute must be present")


def test_failure__path_vcf_zarr_version_not_supported(tmp_path):
    path = Path(tmp_path) / "path"
    path.mkdir()
    root = zarr.create_group(path, zarr_format=2)
    root.attrs["vcf_zarr_version"] = "0.3"

    expect_validate_failures(path, "'vcf_zarr_version' must be '0.4', but was '0.3'")


def test_failure__source_attribute_not_a_string(example_vcz_path):
    root = zarr.open(example_vcz_path, mode="r+")
    root.attrs["source"] = 123

    expect_validate_failures(
        example_vcz_path,
        "'source' group attribute must be a string, but was int",
    )


def test_failure__vcf_meta_information_attribute_not_a_list(example_vcz_path):
    root = zarr.open(example_vcz_path, mode="r+")
    root.attrs["vcf_meta_information"] = "not a list"

    expect_validate_failures(
        example_vcz_path,
        "'vcf_meta_information' group attribute must be a list, but was str",
    )


def test_failure__array_dimension_names_missing(tmp_path):
    path = Path(tmp_path) / "path"
    path.mkdir()
    root = zarr.create_group(path, zarr_format=2)
    root.attrs["vcf_zarr_version"] = "0.4"
    root.create_array("variant_position", data=np.array([1, 2]))

    expect_validate_failures(
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

    expect_validate_failures(
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

    expect_validate_failures(
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

    expect_validate_failures(
        path,
        "Missing required fields: contig_id,sample_id,variant_allele,variant_contig,"
        "variant_position",
    )


def test_failure__field_dimension_names_incorrect(example_vcz_path):
    root = zarr.open(example_vcz_path, mode="r+")
    root["variant_position"].attrs["_ARRAY_DIMENSIONS"] = ["contigs"]

    expect_validate_failures(
        example_vcz_path,
        "Incorrect dimension names for 'variant_position': "
        "expected ['variants'] but was ['contigs']",
    )


def test_failure__field_dtype_kind_incorrect(example_vcz_path):
    root = zarr.open(example_vcz_path, mode="r+")
    del root["variant_contig"]
    arr = root.create_array(
        "variant_contig", data=np.array([0.0, 1.0], dtype=np.float32)
    )
    arr.attrs["_ARRAY_DIMENSIONS"] = ["variants"]

    expect_validate_failures(
        example_vcz_path,
        "Incorrect datatype for 'variant_contig': expected 'int' but array dtype "
        "kind was 'f'",
    )


def test_failure__string_field_dtype_kind_incorrect(example_vcz_path):
    root = zarr.open(example_vcz_path, mode="r+")
    del root["variant_id"]
    arr = root.create_array("variant_id", data=np.array([0, 1], dtype=np.int32))
    arr.attrs["_ARRAY_DIMENSIONS"] = ["variants"]

    expect_validate_failures(
        example_vcz_path,
        "Incorrect datatype for 'variant_id': expected 'str' but array dtype kind "
        "was 'i'",
        "String field 'variant_id' must have a vlen-utf8 filter",
    )


def test_failure__string_field_missing_vlen_utf8_filter(example_vcz_path):
    root = zarr.open(example_vcz_path, mode="r+")
    del root["variant_id"]
    arr = root.create_array("variant_id", data=np.array(["a", "b"]))
    arr.attrs["_ARRAY_DIMENSIONS"] = ["variants"]

    expect_validate_failures(
        example_vcz_path,
        "String field 'variant_id' must have a vlen-utf8 filter",
    )


def test_failure__optional_field_dtype_kind_incorrect(example_vcz_path):
    root = zarr.open(example_vcz_path, mode="r+")
    arr = root.create_array(
        "contig_length", data=np.array([0.0, 1.0], dtype=np.float32)
    )
    arr.attrs["_ARRAY_DIMENSIONS"] = ["contigs"]

    expect_validate_failures(
        example_vcz_path,
        "Incorrect datatype for 'contig_length': expected 'int' but array dtype "
        "kind was 'f'",
    )


def test_failure__info_field_wrong_number_of_dimensions(example_vcz_path):
    root = zarr.open(example_vcz_path, mode="r+")
    arr = root.create_array("variant_AD", data=np.array([1, 2], dtype=np.int32))
    arr.attrs["_ARRAY_DIMENSIONS"] = ["variants"]

    expect_validate_failures(
        example_vcz_path,
        "INFO field 'variant_AD' must be 2-dimensional with dimensions "
        "['variants', ...], but had dimensions ['variants']",
    )


def test_failure__info_field_invalid_dtype(example_vcz_path):
    root = zarr.open(example_vcz_path, mode="r+")
    arr = root.create_array(
        "variant_AD",
        data=np.array([[1 + 2j, 3 + 4j], [5 + 6j, 7 + 8j]], dtype=np.complex64),
    )
    arr.attrs["_ARRAY_DIMENSIONS"] = ["variants", "alleles"]

    expect_validate_failures(
        example_vcz_path,
        "INFO field 'variant_AD' has invalid dtype kind 'c'. "
        "Must be one of: 'b' (bool), 'i' (int), 'f' (float), 'S','U','T' (str)",
    )


def test_failure__format_field_wrong_number_of_dimensions(example_vcz_path):
    root = zarr.open(example_vcz_path, mode="r+")
    arr = root.create_array(
        "call_AD",
        data=np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32),
    )
    arr.attrs["_ARRAY_DIMENSIONS"] = ["variants", "samples"]

    expect_validate_failures(
        example_vcz_path,
        "FORMAT field 'call_AD' must be 3-dimensional with dimensions "
        "['variants', 'samples', ...], but had dimensions ['variants', 'samples']",
    )


def test_failure__format_field_invalid_dtype(example_vcz_path):
    root = zarr.open(example_vcz_path, mode="r+")
    arr = root.create_array(
        "call_AD",
        data=np.array(
            [
                [[1 + 2j, 3 + 4j], [5 + 6j, 7 + 8j], [1 + 0j, 2 + 0j]],
                [[1 + 2j, 3 + 4j], [5 + 6j, 7 + 8j], [1 + 0j, 2 + 0j]],
            ],
            dtype=np.complex64,
        ),
    )
    arr.attrs["_ARRAY_DIMENSIONS"] = ["variants", "samples", "alleles"]

    expect_validate_failures(
        example_vcz_path,
        "FORMAT field 'call_AD' has invalid dtype kind 'c'. "
        "Must be one of: 'b' (bool), 'i' (int), 'f' (float), 'S','U','T' (str)",
    )


def test_failure__call_genotype_wrong_dimensions(example_vcz_path):
    root = zarr.open(example_vcz_path, mode="r+")
    arr = root.create_array(
        "call_genotype",
        data=np.array([[0, 1, 0], [1, 0, 1]], dtype=np.int32),
    )
    arr.attrs["_ARRAY_DIMENSIONS"] = ["variants", "samples"]

    expect_validate_failures(
        example_vcz_path,
        "Incorrect dimension names for 'call_genotype': "
        "expected ['variants', 'samples', 'ploidy'] but was ['variants', 'samples']",
    )


def test_failure__call_genotype_wrong_dtype(example_vcz_path):
    root = zarr.open(example_vcz_path, mode="r+")
    arr = root.create_array(
        "call_genotype",
        data=np.array(
            [
                [[0.0, 1.0], [1.0, 0.0], [0.0, 1.0]],
                [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]],
            ],
            dtype=np.float32,
        ),
    )
    arr.attrs["_ARRAY_DIMENSIONS"] = ["variants", "samples", "ploidy"]

    expect_validate_failures(
        example_vcz_path,
        "Incorrect datatype for 'call_genotype': expected 'int' but array dtype "
        "kind was 'f'",
    )


def test_failure__call_genotype_phased_wrong_dimensions(example_vcz_path):
    root = zarr.open(example_vcz_path, mode="r+")
    arr = root.create_array(
        "call_genotype_phased",
        data=np.array([True, False], dtype=np.bool_),
    )
    arr.attrs["_ARRAY_DIMENSIONS"] = ["variants"]

    expect_validate_failures(
        example_vcz_path,
        "Incorrect dimension names for 'call_genotype_phased': "
        "expected ['variants', 'samples'] but was ['variants']",
    )


def test_failure__call_genotype_phased_wrong_dtype(example_vcz_path):
    root = zarr.open(example_vcz_path, mode="r+")
    arr = root.create_array(
        "call_genotype_phased",
        data=np.array([[0, 1, 0], [1, 0, 1]], dtype=np.int32),
    )
    arr.attrs["_ARRAY_DIMENSIONS"] = ["variants", "samples"]

    expect_validate_failures(
        example_vcz_path,
        "Incorrect datatype for 'call_genotype_phased': expected 'bool' but array "
        "dtype kind was 'i'",
    )


def test_failure__mask_array_wrong_dtype(example_vcz_path):
    root = zarr.open(example_vcz_path, mode="r+")
    # Add a valid INFO field
    arr = root.create_array(
        "variant_DP", data=np.array([[10, 20], [30, 40]], dtype=np.int32)
    )
    arr.attrs["_ARRAY_DIMENSIONS"] = ["variants", "alleles"]
    # Add a mask with wrong dtype
    mask = root.create_array(
        "variant_DP_mask", data=np.array([[0, 1], [1, 0]], dtype=np.int32)
    )
    mask.attrs["_ARRAY_DIMENSIONS"] = ["variants", "alleles"]

    expect_validate_failures(
        example_vcz_path,
        "Mask array 'variant_DP_mask' must have dtype kind 'b' (bool), " "but was 'i'",
    )


def test_failure__mask_array_wrong_shape(example_vcz_path):
    root = zarr.open(example_vcz_path, mode="r+")
    arr = root.create_array(
        "variant_DP", data=np.array([[10, 20], [30, 40]], dtype=np.int32)
    )
    arr.attrs["_ARRAY_DIMENSIONS"] = ["variants", "alleles"]
    mask = root.create_array(
        "variant_DP_mask", data=np.array([True, False], dtype=np.bool_)
    )
    mask.attrs["_ARRAY_DIMENSIONS"] = ["variants"]

    expect_validate_failures(
        example_vcz_path,
        "Mask array 'variant_DP_mask' must have the same shape as "
        "'variant_DP' (2, 2), but was (2,)",
    )


def test_failure__fill_array_wrong_dtype(example_vcz_path):
    root = zarr.open(example_vcz_path, mode="r+")
    arr = root.create_array(
        "variant_DP", data=np.array([[10, 20], [30, 40]], dtype=np.int32)
    )
    arr.attrs["_ARRAY_DIMENSIONS"] = ["variants", "alleles"]
    fill = root.create_array(
        "variant_DP_fill", data=np.array([[0, 1], [1, 0]], dtype=np.int32)
    )
    fill.attrs["_ARRAY_DIMENSIONS"] = ["variants", "alleles"]

    expect_validate_failures(
        example_vcz_path,
        "Fill array 'variant_DP_fill' must have dtype kind 'b' (bool), " "but was 'i'",
    )


def test_failure__fill_array_wrong_shape(example_vcz_path):
    root = zarr.open(example_vcz_path, mode="r+")
    arr = root.create_array(
        "variant_DP", data=np.array([[10, 20], [30, 40]], dtype=np.int32)
    )
    arr.attrs["_ARRAY_DIMENSIONS"] = ["variants", "alleles"]
    fill = root.create_array(
        "variant_DP_fill", data=np.array([True, False], dtype=np.bool_)
    )
    fill.attrs["_ARRAY_DIMENSIONS"] = ["variants"]

    expect_validate_failures(
        example_vcz_path,
        "Fill array 'variant_DP_fill' must have the same shape as "
        "'variant_DP' (2, 2), but was (2,)",
    )


@pytest.mark.parametrize("path_type", [Path, str])
def test_success(example_vcz_path, path_type):
    failures = validate(path_type(example_vcz_path))
    assert len(failures) == 0
