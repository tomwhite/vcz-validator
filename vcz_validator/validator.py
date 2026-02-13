import pprint as pp
from dataclasses import dataclass

import zarr
from zarr.errors import GroupNotFoundError

REQUIRED_VARIABLE_NAMES = [
    "variant_contig",
    "variant_position",
    "variant_id",
    "variant_allele",
    "variant_quality",
    "variant_filter",
    "contig_id",
    "filter_id",
    "sample_id",
]


@dataclass
class Check:
    pass


@dataclass
class PathCheck(Check):
    def check(self, path):
        pass


@dataclass
class ZarrCheck(Check):
    def check(self, root):
        pass


@dataclass
class Failure:
    message: str
    stop: bool = False


class CheckPathExists(PathCheck):
    def check(self, path):
        if not path.exists():
            return Failure(f"Path '{path}' does not exist", stop=True)


class CheckPathIsZarrGroup(PathCheck):
    def check(self, path):
        try:
            zarr.open(path, mode="r")
        except GroupNotFoundError:
            return Failure(f"Path '{path}' is not a Zarr group", stop=True)


class CheckZarrFormatIsV2(ZarrCheck):
    def check(self, root):
        zarr_format = root.metadata.zarr_format
        if zarr_format != 2:
            return Failure(f"Zarr format must be 2, but was {zarr_format}", stop=True)


class CheckVcfZarrVersionGroupAttributeIsPresent(ZarrCheck):
    def check(self, root):
        if "vcf_zarr_version" not in root.attrs:
            return Failure(
                "'vcf_zarr_version' group attribute must be present", stop=True
            )


class CheckVcfZarrVersionIsSupported(ZarrCheck):
    def check(self, root):
        vcf_zarr_version = root.attrs["vcf_zarr_version"]
        if vcf_zarr_version != "0.4":
            return Failure(
                f"'vcf_zarr_version' must be '0.4', but was '{vcf_zarr_version}'",
                stop=True,
            )


class CheckAllArraysHaveDimensionNames(ZarrCheck):
    def check(self, root):
        missing_array_names = []
        for name in root.array_keys():
            arr = root[name]
            dims = arr.attrs.get("_ARRAY_DIMENSIONS", None)
            if dims is None:
                missing_array_names.append(name)
        if len(missing_array_names) > 0:
            return Failure(
                "Arrays must have dimension names, but they were missing for: "
                f"{",".join(missing_array_names)}",
                stop=True,
            )


class CheckDimensionNamesLenMatchesArrayDimensionsLen(ZarrCheck):
    def check(self, root):
        all_array_dim_counts = {}
        mismatched_names = []
        for name in root.array_keys():
            arr = root[name]
            dims = arr.attrs["_ARRAY_DIMENSIONS"]
            all_array_dim_counts[name] = {
                "dimension_names": len(dims),
                "ndim": arr.ndim,
            }
            if len(dims) != arr.ndim:
                mismatched_names.append(name)
        if len(mismatched_names) > 0:
            return Failure(
                "Number of dimension names must match array ndim, "
                f"but they were mismatched for: {','.join(mismatched_names)}.\n"
                "The dimension name counts and ndims were:\n"
                f"{pp.pformat(all_array_dim_counts)}",
                stop=True,
            )


class CheckDimensionNamesHaveConsistentSizes(ZarrCheck):
    def check(self, root):
        all_array_dimensions = {}
        dimension_name_to_size = {}
        inconsistent_dimension_names = set()
        for name in root.array_keys():
            arr = root[name]
            dims = arr.attrs["_ARRAY_DIMENSIONS"]
            all_array_dimensions[name] = dict(zip(dims, arr.shape))
            for dim, size in zip(dims, arr.shape):
                if dim in dimension_name_to_size:
                    existing_size = dimension_name_to_size[dim]
                    if existing_size != size:
                        inconsistent_dimension_names.add(dim)
                else:
                    dimension_name_to_size[dim] = size
        if len(inconsistent_dimension_names) > 0:
            return Failure(
                "Dimension names must have consistent sizes, but they were "
                f"inconsistent for: {",".join(inconsistent_dimension_names)}.\n"
                "The array dimensions and sizes were:\n"
                f"{pp.pformat(all_array_dimensions)}",
                stop=True,
            )


class CheckRequiredFieldsArePresent(ZarrCheck):
    def check(self, root):
        missing_required_field_names = set(REQUIRED_VARIABLE_NAMES) - set(
            root.array_keys()
        )
        if len(missing_required_field_names) > 0:
            missing_required_field_names = sorted(missing_required_field_names)
            return Failure(
                "Missing required fields: " f"{",".join(missing_required_field_names)}",
                stop=True,
            )


@dataclass
class CheckArrayDimensionNames(ZarrCheck):
    name: str
    expected_dimension_names: list[str]

    def check(self, root):
        arr = root[self.name]
        dims = arr.attrs["_ARRAY_DIMENSIONS"]
        if dims != self.expected_dimension_names:
            return Failure(
                f"Incorrect dimension names for '{self.name}': "
                f"expected {self.expected_dimension_names} but was {dims}",
            )


@dataclass
class CheckArrayDtypeKind(ZarrCheck):
    name: str
    expected_dtype_kind: str

    def check(self, root):
        arr = root[self.name]
        if arr.dtype.kind != self.expected_dtype_kind:
            return Failure(
                f"Incorrect dtype kind for '{self.name}': "
                f"expected '{self.expected_dtype_kind}' but was '{arr.dtype.kind}'",
            )


@dataclass
class CheckStringFieldHasVLenUTF8Filter(ZarrCheck):
    name: str

    def check(self, root):
        arr = root[self.name]
        has_vlen_utf8 = any(f.codec_id == "vlen-utf8" for f in arr.filters)
        if not has_vlen_utf8:
            return Failure(
                f"String field '{self.name}' must have a vlen-utf8 filter",
            )


def validate(path):
    failures = []

    path_checks = [
        CheckPathExists(),
        CheckPathIsZarrGroup(),
    ]

    for check in path_checks:
        result = check.check(path)
        if isinstance(result, Failure):
            failures.append(result)
            if result.stop:
                return failures

    root = zarr.open(path, mode="r")

    checks = [
        CheckZarrFormatIsV2(),
        CheckVcfZarrVersionGroupAttributeIsPresent(),
        CheckVcfZarrVersionIsSupported(),
        CheckAllArraysHaveDimensionNames(),
        CheckDimensionNamesLenMatchesArrayDimensionsLen(),
        CheckDimensionNamesHaveConsistentSizes(),
        CheckRequiredFieldsArePresent(),
        CheckArrayDimensionNames("variant_contig", ["variants"]),
        CheckArrayDimensionNames("variant_position", ["variants"]),
        CheckArrayDimensionNames("variant_id", ["variants"]),
        CheckArrayDimensionNames("variant_allele", ["variants", "alleles"]),
        CheckArrayDimensionNames("variant_quality", ["variants"]),
        CheckArrayDimensionNames("variant_filter", ["variants", "filters"]),
        CheckArrayDimensionNames("contig_id", ["contigs"]),
        CheckArrayDimensionNames("filter_id", ["filters"]),
        CheckArrayDimensionNames("sample_id", ["samples"]),
        CheckArrayDtypeKind("variant_contig", "i"),
        CheckArrayDtypeKind("variant_position", "i"),
        CheckArrayDtypeKind("variant_id", "T"),
        CheckArrayDtypeKind("variant_allele", "T"),
        CheckArrayDtypeKind("variant_quality", "f"),
        CheckArrayDtypeKind("variant_filter", "b"),
        CheckArrayDtypeKind("contig_id", "T"),
        CheckArrayDtypeKind("filter_id", "T"),
        CheckArrayDtypeKind("sample_id", "T"),
        CheckStringFieldHasVLenUTF8Filter("variant_id"),
        CheckStringFieldHasVLenUTF8Filter("variant_allele"),
        CheckStringFieldHasVLenUTF8Filter("contig_id"),
        CheckStringFieldHasVLenUTF8Filter("filter_id"),
        CheckStringFieldHasVLenUTF8Filter("sample_id"),
    ]

    for check in checks:
        result = check.check(root)
        if isinstance(result, Failure):
            failures.append(result)
            if result.stop:
                return failures

    return failures
