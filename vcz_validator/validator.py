import pprint as pp
from dataclasses import dataclass
from pathlib import Path

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
    "filter_description",
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
            yield Failure(f"Path '{path}' does not exist", stop=True)


class CheckPathIsZarrGroup(PathCheck):
    def check(self, path):
        try:
            zarr.open(path, mode="r")
        except GroupNotFoundError:
            yield Failure(f"Path '{path}' is not a Zarr group", stop=True)


class CheckZarrFormatIsV2(ZarrCheck):
    def check(self, root):
        zarr_format = root.metadata.zarr_format
        if zarr_format != 2:
            yield Failure(f"Zarr format must be 2, but was {zarr_format}", stop=True)


class CheckVcfZarrVersionGroupAttributeIsPresent(ZarrCheck):
    def check(self, root):
        if "vcf_zarr_version" not in root.attrs:
            yield Failure(
                "'vcf_zarr_version' group attribute must be present", stop=True
            )


class CheckVcfZarrVersionIsSupported(ZarrCheck):
    def check(self, root):
        vcf_zarr_version = root.attrs["vcf_zarr_version"]
        if vcf_zarr_version != "0.4":
            yield Failure(
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
            yield Failure(
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
            yield Failure(
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
            yield Failure(
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
            yield Failure(
                "Missing required fields: " f"{",".join(missing_required_field_names)}",
                stop=True,
            )


@dataclass
class CheckArraySpec(ZarrCheck):
    name: str
    dimension_names: list[str]
    dtype_kind: str
    optional: bool = False

    def check(self, root):
        if self.optional and self.name not in root:
            return
        arr = root[self.name]

        dims = arr.attrs["_ARRAY_DIMENSIONS"]
        if dims != self.dimension_names:
            yield Failure(
                f"Incorrect dimension names for '{self.name}': "
                f"expected {self.dimension_names} but was {dims}",
            )

        if arr.dtype.kind != self.dtype_kind:
            yield Failure(
                f"Incorrect dtype kind for '{self.name}': "
                f"expected '{self.dtype_kind}' but was '{arr.dtype.kind}'",
            )

        if self.dtype_kind == "T":
            has_vlen_utf8 = any(f.codec_id == "vlen-utf8" for f in arr.filters)
            if not has_vlen_utf8:
                yield Failure(
                    f"String field '{self.name}' must have a vlen-utf8 filter",
                )


GENOTYPE_FIELD_NAMES = ["call_genotype", "call_genotype_phased"]

VALID_FIELD_DTYPE_KINDS = {"b", "i", "f", "T"}


class CheckInfoFields(ZarrCheck):
    def check(self, root):
        for name in root.array_keys():
            if not name.startswith("variant_") or name in REQUIRED_VARIABLE_NAMES:
                continue
            arr = root[name]
            dims = arr.attrs["_ARRAY_DIMENSIONS"]
            if len(dims) != 2 or dims[0] != "variants":
                yield Failure(
                    f"INFO field '{name}' must be 2-dimensional with dimensions "
                    f"['variants', ...], but had dimensions {dims}",
                )
            if arr.dtype.kind not in VALID_FIELD_DTYPE_KINDS:
                yield Failure(
                    f"INFO field '{name}' has invalid dtype kind '{arr.dtype.kind}'. "
                    "Must be one of: 'b' (bool), 'i' (int), 'f' (float), 'T' (string)",
                )


class CheckFormatFields(ZarrCheck):
    def check(self, root):
        for name in root.array_keys():
            if not name.startswith("call_") or name in GENOTYPE_FIELD_NAMES:
                continue
            arr = root[name]
            dims = arr.attrs["_ARRAY_DIMENSIONS"]
            if len(dims) != 3 or dims[0] != "variants" or dims[1] != "samples":
                yield Failure(
                    f"FORMAT field '{name}' must be 3-dimensional with dimensions "
                    f"['variants', 'samples', ...], but had dimensions {dims}",
                )
            if arr.dtype.kind not in VALID_FIELD_DTYPE_KINDS:
                yield Failure(
                    f"FORMAT field '{name}' has invalid dtype kind '{arr.dtype.kind}'. "
                    "Must be one of: 'b' (bool), 'i' (int), 'f' (float), 'T' (string)",
                )


def validate(path):
    path = Path(path)
    failures = []

    path_checks = [
        CheckPathExists(),
        CheckPathIsZarrGroup(),
    ]

    for check in path_checks:
        for failure in check.check(path):
            failures.append(failure)
            if failure.stop:
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
        CheckArraySpec("variant_contig", ["variants"], "i"),
        CheckArraySpec("variant_position", ["variants"], "i"),
        CheckArraySpec("variant_id", ["variants"], "T"),
        CheckArraySpec("variant_allele", ["variants", "alleles"], "T"),
        CheckArraySpec("variant_quality", ["variants"], "f"),
        CheckArraySpec("variant_filter", ["variants", "filters"], "b"),
        CheckArraySpec("contig_id", ["contigs"], "T"),
        CheckArraySpec("contig_length", ["contigs"], "i", optional=True),
        CheckArraySpec("filter_id", ["filters"], "T"),
        CheckArraySpec("filter_description", ["filters"], "T"),
        CheckArraySpec("sample_id", ["samples"], "T"),
        CheckArraySpec(
            "call_genotype", ["variants", "samples", "ploidy"], "i", optional=True
        ),
        CheckArraySpec(
            "call_genotype_phased", ["variants", "samples"], "b", optional=True
        ),
        CheckInfoFields(),
        CheckFormatFields(),
    ]

    for check in checks:
        for failure in check.check(root):
            failures.append(failure)
            if failure.stop:
                return failures

    return failures
