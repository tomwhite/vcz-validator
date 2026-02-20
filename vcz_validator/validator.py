import pprint as pp
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import zarr
from zarr.errors import GroupNotFoundError

REQUIRED_VARIABLE_NAMES = [
    "variant_contig",
    "variant_position",
    "variant_allele",
    "contig_id",
    "sample_id",
]

OPTIONAL_VARIABLE_NAMES = [
    "variant_id",
    "variant_quality",
    "variant_filter",
    "filter_id",
    "filter_description",
    "variant_length",
]

GENOTYPE_VARIABLE_NAMES = ["call_genotype", "call_genotype_phased"]

RESERVED_VARIABLE_NAMES = (
    REQUIRED_VARIABLE_NAMES + OPTIONAL_VARIABLE_NAMES + GENOTYPE_VARIABLE_NAMES
)


class Datatype(Enum):
    BOOL = "bool", ["b"]
    INT = "int", ["i"]
    FLOAT = "float", ["f"]
    CHAR = "char", ["U"]
    STR = "str", ["S", "U", "T"]

    def __str__(self) -> str:
        return self.value[0]

    def is_kind(self, dtype_kind):
        return dtype_kind in self.value[1]

    @classmethod
    def is_valid(cls, dtype_kind):
        return any([d.is_kind(dtype_kind) for d in cls])


VALID_FIELD_DTYPE_KINDS_MESSAGE = (
    "Must be one of: 'b' (bool), 'i' (int), 'f' (float), 'S','U','T' (str)"
)


@dataclass
class Failure:
    message: str
    stop: bool = False


class CheckPathExists:
    def check(self, path):
        if not path.exists():
            yield Failure(f"Path '{path}' does not exist", stop=True)


class CheckPathIsZarrGroup:
    def check(self, path):
        try:
            zarr.open(path, mode="r")
        except GroupNotFoundError:
            yield Failure(f"Path '{path}' is not a Zarr group", stop=True)


class CheckZarrFormatIsV2:
    def check(self, root):
        zarr_format = root.metadata.zarr_format
        if zarr_format != 2:
            yield Failure(f"Zarr format must be 2, but was {zarr_format}", stop=True)


class CheckVcfZarrVersionGroupAttributeIsPresent:
    def check(self, root):
        if "vcf_zarr_version" not in root.attrs:
            yield Failure(
                "'vcf_zarr_version' group attribute must be present", stop=True
            )


class CheckVcfZarrVersionIsSupported:
    def check(self, root):
        vcf_zarr_version = root.attrs["vcf_zarr_version"]
        if vcf_zarr_version != "0.4":
            yield Failure(
                f"'vcf_zarr_version' must be '0.4', but was '{vcf_zarr_version}'",
                stop=True,
            )


class CheckSourceAttribute:
    def check(self, root):
        if "source" in root.attrs and not isinstance(root.attrs["source"], str):
            yield Failure(
                "'source' group attribute must be a string, but was "
                + type(root.attrs["source"]).__name__,
            )


class CheckVcfMetaInformationAttribute:
    def check(self, root):
        if "vcf_meta_information" in root.attrs and not isinstance(
            root.attrs["vcf_meta_information"], list
        ):
            yield Failure(
                "'vcf_meta_information' group attribute must be a list, but was "
                + type(root.attrs["vcf_meta_information"]).__name__,
            )


class CheckAllArraysHaveDimensionNames:
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


class CheckDimensionNamesLenMatchesArrayDimensionsLen:
    def check(self, root):
        all_array_dim_counts = {}
        mismatched_names = []
        for name in root.array_keys():
            arr = root[name]
            dims = _dimension_names(arr)
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


class CheckDimensionNamesHaveConsistentSizes:
    def check(self, root):
        all_array_dimensions = {}
        dimension_name_to_size = {}
        inconsistent_dimension_names = set()
        for name in root.array_keys():
            arr = root[name]
            dims = _dimension_names(arr)
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


class CheckRequiredFieldsArePresent:
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
class CheckArraySpec:
    name: str
    dimension_names: list[str]
    datatype: Datatype
    optional: bool = False

    def check(self, root):
        if self.optional and self.name not in root:
            return
        arr = root[self.name]

        dims = _dimension_names(arr)
        if dims != self.dimension_names:
            yield Failure(
                f"Incorrect dimension names for '{self.name}': "
                f"expected {self.dimension_names} but was {dims}",
            )

        if not self.datatype.is_kind(arr.dtype.kind):
            yield Failure(
                f"Incorrect datatype for '{self.name}': "
                f"expected '{self.datatype}' but array dtype kind was "
                f"'{arr.dtype.kind}'",
            )

        if self.datatype == Datatype.STR:
            has_vlen_utf8 = any(f.codec_id == "vlen-utf8" for f in arr.filters)
            if not has_vlen_utf8:
                yield Failure(
                    f"String field '{self.name}' must have a vlen-utf8 filter",
                )


def _dimension_names(arr):
    return arr.attrs["_ARRAY_DIMENSIONS"]


def _is_mask_or_fill(name):
    return name.endswith("_mask") or name.endswith("_fill")


class CheckInfoFields:
    def check(self, root):
        for name in root.array_keys():
            if not name.startswith("variant_") or name in RESERVED_VARIABLE_NAMES:
                continue
            if _is_mask_or_fill(name):
                continue
            arr = root[name]
            dims = _dimension_names(arr)
            if (len(dims) != 1 and len(dims) != 2) or dims[0] != "variants":
                yield Failure(
                    f"INFO field '{name}' must be 1- or 2-dimensional with dimensions "
                    f"['variants', ...], but had dimensions {dims}",
                )
            if not Datatype.is_valid(arr.dtype.kind):
                yield Failure(
                    f"INFO field '{name}' has invalid dtype kind '{arr.dtype.kind}'. "
                    + VALID_FIELD_DTYPE_KINDS_MESSAGE,
                )


class CheckFormatFields:
    def check(self, root):
        for name in root.array_keys():
            if not name.startswith("call_") or name in GENOTYPE_VARIABLE_NAMES:
                continue
            if _is_mask_or_fill(name):
                continue
            arr = root[name]
            dims = _dimension_names(arr)
            if (
                (len(dims) != 2 and len(dims) != 3)
                or dims[0] != "variants"
                or dims[1] != "samples"
            ):
                yield Failure(
                    f"FORMAT field '{name}' must be 2- or 3-dimensional with "
                    "dimensions ['variants', 'samples', ...], "
                    f"but had dimensions {dims}",
                )
            if not Datatype.is_valid(arr.dtype.kind):
                yield Failure(
                    f"FORMAT field '{name}' has invalid dtype kind '{arr.dtype.kind}'. "
                    + VALID_FIELD_DTYPE_KINDS_MESSAGE,
                )


class CheckMaskAndFillArrays:
    def check(self, root):
        array_names = set(root.array_keys())
        for name in array_names:
            for suffix, label in [("_mask", "Mask"), ("_fill", "Fill")]:
                if not name.endswith(suffix):
                    continue
                parent_name = name[: -len(suffix)]
                if parent_name not in array_names:
                    continue
                arr = root[name]
                parent_arr = root[parent_name]
                if not Datatype.BOOL.is_kind(arr.dtype.kind):
                    yield Failure(
                        f"{label} array '{name}' must have dtype kind 'b' (bool), "
                        f"but was '{arr.dtype.kind}'",
                    )
                if arr.shape != parent_arr.shape:
                    yield Failure(
                        f"{label} array '{name}' must have the same shape as "
                        f"'{parent_name}' {parent_arr.shape}, but was {arr.shape}",
                    )


def _run_checks(checks, arg):
    failures = []
    for check in checks:
        for failure in check.check(arg):
            failures.append(failure)
            if failure.stop:
                return failures, True
    return failures, False


def validate(path):
    path = Path(path)

    failures, stopped = _run_checks(
        [
            CheckPathExists(),
            CheckPathIsZarrGroup(),
        ],
        path,
    )
    if stopped:
        return failures

    root = zarr.open(path, mode="r")

    zarr_failures, _ = _run_checks(
        [
            CheckZarrFormatIsV2(),
            CheckVcfZarrVersionGroupAttributeIsPresent(),
            CheckVcfZarrVersionIsSupported(),
            CheckSourceAttribute(),
            CheckVcfMetaInformationAttribute(),
            CheckAllArraysHaveDimensionNames(),
            CheckDimensionNamesLenMatchesArrayDimensionsLen(),
            CheckDimensionNamesHaveConsistentSizes(),
            CheckRequiredFieldsArePresent(),
            CheckArraySpec("variant_contig", ["variants"], Datatype.INT),
            CheckArraySpec("variant_position", ["variants"], Datatype.INT),
            CheckArraySpec("variant_id", ["variants"], Datatype.STR, optional=True),
            CheckArraySpec("variant_allele", ["variants", "alleles"], Datatype.STR),
            CheckArraySpec(
                "variant_quality", ["variants"], Datatype.FLOAT, optional=True
            ),
            CheckArraySpec(
                "variant_filter", ["variants", "filters"], Datatype.BOOL, optional=True
            ),
            CheckArraySpec("contig_id", ["contigs"], Datatype.STR),
            CheckArraySpec("contig_length", ["contigs"], Datatype.INT, optional=True),
            CheckArraySpec("filter_id", ["filters"], Datatype.STR, optional=True),
            CheckArraySpec(
                "filter_description", ["filters"], Datatype.STR, optional=True
            ),
            CheckArraySpec("sample_id", ["samples"], Datatype.STR),
            CheckArraySpec("variant_length", ["variants"], Datatype.INT, optional=True),
            CheckArraySpec(
                "call_genotype",
                ["variants", "samples", "ploidy"],
                Datatype.INT,
                optional=True,
            ),
            CheckArraySpec(
                "call_genotype_phased",
                ["variants", "samples"],
                Datatype.BOOL,
                optional=True,
            ),
            CheckInfoFields(),
            CheckFormatFields(),
            CheckMaskAndFillArrays(),
        ],
        root,
    )
    failures.extend(zarr_failures)

    return failures
