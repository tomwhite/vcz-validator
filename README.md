# vcz-validator

A validator for [VCF Zarr](https://github.com/pystatgen/vcf-zarr-spec) stores.

## Installation

```bash
pip install vcz-validator
```

## Usage

### Command line

```bash
vcz-validate path/to/store.vcz
```

### Python

```python
from vcz_validator import validate

failures = validate("path/to/store.vcz")
for failure in failures:
    print(failure.message)
```
