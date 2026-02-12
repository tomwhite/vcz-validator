import click

from vcz_validator.validator import validate as validate_vcz


@click.command()
@click.argument("path", type=click.Path())
def validate(path):
    """Validate a VCZ store"""
    failures = validate_vcz(path)

    if len(failures) == 0:
        print("Success")
    else:
        for failure in failures:
            print(f"Failure:\n{failure.message}\n")
        print(f"Failures: {len(failures)}")
