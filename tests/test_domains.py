from pathlib import Path

from chemx.domains import DOMAIN_SLUGS, detect_domain, list_domains, output_schema


def test_all_domain_contracts_load_and_generate_strict_schema() -> None:
    specs = list_domains()
    assert [spec.slug for spec in specs] == list(DOMAIN_SLUGS)
    for spec in specs:
        schema = output_schema(spec)
        assert schema["additionalProperties"] is False
        assert schema["properties"]["domain"]["const"] == spec.slug
        values = schema["properties"]["records"]["items"]["properties"]["values"]
        assert values["additionalProperties"] is False
        assert set(values["properties"]) == {field.name for field in spec.fields}


def test_domain_detection_uses_path() -> None:
    path = Path("datasets/NANOMATERIALS/SelTox/article.pdf")
    assert detect_domain(path).slug == "seltox"

