from pathlib import Path

import pytest

from scripts.data.build_card_training_view import (
    convert_annotation,
    normalize_card_code,
)


@pytest.mark.parametrize(
    ("source", "expected"),
    [("Ah", "AH"), ("10d", "10D"), ("jC", "JC"), ("QS", "QS")],
)
def test_normalize_card_code(source: str, expected: str) -> None:
    assert normalize_card_code(source) == expected


def test_convert_pascal_voc_annotation_by_name(tmp_path: Path) -> None:
    xml_path = tmp_path / "sample.xml"
    xml_path.write_text(
        """<annotation>
<filename>sample.jpg</filename>
<size><width>200</width><height>100</height><depth>3</depth></size>
<object><name>Ah</name><bndbox>
<xmin>20</xmin><ymin>10</ymin><xmax>60</xmax><ymax>30</ymax>
</bndbox></object>
</annotation>""",
        encoding="utf-8",
    )

    filename, rows, counts = convert_annotation(xml_path, {"AH": 7})

    assert filename == "sample.jpg"
    assert rows == ["7 0.20000000 0.20000000 0.20000000 0.20000000"]
    assert counts == {7: 1}


def test_reject_invalid_card_code() -> None:
    with pytest.raises(ValueError):
        normalize_card_code("1x")
