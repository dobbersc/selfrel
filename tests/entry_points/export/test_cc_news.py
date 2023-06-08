from pathlib import Path

from selfrel.entry_points.export.cc_news import export_cc_news


def test_export_cc_news(tmp_path: Path, resources_dir: Path) -> None:
    export_cc_news(out_dir=tmp_path, export_metadata=True, dataset_slice=":3", max_sentence_length=50, processes=1)

    result: str = (tmp_path / "cc-news.conllup").read_text(encoding="utf-8")
    expected: str = (resources_dir / "cc_news" / "cc-news.conllup").read_text(encoding="utf-8")
    assert result == expected

    metadata_result: str = (tmp_path / "metadata.json").read_text(encoding="utf-8")
    metadata_expected: str = (resources_dir / "cc_news" / "metadata.json").read_text(encoding="utf-8")
    assert metadata_result == metadata_expected
