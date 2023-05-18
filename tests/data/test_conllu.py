from pathlib import Path

import pytest

from selfrel.data import CoNLLUPlusDataset


@pytest.mark.parametrize("persist", [True, False])
def test_conllu_plus_dataset(persist: bool, resources_dir: Path) -> None:
    dataset = CoNLLUPlusDataset(resources_dir / "cc-news-short.conllup", dataset_name="cc-news", persist=persist)
    original_texts: list[str] = [
        "The New York City Ballet Board of Directors announced on Saturday the interim team that has been appointed to run the artistic side of the company during ballet master in chief Peter Martins' leave of absence.",  # noqa: E501
        "Martins requested a temporary leave from both NYCB and the School of American Ballet last Thursday while the company undergoes an internal investigation into the sexual harassment accusations aimed at him.",  # noqa: E501
        "The four-person group is made up of members of the company's current artistic staff.",
    ]

    # Test properties
    assert dataset.dataset_path == resources_dir / "cc-news-short.conllup"
    assert dataset.dataset_name == "cc-news"
    assert dataset.is_persistent == persist

    # Test __getitem__
    assert len(dataset) == 3
    assert dataset[0].to_original_text() == original_texts[0]
    assert dataset[1].to_original_text() == original_texts[1]
    assert dataset[2].to_original_text() == original_texts[2]

    # Test __iter__
    assert len(dataset) == len(list(iter(dataset)))
    for sentence, original_text in zip(dataset, original_texts):
        assert sentence.to_original_text() == original_text
