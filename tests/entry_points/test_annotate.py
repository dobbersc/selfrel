from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from selfrel.data import CoNLLUPlusDataset
from selfrel.entry_points.annotate import annotate

if TYPE_CHECKING:
    from flair.data import Sentence

# TODO: Add test for "relation" and "sentence" abstraction level


@pytest.mark.usefixtures("_init_ray")
def test_annotate_tokens(tmp_path: Path, resources_dir: Path) -> None:
    annotate(
        resources_dir / "cc-news-short.conllup",
        out=tmp_path / "cc-news-pos.conllup",
        model="flair/upos-english-fast",
        num_gpus=0,
    )

    result: CoNLLUPlusDataset = CoNLLUPlusDataset(tmp_path / "cc-news-pos.conllup")
    assert result

    for sentence in result:
        for token in sentence:
            assert token.get_label("pos").value


@pytest.mark.usefixtures("_init_ray")
def test_annotate_span(tmp_path: Path, resources_dir: Path) -> None:
    annotate(
        resources_dir / "cc-news-short.conllup",
        out=tmp_path / "cc-news-ner.conllup",
        model="flair/ner-english-fast",
        num_gpus=0,
    )

    result: CoNLLUPlusDataset = CoNLLUPlusDataset(tmp_path / "cc-news-ner.conllup")
    assert result

    sentence_with_entities: Sentence = result[1]
    assert {(entity.text, entity.get_label("ner").value) for entity in sentence_with_entities.get_spans("ner")} == {
        ("Martins", "PER"),
        ("NYCB", "ORG"),
        ("School of American Ballet", "ORG"),
    }

    sentence_without_entities: Sentence = result[5]
    assert not sentence_without_entities.get_spans("ner")


@pytest.mark.usefixtures("_init_ray")
def test_annotate_on_annotated_dataset(tmp_path: Path, resources_dir: Path) -> None:
    dataset: CoNLLUPlusDataset = CoNLLUPlusDataset(resources_dir / "cc-news-short.conllup")
    dataset[1][:1].add_label("test", value="test_value")

    annotate(
        dataset,
        out=tmp_path / "cc-news-annotated.conllup",
        model="flair/ner-english-fast",
        num_gpus=0,
    )

    result: CoNLLUPlusDataset = CoNLLUPlusDataset(tmp_path / "cc-news-annotated.conllup")
    assert result

    assert result[1].get_label("test").value == "test_value"
