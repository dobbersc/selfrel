import unicodedata
from collections.abc import Iterator
from typing import Final, Literal, Optional

import regex
from syntok import segmenter

__all__ = ["preprocess", "segment"]


__CONTROL_CHARACTERS_EXCEPT_TAB_CR_LF: Final[str] = "[^\\P{C}\t\r\n]"
__VS16_CHARACTER: Final[str] = "\ufe0f"

__PROBLEM_CHARACTERS_PATTERN: Final[regex.Pattern[str]] = regex.compile(
    f"{__CONTROL_CHARACTERS_EXCEPT_TAB_CR_LF}|{__VS16_CHARACTER}",
    flags=regex.UNICODE,
)


def preprocess(text: str, normalization: Optional[Literal["NFD", "NFC", "NFKD", "NFKC"]] = None) -> str:
    """
    Preprocesses the given text with the following steps:

    1.  Normalize according to the Unicode standard
    2.  Filter problem characters that may lead to machine learning model crashes.
        More precisely, these problem character are:
            - All Control characters (Cc) except `\t`, `\r` and `\n`
            - All Formatting characters (Cf)
            - All Private Use characters (Co)
            - All Surrogate characters (Cs)
            - The Variation Selector-16 (VS16) character

    :param text: The text to preprocess
    :param normalization: An optional Unicode normalization form
    :return: The preprocessed text
    """
    if normalization is not None:
        text = unicodedata.normalize(normalization, text)
    text = __PROBLEM_CHARACTERS_PATTERN.sub("", text)
    return text


def segment(document: str) -> Iterator[Iterator[list[segmenter.Token]]]:
    """
    Segments a document into paragraphs, sentences, and tokens.

    Note that hyphenated words at linebreaks are joined.

    :param document: The document to process
    :return: An iterator over paragraphs and sentences as lists of tokens
    """
    tokenizer = segmenter.Tokenizer(emit_hyphen_or_underscore_sep=True, replace_not_contraction=False)
    for paragraph in segmenter.preprocess(document):
        yield segmenter.segment(tokenizer.tokenize(paragraph))
