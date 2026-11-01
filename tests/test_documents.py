import pytest

from nnserve.documents import load_document, parse_page_selection, render_pages
from nnserve.schemas import OcrDocument
from tests.helpers import PNG_DATA_URI


def test_data_uri_is_decoded():
    source = load_document(OcrDocument(type="image_url", image_url=PNG_DATA_URI))

    assert source.media_type == "image/png"
    assert source.size_bytes > 0


def test_page_selection_forms():
    assert parse_page_selection(None) is None
    assert parse_page_selection("3") == {3}
    assert parse_page_selection("0-2") == {0, 1, 2}
    assert parse_page_selection([0, 3, 4]) == {0, 3, 4}


def test_page_selection_rejects_negative():
    with pytest.raises(ValueError):
        parse_page_selection([-1])


def test_image_renders_one_page():
    source = load_document(OcrDocument(type="image_url", image_url=PNG_DATA_URI))
    pages = render_pages(source)

    assert len(pages) == 1
    assert pages[0].index == 0
    assert pages[0].width == 1
    assert pages[0].height == 1
    assert pages[0].dpi == 200.0


def test_image_page_selection_out_of_range():
    source = load_document(OcrDocument(type="image_url", image_url=PNG_DATA_URI))

    with pytest.raises(ValueError):
        render_pages(source, {1})
