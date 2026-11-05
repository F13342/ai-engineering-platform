import pytest
from pathlib import Path

from ai_platform.document_generation import DocumentGenerationService


ARABIC_CONTENT = """# تقرير اختبار
## مقدمة
هذا تقرير تجريبي باللغة العربية.
## الاستخدامات
- الاستخدام الأول
- الاستخدام الثاني
## الخاتمة
هذه خاتمة التقرير."""

ENGLISH_CONTENT = """# Test Report
## Introduction
This is a sample report in English.
## Applications
- First application
- Second application
## Conclusion
This is the report conclusion."""


@pytest.mark.parametrize("output_format", ["docx", "pdf", "pptx"])
@pytest.mark.parametrize(
    ("language", "title", "content"),
    [
        ("ar", "تقرير اختبار", ARABIC_CONTENT),
        ("en", "Test Report", ENGLISH_CONTENT),
    ],
)
def test_generate_document_in_supported_format_and_language(
    tmp_path: Path,
    output_format: str,
    language: str,
    title: str,
    content: str,
) -> None:
    service = DocumentGenerationService(output_dir=tmp_path)

    result = service.generate(
        title=title,
        content=content,
        output_format=output_format,
        language=language,
    )

    assert result.path.is_file()
    assert result.path.parent == tmp_path
    assert result.file_type == output_format
    assert result.filename.endswith(f".{output_format}")
    assert result.media_type
    assert result.path.stat().st_size > 100


@pytest.mark.parametrize("language", ["ar", "en"])
def test_docx_contains_content_and_direction(
    tmp_path: Path,
    language: str,
) -> None:
    from docx import Document
    from docx.oxml.ns import qn

    title = "اختبار العربية" if language == "ar" else "English Test"
    content = (
        "## مقدمة\nهذا محتوى للاختبار."
        if language == "ar"
        else "## Introduction\nThis is test content."
    )

    result = DocumentGenerationService(output_dir=tmp_path).generate(
        title=title,
        content=content,
        output_format="docx",
        language=language,
    )

    document = Document(result.path)
    all_text = "\n".join(paragraph.text for paragraph in document.paragraphs)

    assert title in all_text
    assert ("محتوى للاختبار" in all_text) if language == "ar" else (
        "test content" in all_text
    )

    paragraphs_with_direction = []
    for paragraph in document.paragraphs:
        p_pr = paragraph._p.pPr
        if p_pr is not None:
            bidi = p_pr.find(qn("w:bidi"))
            if bidi is not None:
                paragraphs_with_direction.append(bidi.get(qn("w:val")))

    expected_direction = "1" if language == "ar" else "0"
    assert expected_direction in paragraphs_with_direction


def test_pptx_sets_paragraph_direction_for_arabic(
    tmp_path: Path,
) -> None:
    from pptx import Presentation

    result = DocumentGenerationService(output_dir=tmp_path).generate(
        title="اختبار العرض",
        content="## مقدمة\nهذا عرض تجريبي.\n- النقطة الأولى",
        output_format="pptx",
        language="ar",
    )

    presentation = Presentation(result.path)
    directions = []

    for slide in presentation.slides:
        for shape in slide.shapes:
            if not shape.has_text_frame:
                continue
            for paragraph in shape.text_frame.paragraphs:
                p_pr = paragraph._p.pPr
                if p_pr is not None:
                    directions.append(p_pr.get("rtl"))

    assert "1" in directions


@pytest.mark.parametrize(
    ("title", "content", "output_format", "language"),
    [
        ("", "some content", "docx", "en"),
        ("Some title", "  ", "docx", "en"),
        ("Some title", "some content", "txt", "en"),
        ("Some title", "some content", "docx", "fr"),
    ],
)
def test_invalid_generation_request_is_rejected(
    tmp_path: Path,
    title: str,
    content: str,
    output_format: str,
    language: str,
) -> None:
    service = DocumentGenerationService(output_dir=tmp_path)

    with pytest.raises(ValueError):
        service.generate(
            title=title,
            content=content,
            output_format=output_format,
            language=language,
        )
