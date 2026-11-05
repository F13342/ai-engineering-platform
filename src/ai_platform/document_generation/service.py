"""Generate Word, PDF, and PowerPoint files with Arabic/English support."""

from __future__ import annotations

import os
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from html import escape


SUPPORTED_FORMATS = {"docx", "pdf", "pptx"}
SUPPORTED_LANGUAGES = {"ar", "en"}

MEDIA_TYPES = {
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "pdf": "application/pdf",
    "pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
}


@dataclass(frozen=True)
class GeneratedDocument:
    """Metadata for a generated file."""

    path: Path
    filename: str
    media_type: str
    file_type: str


class DocumentGenerationService:
    """Render text content into downloadable office documents."""

    def __init__(self, output_dir: str | Path | None = None) -> None:
        configured_dir = os.environ.get("AI_PLATFORM_DOCUMENT_OUTPUT_DIR")
        selected_dir = output_dir or configured_dir or (
            Path.cwd() / "runtime_data" / "generated_documents"
        )
        self.output_dir = Path(selected_dir).resolve()

    def generate(
        self,
        *,
        title: str,
        content: str,
        output_format: str,
        language: str = "en",
    ) -> GeneratedDocument:
        """Validate the request, render a file, and return its metadata."""
        if not isinstance(title, str) or not title.strip():
            raise ValueError("title must not be blank")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("content must not be blank")

        output_format = output_format.lower().strip()
        language = language.lower().strip()

        if output_format not in SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported output format: {output_format}. "
                f"Supported formats: {', '.join(sorted(SUPPORTED_FORMATS))}"
            )

        if language not in SUPPORTED_LANGUAGES:
            raise ValueError("language must be 'ar' or 'en'")

        self.output_dir.mkdir(parents=True, exist_ok=True)

        safe_title = re.sub(r"[^\w-]+", "-", title.strip(), flags=re.UNICODE)
        safe_title = safe_title.strip("-_").lower()[:60] or "document"
        filename = f"{safe_title}-{uuid.uuid4().hex[:10]}.{output_format}"
        path = self.output_dir / filename

        blocks = self._parse_content(content, title)

        if output_format == "docx":
            self._render_docx(path, title, blocks, language)
        elif output_format == "pptx":
            self._render_pptx(path, title, blocks, language)
        else:
            self._render_pdf(path, title, blocks, language)

        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError("Document generation did not produce a valid file.")

        return GeneratedDocument(
            path=path,
            filename=filename,
            media_type=MEDIA_TYPES[output_format],
            file_type=output_format,
        )

    @staticmethod
    def _parse_content(content: str, title: str) -> list[tuple[str, str]]:
        """Recognize basic Markdown headings and list items."""
        blocks: list[tuple[str, str]] = []

        for raw_line in content.splitlines():
            line = raw_line.strip()
            if not line:
                continue

            heading = re.match(r"^(#{1,3})\s+(.+)$", line)
            if heading:
                text = heading.group(2).strip()
                if text.casefold() != title.strip().casefold():
                    blocks.append(("heading", text))
                continue

            bullet = re.match(r"^(?:[-*•]|\d+[.)])\s+(.+)$", line)
            if bullet:
                blocks.append(("bullet", bullet.group(1).strip()))
                continue

            text = line.replace("**", "").replace("__", "").replace("`", "")
            if text.casefold() != title.strip().casefold():
                blocks.append(("body", text))

        return blocks

    @staticmethod
    def _set_docx_direction(paragraph, language: str) -> None:
        from docx.enum.text import WD_ALIGN_PARAGRAPH
        from docx.oxml import OxmlElement
        from docx.oxml.ns import qn

        rtl = language == "ar"
        paragraph.alignment = (
            WD_ALIGN_PARAGRAPH.RIGHT if rtl else WD_ALIGN_PARAGRAPH.LEFT
        )

        p_pr = paragraph._p.get_or_add_pPr()
        bidi = p_pr.find(qn("w:bidi"))
        if bidi is None:
            bidi = OxmlElement("w:bidi")
            p_pr.append(bidi)
        bidi.set(qn("w:val"), "1" if rtl else "0")

    @staticmethod
    def _style_docx_runs(paragraph, language: str) -> None:
        from docx.oxml import OxmlElement
        from docx.oxml.ns import qn

        for run in paragraph.runs:
            run.font.name = "Arial"
            run.font.size = __import__("docx").shared.Pt(11)

            r_pr = run._element.get_or_add_rPr()
            r_fonts = r_pr.rFonts
            if r_fonts is None:
                r_fonts = OxmlElement("w:rFonts")
                r_pr.insert(0, r_fonts)

            for attribute in ("ascii", "hAnsi", "cs", "eastAsia"):
                r_fonts.set(qn(f"w:{attribute}"), "Arial")

            rtl = language == "ar"
            rtl_element = r_pr.find(qn("w:rtl"))
            if rtl_element is None:
                rtl_element = OxmlElement("w:rtl")
                r_pr.append(rtl_element)
            rtl_element.set(qn("w:val"), "1" if rtl else "0")

            lang_element = r_pr.find(qn("w:lang"))
            if lang_element is None:
                lang_element = OxmlElement("w:lang")
                r_pr.append(lang_element)
            lang_element.set(qn("w:val"), "ar-SA" if rtl else "en-US")

    def _render_docx(
        self, path: Path, title: str,
        blocks: list[tuple[str, str]], language: str,
    ) -> None:
        from docx import Document
        from docx.shared import Inches, Pt

        document = Document()
        section = document.sections[0]
        section.top_margin = Inches(0.7)
        section.bottom_margin = Inches(0.7)
        section.left_margin = Inches(0.8)
        section.right_margin = Inches(0.8)

        document.core_properties.title = title

        title_paragraph = document.add_heading(title, level=0)
        self._set_docx_direction(title_paragraph, language)
        for run in title_paragraph.runs:
            run.font.name = "Arial"
            run.font.size = Pt(22)
        self._style_docx_runs(title_paragraph, language)

        for kind, text in blocks:
            if kind == "heading":
                paragraph = document.add_heading(text, level=2)
            elif kind == "bullet":
                paragraph = document.add_paragraph("• " + text)
            else:
                paragraph = document.add_paragraph(text)

            self._set_docx_direction(paragraph, language)
            self._style_docx_runs(paragraph, language)

        document.save(path)

    @staticmethod
    def _set_pptx_paragraph(paragraph, language: str) -> None:
        from pptx.enum.text import PP_ALIGN

        rtl = language == "ar"
        paragraph.alignment = PP_ALIGN.RIGHT if rtl else PP_ALIGN.LEFT
        p_pr = paragraph._p.get_or_add_pPr()
        p_pr.set("rtl", "1" if rtl else "0")

    def _render_pptx(
        self, path: Path, title: str,
        blocks: list[tuple[str, str]], language: str,
    ) -> None:
        from pptx import Presentation
        from pptx.util import Inches, Pt
        from pptx.dml.color import RGBColor

        presentation = Presentation()
        presentation.slide_width = Inches(13.333)
        presentation.slide_height = Inches(7.5)

        rtl = language == "ar"

        # Cover slide
        cover = presentation.slides.add_slide(presentation.slide_layouts[6])
        title_box = cover.shapes.add_textbox(
            Inches(0.7), Inches(2.2), Inches(12), Inches(1.6)
        )
        title_frame = title_box.text_frame
        title_frame.word_wrap = True
        title_paragraph = title_frame.paragraphs[0]
        title_paragraph.text = title
        self._set_pptx_paragraph(title_paragraph, language)

        for run in title_paragraph.runs:
            run.font.name = "Arial"
            run.font.size = Pt(30)
            run.font.bold = True
            run.font.color.rgb = RGBColor(30, 50, 75)

        # Content slides: at most five content blocks per slide
        chunks = [blocks[i:i + 5] for i in range(0, len(blocks), 5)]
        if not chunks:
            chunks = [[]]

        for page_number, chunk in enumerate(chunks, start=1):
            slide = presentation.slides.add_slide(presentation.slide_layouts[6])

            heading_box = slide.shapes.add_textbox(
                Inches(0.65), Inches(0.35), Inches(12), Inches(0.8)
            )
            heading_paragraph = heading_box.text_frame.paragraphs[0]
            heading_paragraph.text = f"{title} — {page_number}"
            self._set_pptx_paragraph(heading_paragraph, language)

            for run in heading_paragraph.runs:
                run.font.name = "Arial"
                run.font.size = Pt(23)
                run.font.bold = True
                run.font.color.rgb = RGBColor(32, 120, 110)

            body_box = slide.shapes.add_textbox(
                Inches(0.8), Inches(1.4), Inches(11.8), Inches(5.5)
            )
            body_frame = body_box.text_frame
            body_frame.word_wrap = True
            body_frame.clear()

            for index, (kind, text) in enumerate(chunk):
                paragraph = (
                    body_frame.paragraphs[0]
                    if index == 0
                    else body_frame.add_paragraph()
                )
                paragraph.text = ("• " if kind == "bullet" else "") + text
                paragraph.space_after = Pt(10)
                self._set_pptx_paragraph(paragraph, language)

                for run in paragraph.runs:
                    run.font.name = "Arial"
                    run.font.size = Pt(17 if kind != "heading" else 20)
                    run.font.bold = kind == "heading"

        presentation.save(path)

    @staticmethod
    def _find_pdf_fonts():
        """Find a font with Arabic glyphs on Windows or Linux."""
        configured_font = os.environ.get("AI_PLATFORM_ARABIC_FONT")
        candidates = [
            (configured_font, None) if configured_font else (None, None),
            (r"C:\Windows\Fonts\arial.ttf", r"C:\Windows\Fonts\arialbd.ttf"),
            ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
             "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
            ("/usr/share/fonts/truetype/freefont/FreeSans.ttf",
             "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf"),
        ]

        for normal, bold in candidates:
            if normal and Path(normal).is_file():
                bold_path = Path(bold) if bold else Path(normal)
                if bold_path.is_file():
                    return Path(normal), bold_path

        return None, None

    def _render_pdf(
        self, path: Path, title: str,
        blocks: list[tuple[str, str]], language: str,
    ) -> None:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle
        from reportlab.lib.enums import TA_RIGHT, TA_LEFT, TA_CENTER
        from reportlab.lib.units import cm
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
        from reportlab.lib.colors import HexColor

        rtl = language == "ar"
        normal_font = "Helvetica"
        bold_font = "Helvetica-Bold"
        reshape_arabic = None

        if rtl:
            try:
                import arabic_reshaper
                from bidi.algorithm import get_display
            except ImportError as exc:
                raise RuntimeError(
                    "Arabic PDF needs arabic-reshaper and python-bidi."
                ) from exc

            normal_path, bold_path = self._find_pdf_fonts()
            if normal_path is None or bold_path is None:
                raise RuntimeError(
                    "Arabic PDF needs a font. Set AI_PLATFORM_ARABIC_FONT "
                    "to a suitable TTF font file."
                )

            normal_font = "AIGeneratedArabic"
            bold_font = "AIGeneratedArabicBold"
            registered = set(pdfmetrics.getRegisteredFontNames())

            if normal_font not in registered:
                pdfmetrics.registerFont(TTFont(normal_font, str(normal_path)))
            if bold_font not in registered:
                pdfmetrics.registerFont(TTFont(bold_font, str(bold_path)))

            reshape_arabic = lambda value: get_display(
                arabic_reshaper.reshape(value)
            )

        title_style = ParagraphStyle(
            "GeneratedTitle",
            fontName=bold_font,
            fontSize=18,
            leading=27,
            alignment=TA_CENTER,
            spaceAfter=18,
        )
        heading_style = ParagraphStyle(
            "GeneratedHeading",
            fontName=bold_font,
            fontSize=14,
            leading=22,
            alignment=TA_RIGHT if rtl else TA_LEFT,
            spaceBefore=10,
            spaceAfter=6,
        )
        body_style = ParagraphStyle(
            "GeneratedBody",
            fontName=normal_font,
            fontSize=11,
            leading=19 if rtl else 17,
            alignment=TA_RIGHT if rtl else TA_LEFT,
            spaceAfter=8,
        )

        document = SimpleDocTemplate(
            str(path),
            pagesize=A4,
            rightMargin=2 * cm,
            leftMargin=2 * cm,
            topMargin=2 * cm,
            bottomMargin=2 * cm,
            title=title,
        )

        story = []
        visible_title = reshape_arabic(title) if rtl else title
        story.append(Paragraph(escape(visible_title), title_style))
        story.append(Spacer(1, 0.2 * cm))

        max_width = A4[0] - 4 * cm

        for kind, text in blocks:
            value = ("• " + text) if kind == "bullet" else text

            if rtl and reshape_arabic is not None:
                # Shape the Arabic after wrapping so line order stays stable.
                words = value.split()
                lines: list[str] = []
                current: list[str] = []

                for word in words:
                    candidate = " ".join(current + [word])
                    rendered = reshape_arabic(candidate)
                    width = pdfmetrics.stringWidth(
                        rendered,
                        bold_font if kind == "heading" else normal_font,
                        14 if kind == "heading" else 11,
                    )

                    if current and width > max_width * 0.94:
                        lines.append(reshape_arabic(" ".join(current)))
                        current = [word]
                    else:
                        current.append(word)

                if current:
                    lines.append(reshape_arabic(" ".join(current)))

                paragraph_text = "<br/>".join(escape(line) for line in lines)
            else:
                paragraph_text = escape(value)

            story.append(
                Paragraph(
                    paragraph_text,
                    heading_style if kind == "heading" else body_style,
                )
            )

        document.build(story)
