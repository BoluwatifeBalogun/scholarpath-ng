"""Single-student prediction report as a PDF."""
import io

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

from . import config

GREEN = colors.HexColor("#141414")
INK = colors.HexColor("#141414")
MUTED = colors.HexColor("#6B7066")
LINE = colors.HexColor("#141414")
PALE = colors.HexColor("#FFF6CF")


def prediction_pdf(p, vm) -> bytes:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm,
                            topMargin=16 * mm, bottomMargin=16 * mm,
                            title=f"Prediction report {p.id}", author=config.APP_NAME)
    base = getSampleStyleSheet()["Normal"]
    body = ParagraphStyle("body", parent=base, fontName="Helvetica", fontSize=9.5,
                          leading=13.5, textColor=INK)
    small = ParagraphStyle("small", parent=body, fontSize=8.5, leading=12, textColor=MUTED)
    h1 = ParagraphStyle("h1", parent=body, fontName="Helvetica-Bold", fontSize=18, leading=22)
    h2 = ParagraphStyle("h2", parent=body, fontName="Helvetica-Bold", fontSize=11.5,
                        leading=15, spaceBefore=10, spaceAfter=5, textColor=GREEN)
    big = ParagraphStyle("big", parent=body, fontName="Helvetica-Bold", fontSize=22, leading=26)

    r = p.result
    story = [
        Paragraph(config.APP_NAME, ParagraphStyle("brand", parent=body, textColor=GREEN,
                                                  fontName="Helvetica-Bold", fontSize=10)),
        Paragraph("Academic performance prediction report", h1),
        Paragraph(f"Report {p.id} &nbsp;|&nbsp; {p.created_at.strftime('%d %B %Y, %H:%M')} UTC "
                  f"&nbsp;|&nbsp; Decision Tree model v{p.model.version}", small),
        Spacer(1, 8),
    ]

    head = Table([[
        [Paragraph("Student reference", small),
         Paragraph(p.reference or "Not given", body),
         Spacer(1, 6),
         Paragraph("Institution type", small), Paragraph(p.institution_type, body)],
        [Paragraph("Predicted performance class", small),
         Paragraph(p.label, big),
         Paragraph(f"{vm['meaning']}. {p.confidence * 100:.0f}% of the {r['leaf_samples']} "
                   "training records that followed the same route fell in this class.", small)],
    ]], colWidths=[60 * mm, 114 * mm])
    head.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), PALE), ("BOX", (0, 0), (-1, -1), 1.2, LINE),
        ("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10), ("TOPPADDING", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10)]))
    story.append(head)

    def grid(rows, widths, header=True):
        t = Table(rows, colWidths=widths, repeatRows=1 if header else 0)
        style = [("FONTNAME", (0, 0), (-1, -1), "Helvetica"), ("FONTSIZE", (0, 0), (-1, -1), 9),
                 ("TEXTCOLOR", (0, 0), (-1, -1), INK), ("LINEBELOW", (0, 0), (-1, -1), 0.4, colors.HexColor("#D9DCD3")),
                 ("VALIGN", (0, 0), (-1, -1), "TOP"), ("TOPPADDING", (0, 0), (-1, -1), 4),
                 ("BOTTOMPADDING", (0, 0), (-1, -1), 4)]
        if header:
            style += [("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                      ("TEXTCOLOR", (0, 0), (-1, 0), MUTED)]
        t.setStyle(TableStyle(style))
        return t

    story.append(Paragraph("How the matching records were spread", h2))
    story.append(grid([["Class", "Share of matching training records"]] +
                      [[x["label"], f"{x['p'] * 100:.1f}%"] for x in vm["probs"]],
                      [70 * mm, 104 * mm]))

    story.append(Paragraph("The record as entered", h2))
    rows = vm["rows"]
    half = (len(rows) + 1) // 2
    two = []
    for i in range(half):
        left = rows[i]
        right = rows[i + half] if i + half < len(rows) else {"label": "", "value": ""}
        two.append([left["label"], left["value"], right["label"], right["value"]])
    story.append(grid(two, [52 * mm, 35 * mm, 52 * mm, 35 * mm], header=False))

    story.append(Paragraph("The questions the tree asked", h2))
    story.append(grid([["No.", "Test satisfied", "Value entered"]] +
                      [[str(i), Paragraph(s["text"], body), s.get("value") or ""]
                       for i, s in enumerate(r["path"], 1)],
                      [14 * mm, 120 * mm, 40 * mm]))

    story.append(Paragraph("Recommended next steps", h2))
    for rec in r["recommendations"]:
        story.append(Paragraph(f"<b>{rec['title']}.</b> {rec['advice']}", body))
        story.append(Spacer(1, 4))
    story.append(Paragraph(
        "Recommendations are fixed rules applied to the values entered. They are not "
        "produced by the Decision Tree.", small))

    story.append(Spacer(1, 10))
    story.append(Paragraph(
        "This report is a decision-support result. It is a prediction from patterns in "
        "training data, not a guarantee, and it does not replace a lecturer's assessment "
        "or the institution's grading procedure.", small))
    doc.build(story)
    return buffer.getvalue()
