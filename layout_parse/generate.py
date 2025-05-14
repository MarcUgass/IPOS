import json
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from reportlab.lib.colors import Color
from reportlab.platypus import Paragraph, Frame
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
import os
import fitz  # PyMuPDF
from PIL import Image

PAGE_WIDTH, PAGE_HEIGHT = A4


def flip_y(y):
    return PAGE_HEIGHT - y


def draw_text(c, obj):
    x1, y1, x2, y2 = obj["position"]
    y1, y2 = flip_y(y1), flip_y(y2)
    if y1 < y2:
        y1, y2 = y2, y1  # Ensure y1 is top, y2 is bottom

    width = x2 - x1
    height = y1 - y2

    text = obj.get("text", "")
    font = obj.get("font", "Helvetica")
    size = obj.get("size", 12)
    color = obj.get("color", [0, 0, 0])
    align = obj.get("align", "left")
    valign = obj.get("valign", "top")

    # Convert align to reportlab constants
    align_map = {"left": TA_LEFT, "center": TA_CENTER, "right": TA_RIGHT}
    text_align = align_map.get(align, TA_LEFT)

    # Create a style for the paragraph
    style = ParagraphStyle(
        name="CustomStyle",
        fontName=font,
        fontSize=size,
        leading=size * 1.2,
        textColor=Color(color[0], color[1], color[2]),
        alignment=text_align,
    )

    paragraph = Paragraph(text, style)

    # Create a frame to contain the paragraph
    frame = Frame(x1, y2, width, height, showBoundary=0)

    # Use a list for flowables
    story = [paragraph]

    # Draw on canvas
    frame.addFromList(story, c)


def draw_image(c, obj):
    x1, y1, x2, y2 = obj["position"]
    y1, y2 = flip_y(y1), flip_y(y2)

    width = x2 - x1
    height = y2 - y1
    path = obj["src"]

    if os.path.exists(path):
        image = ImageReader(path)
        c.drawImage(image, x1, y1, width=width, height=height)
    else:
        print(f"Image file not found: {path}")


def generate_pdf(json_file):
    with open(json_file, "r", encoding='utf-8') as f:
        data = json.load(f)

    c = canvas.Canvas("document.pdf", pagesize=A4)

    for key in sorted(data, key=int):
        obj = data[key]
        obj_type = obj.get("type")

        if obj_type == "text":
            draw_text(c, obj)
        elif obj_type == "image":
            draw_image(c, obj)

    c.save()

    # Open the PDF document
    doc = fitz.open("document.pdf")
    # Assuming it's a single page, get the first page (index 0)
    page = doc[0]

    # Render the page to a Pixmap. Higher DPI means higher resolution JPG.
    pix = page.get_pixmap(dpi=150)

    # Create a PIL Image from the Pixmap
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

    # Save the image as JPG
    img.save("image.jpg")

    # Close the document
    doc.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate PDF from JSON.")
    parser.add_argument("json_file", help="Path to the JSON input file")

    args = parser.parse_args()

    generate_pdf(args.json_file)
