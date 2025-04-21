import json
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from reportlab.lib.colors import Color
import os


def draw_text(c, obj):
    x1, y1, x2, y2 = obj["position"]
    width = x2 - x1
    height = y2 - y1

    text = obj.get("text", "")
    font = obj.get("font", "Helvetica")
    size = obj.get("size", 12)
    color = obj.get("color", [0, 0, 0])
    align = obj.get("align", "left")
    valign = obj.get("valign", "top")

    c.setFont(font, size)
    c.setFillColor(Color(color[0], color[1], color[2]))

    # Adjust text x position based on alignment
    if align == "center":
        text_width = c.stringWidth(text, font, size)
        tx = x1 + (width - text_width) / 2
    elif align == "right":
        text_width = c.stringWidth(text, font, size)
        tx = x2 - text_width
    else:  # left
        tx = x1

    # Adjust text y position based on vertical alignment
    if valign == "middle":
        ty = y1 + (height - size) / 2
    elif valign == "bottom":
        ty = y2 - size
    else:  # top
        ty = y1 + height - size

    c.drawString(tx, ty, text)


def draw_image(c, obj):
    x1, y1, x2, y2 = obj["position"]
    width = x2 - x1
    height = y2 - y1
    path = obj["src"]

    if os.path.exists(path):
        image = ImageReader(path)
        c.drawImage(image, x1, y1, width=width, height=height)
    else:
        print(f"Image file not found: {path}")


def generate_pdf(json_file, output_file="output.pdf"):
    with open(json_file, "r", encoding='utf-8') as f:
        data = json.load(f)

    c = canvas.Canvas(output_file, pagesize=A4)

    for key in sorted(data, key=int):
        obj = data[key]
        obj_type = obj.get("type")

        if obj_type == "text":
            draw_text(c, obj)
        elif obj_type == "image":
            draw_image(c, obj)

    c.save()
    print(f"PDF saved as {output_file}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate PDF from JSON.")
    parser.add_argument("json_file", help="Path to the JSON input file")
    parser.add_argument("--output", default="output.pdf", help="Path to the output PDF file")
    args = parser.parse_args()

    generate_pdf(args.json_file, args.output)
