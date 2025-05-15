import json
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from reportlab.lib.colors import Color
from reportlab.platypus import Paragraph, Frame
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
import os

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

def change_coordinates(obj, scale_height, scale_width):
    x1, y1, x2, y2 = obj["position"]
    obj["position"] = [int(x1 * scale_width), int(y1 * scale_height), int(x2 * scale_width), int(y2 * scale_height)]
    return obj


def scale(image_height, image_width):
    scale_height, scale_width = PAGE_HEIGHT/image_height, PAGE_WIDTH/image_width
    return scale_height, scale_width

def generate_pdf(json_file, output_pdf_path, image_height, image_width):
    with open(json_file, "r", encoding='utf-8') as f:
        data = json.load(f)

    c = canvas.Canvas(output_pdf_path, pagesize=A4)

    scale_height, scale_width = scale(image_height, image_width)

    for key in sorted(data, key=int):
        obj = data[key]
        obj_type = obj.get("type")
        obj = change_coordinates(obj, scale_height, scale_width)
        if obj_type == "text":
            draw_text(c, obj)
        elif obj_type == "image":
            draw_image(c, obj)

    c.save()
    print("saved")


"""
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate PDF from JSON.")
    parser.add_argument("json_file", help="Path to the JSON input file")

    args = parser.parse_args()

    generate_pdf(args.json_file)
    """
