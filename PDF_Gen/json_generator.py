import argparse
import random
import json
import math
import os

# Canvas dimensions
PAGE_WIDTH = 595
PAGE_HEIGHT = 860

# Rectangle settings
MIN_RECT_SIZE = 50
MAX_RECT_SIZE = 250
MAX_ATTEMPTS = 1000
NUM_RECTS = random.randint(5, 12)

# Font settings
FONTS = ["Helvetica", "Times-Roman"]
FONT_SIZE = {
    "Helvetica": 12,
    "Times-Roman": 12
}
FONT_WIDTH_ESTIMATES = {
    "Helvetica": 0.8,
    "Times-Roman": 0.9
}

# Load Don Quijote text
with open("el_quijote2.txt", "r", encoding="utf-8") as f:
    DONQUIJOTE_TEXT = f.read().replace("\n", " ")

# === Utility Functions ===

def get_random_fragment(min_length=100, max_length=300):
    start = random.randint(0, len(DONQUIJOTE_TEXT) - max_length - 1)
    end = start + random.randint(min_length, max_length)
    fragment = DONQUIJOTE_TEXT[start:end]
    if "." in fragment:
        fragment = fragment.rsplit(".", 1)[0] + "."
    return fragment.strip()

def calculate_text_width(text, font, font_size):
    return len(text) * FONT_WIDTH_ESTIMATES[font] * font_size

def is_overlapping(r1, r2):
    return not (r1[2] <= r2[0] or r1[0] >= r2[2] or r1[3] <= r2[1] or r1[1] >= r2[3])

def does_overlap_any(new_rect, existing_rects):
    return any(is_overlapping(new_rect, obj["position"]) for obj in existing_rects.values())

def generate_text_block(block_id, existing_rects):
    max_tries = 50
    for _ in range(max_tries):
        fragment = get_random_fragment()
        font = random.choice(FONTS)
        font_size = FONT_SIZE[font]
        text_width = calculate_text_width(fragment, font, font_size)

        x1 = random.randint(10, PAGE_WIDTH - 250)
        y1 = random.randint(10, PAGE_HEIGHT - 60)
        width = min(text_width + 20, PAGE_WIDTH - x1 - 10)
        max_chars_per_line = width // (FONT_WIDTH_ESTIMATES[font] * font_size)
        num_lines = math.ceil(len(fragment) / max_chars_per_line)
        height = num_lines * font_size + 10
        x2 = x1 + width
        y2 = min(y1 + height, PAGE_HEIGHT - 10)

        position = [x1, y1, x2, y2]
        if not does_overlap_any(position, existing_rects):
            return {
                str(block_id): {
                    "type": "text",
                    "text": fragment,
                    "font": font,
                    "size": font_size,
                    "color": [0, 0, 0],
                    "position": position,
                    "align": "left",
                    "valign": "top"
                }
            }
    return None  # Couldn't place without overlap

# === Main Process ===
def main():
        
    parser = argparse.ArgumentParser(description="Generate a document layout JSON with specified image/text ratio.")
    parser.add_argument("output_file", type=str, help="Name of the output JSON file.")

    parser.add_argument("--ratio", type=float, help="Ratio of image blocks to total blocks (0.0 to 1.0).")
    parser.add_argument("--total_blocks", type=int, default=10, help="Total number of blocks to generate (default: 10).")

    args = parser.parse_args()

    image_ratio = args.ratio
    output_filename = args.output_file
    total_blocks = args.total_blocks
    rectangles = {}
    attempts = 0
    index = 1

    # STEP 1: Place image blocks first (30%)
    num_images = int(total_blocks * image_ratio)

    while len(rectangles) < num_images and attempts < MAX_ATTEMPTS:
        w = random.randint(MIN_RECT_SIZE, MAX_RECT_SIZE)
        h = random.randint(MIN_RECT_SIZE, MAX_RECT_SIZE)
        x1 = random.randint(0, PAGE_WIDTH - w)
        y1 = random.randint(0, PAGE_HEIGHT - h)
        x2 = x1 + w
        y2 = y1 + h
        pos = [x1, y1, x2, y2]

        if not does_overlap_any(pos, rectangles):
            rectangles[str(index)] = {
                "type": "image",
                "src": f"imagesGEN/{random.randint(1, 127)}.jpg",
                "position": pos
            }
            index += 1
        attempts += 1

    # STEP 2: Place text blocks (remaining)
    while len(rectangles) < total_blocks and attempts < MAX_ATTEMPTS:
        block = generate_text_block(index, rectangles)
        if block:
            rectangles.update(block)
            index += 1
        attempts += 1

    # === Save Final JSON ===
    with open(output_filename+"/layout.json", "w", encoding="utf-8") as f:
        json.dump(rectangles, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    main()