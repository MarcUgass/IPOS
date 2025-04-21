import random
import json

# Canvas size
WIDTH = 595
HEIGHT = 860

# Configuration
MIN_RECT_SIZE = 50
MAX_RECT_SIZE = 250
MAX_ATTEMPTS = 1000

# Random number of rectangles
num_rects = random.randint(5, 12)
rectangles = {}

def is_overlapping(r1, r2):
    return not (r1[2] <= r2[0] or r1[0] >= r2[2] or r1[3] <= r2[1] or r1[1] >= r2[3])

def does_overlap_any(new_rect, existing_rects):
    return any(is_overlapping(new_rect, rect["position"]) for rect in existing_rects.values())

# Try to generate non-overlapping rectangles
attempts = 0
index = 1

while len(rectangles) < num_rects and attempts < MAX_ATTEMPTS:
    w = random.randint(MIN_RECT_SIZE, MAX_RECT_SIZE)
    h = random.randint(MIN_RECT_SIZE, MAX_RECT_SIZE)
    x1 = random.randint(0, WIDTH - w)
    y1 = random.randint(0, HEIGHT - h)
    x2 = x1 + w
    y2 = y1 + h
    new_position = [x1, y1, x2, y2]

    if not does_overlap_any(new_position, rectangles):
        rectangles[str(index)] = {
            "type": "image",
            "src": f"imagesGEN/{index}.jpg",
            "position": new_position
        }
        index += 1

    attempts += 1

# Save to JSON
with open('rectangles.json', 'w') as f:
    json.dump(rectangles, f, indent=4)

print(f"Saved {len(rectangles)} non-overlapping rectangles to rectangles.json")
