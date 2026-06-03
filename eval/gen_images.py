"""Generate eval images and the ocr/vision datasets that reference them.

Run: python eval/gen_images.py   (from the project root)
Deterministic: same images and ground truth every run.
"""

import json
import os

from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(__file__)
IMG_DIR = os.path.join(HERE, "images")
DATA_DIR = os.path.join(HERE, "datasets")


def _font(size: int):
    for p in [
        "/System/Library/Fonts/Menlo.ttc",
        "/System/Library/Fonts/Monaco.ttf",
        "/Library/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    ]:
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            continue
    return ImageFont.load_default()


def make_text_image(path: str, lines: list[str], size: int = 20, pad: int = 20):
    font = _font(size)
    line_h = size + 8
    w = 680
    h = line_h * len(lines) + 2 * pad
    img = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(img)
    y = pad
    for ln in lines:
        d.text((pad, y), ln, fill="black", font=font)
        y += line_h
    img.save(path)


def make_shapes_image(path: str, shapes: list[dict]):
    """shapes: list of {kind: circle|square|triangle, color, x, y, r}."""
    img = Image.new("RGB", (480, 320), "white")
    d = ImageDraw.Draw(img)
    for s in shapes:
        x, y, r = s["x"], s["y"], s["r"]
        if s["kind"] == "circle":
            d.ellipse([x - r, y - r, x + r, y + r], fill=s["color"])
        elif s["kind"] == "square":
            d.rectangle([x - r, y - r, x + r, y + r], fill=s["color"])
        elif s["kind"] == "triangle":
            d.polygon([(x, y - r), (x - r, y + r), (x + r, y + r)], fill=s["color"])
    img.save(path)


def build():
    os.makedirs(IMG_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

    # ---- OCR dataset ----
    invoice = [
        "INVOICE  No: INV-2026-00731",
        "Date: 2026-06-03    Terms: Net 30",
        "Bill To: Acme Corp., 1200 O'Brien St., Ste. 4B",
        "Item             Qty   Unit($)   Total($)",
        "Widget-X1         12    9.95     119.40",
        "Bolt (M8x1.25)   144    0.08      11.52",
        "Gasket O-ring      6    3.10      18.60",
        "Subtotal                        149.52",
        "Tax (8.25%)                      12.34",
        "TOTAL                           161.86",
        "Note: lot #l00I-O0 shipped via FedEx.",
    ]
    paragraph = [
        "The quick brown fox jumps over the lazy dog.",
        "Pack my box with five dozen liquor jugs.",
        "Order #A1B2-C3 ships on 07/14 at 09:45.",
        "Contact: jane.doe@example.com  (555) 010-2398",
        "Balance due: $1,024.75 by EOD Friday.",
    ]
    make_text_image(os.path.join(IMG_DIR, "ocr_invoice.png"), invoice, size=20)
    make_text_image(os.path.join(IMG_DIR, "ocr_paragraph.png"), paragraph, size=22)

    ocr_ds = [
        {
            "image": "images/ocr_invoice.png",
            "truth": "\n".join(invoice),
            "key_tokens": ["INV-2026-00731", "O'Brien", "M8x1.25", "8.25%",
                           "l00I-O0", "119.40", "11.52", "18.60", "149.52",
                           "12.34", "161.86"],
        },
        {
            "image": "images/ocr_paragraph.png",
            "truth": "\n".join(paragraph),
            "key_tokens": ["#A1B2-C3", "07/14", "09:45", "jane.doe@example.com",
                           "(555) 010-2398", "$1,024.75"],
        },
    ]
    json.dump(ocr_ds, open(os.path.join(DATA_DIR, "ocr.json"), "w"), indent=2)

    # ---- Vision dataset (countable / identifiable shapes) ----
    s1 = (
        [{"kind": "circle", "color": "red", "x": 60 + i * 90, "y": 80, "r": 30} for i in range(3)]
        + [{"kind": "square", "color": "blue", "x": 90 + i * 110, "y": 220, "r": 28} for i in range(2)]
    )
    s2 = (
        [{"kind": "triangle", "color": "green", "x": 130, "y": 160, "r": 70}]
        + [{"kind": "circle", "color": "purple", "x": 330 + i * 70, "y": 160, "r": 32} for i in range(2)]
    )
    make_shapes_image(os.path.join(IMG_DIR, "vision_shapes1.png"), s1)
    make_shapes_image(os.path.join(IMG_DIR, "vision_shapes2.png"), s2)

    vision_ds = [
        {"image": "images/vision_shapes1.png",
         "question": "How many red circles are in the image? Answer with just the number.",
         "accept": ["3", "three"]},
        {"image": "images/vision_shapes1.png",
         "question": "How many blue squares are in the image? Answer with just the number.",
         "accept": ["2", "two"]},
        {"image": "images/vision_shapes2.png",
         "question": "What color is the triangle? Answer with one word.",
         "accept": ["green"]},
        {"image": "images/vision_shapes2.png",
         "question": "How many purple circles are there? Answer with just the number.",
         "accept": ["2", "two"]},
    ]
    json.dump(vision_ds, open(os.path.join(DATA_DIR, "vision.json"), "w"), indent=2)

    print("Generated images in", IMG_DIR)
    print("Wrote datasets: ocr.json, vision.json")


if __name__ == "__main__":
    build()
