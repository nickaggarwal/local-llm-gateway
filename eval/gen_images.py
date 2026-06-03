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
    code_snippet = [
        "def binary_search(arr, x):",
        "    lo, hi = 0, len(arr) - 1",
        "    while lo <= hi:",
        "        mid = (lo + hi) // 2",
        "        if arr[mid] == x: return mid",
        "        elif arr[mid] < x: lo = mid + 1",
        "        else: hi = mid - 1",
        "    return -1   # not found",
    ]
    numeric_table = [
        "ID     Date        Amount    Rate%",
        "0012   2026-01-05    1250.00   3.25",
        "0034   2026-02-18     980.49   3.10",
        "0156   2026-03-30   12045.75   2.95",
        "1099   2026-11-15      -42.00   0.00",
        "SUM                 14234.24",
    ]
    receipt = [
        "QuickMart #4471",
        "2x Coffee @ 3.49 ......  6.98",
        "1x Bagel  @ 2.25 ......  2.25",
        "SUBTOTAL ............... 9.23",
        "VAT 7% ................. 0.65",
        "TOTAL .................. 9.88",
        "CARD **** 4012   AUTH 00731",
    ]
    address = [
        "SHIP TO:",
        "Dr. Jane Q. McAllister-Ng",
        "Apt 12-C, 4587 N. 1st Ave.",
        "St. Paul, MN 55101-2293",
        "Tracking: 1Z-999-AA1-01-2345-6784",
    ]
    for name, lines, sz in [
        ("ocr_invoice", invoice, 20), ("ocr_paragraph", paragraph, 22),
        ("ocr_code", code_snippet, 20), ("ocr_table", numeric_table, 20),
        ("ocr_receipt", receipt, 22), ("ocr_address", address, 22),
    ]:
        make_text_image(os.path.join(IMG_DIR, f"{name}.png"), lines, size=sz)

    ocr_ds = [
        {"image": "images/ocr_invoice.png", "truth": "\n".join(invoice),
         "key_tokens": ["INV-2026-00731", "O'Brien", "M8x1.25", "8.25%",
                        "l00I-O0", "119.40", "11.52", "18.60", "149.52",
                        "12.34", "161.86"]},
        {"image": "images/ocr_paragraph.png", "truth": "\n".join(paragraph),
         "key_tokens": ["#A1B2-C3", "07/14", "09:45", "jane.doe@example.com",
                        "(555) 010-2398", "$1,024.75"]},
        {"image": "images/ocr_code.png", "truth": "\n".join(code_snippet),
         "key_tokens": ["binary_search(arr, x)", "len(arr) - 1", "(lo + hi) // 2",
                        "lo = mid + 1", "hi = mid - 1", "return -1"]},
        {"image": "images/ocr_table.png", "truth": "\n".join(numeric_table),
         "key_tokens": ["0012", "1250.00", "3.25", "12045.75", "-42.00",
                        "14234.24", "0156"]},
        {"image": "images/ocr_receipt.png", "truth": "\n".join(receipt),
         "key_tokens": ["QuickMart #4471", "6.98", "2.25", "9.23", "0.65",
                        "9.88", "4012", "00731"]},
        {"image": "images/ocr_address.png", "truth": "\n".join(address),
         "key_tokens": ["McAllister-Ng", "Apt 12-C", "4587 N. 1st Ave.",
                        "55101-2293", "1Z-999-AA1-01-2345-6784"]},
    ]
    json.dump(ocr_ds, open(os.path.join(DATA_DIR, "ocr.json"), "w"), indent=2)

    print("Generated OCR images in", IMG_DIR)
    print("Wrote dataset: ocr.json")
    print("(vision.json comes from a public VQA set — run: python eval/fetch_vqa.py)")


if __name__ == "__main__":
    build()
