import os
import json
import random
from PIL import Image, ImageDraw, ImageFont

# -----------------------------
# YOUR MAIN PROJECT PATH
# -----------------------------
BASE_PATH = os.path.dirname(os.path.abspath(__file__))

IMAGE_FOLDER = os.path.join(BASE_PATH, "invoice_images")
GT_FOLDER = os.path.join(BASE_PATH, "ground_truth")

os.makedirs(IMAGE_FOLDER, exist_ok=True)
os.makedirs(GT_FOLDER, exist_ok=True)

# -----------------------------
# DATA
# -----------------------------
companies = ["TechNova Ltd.", "Alpha Solutions", "GreenEnergy Corp", "NextGen Systems"]

dates = [
    "April 19, 2024", "May 2, 2024",
    "June 11, 2024", "July 5, 2024",
    "August 22, 2024"
]

# -----------------------------
# GENERATE DATA
# -----------------------------
def generate_invoice_data(index):

    invoice_number = f"INV-{random.randint(1000,9999)}-{index}"
    date = random.choice(dates)
    total_value = random.randint(100, 5000) * 10
    total = f"${total_value:,}"

    # 60% simple, 40% complex 
    complexity = "simple" if random.random() < 0.6 else "complex"

    return {
        "invoice_number": invoice_number,
        "date": date,
        "total": total,
        "complexity": complexity
    }

# -----------------------------
# IMAGE CREATION
# -----------------------------
def create_invoice_image(data, path):

    height = 600 if data["complexity"] == "simple" else 1000

    image = Image.new("RGB", (800, height), "white")
    draw = ImageDraw.Draw(image)

    try:
        font = ImageFont.truetype("arial.ttf", 28)
    except:
        font = ImageFont.load_default()

    y = 50

    draw.text((50, y), random.choice(companies), fill="black", font=font)
    y += 80

    draw.text((50, y), f"Invoice Number: {data['invoice_number']}", fill="black", font=font)
    y += 60

    draw.text((50, y), f"Date: {data['date']}", fill="black", font=font)
    y += 60

    # COMPLEX invoices
    if data["complexity"] == "complex":
        for i in range(random.randint(3, 8)):
            item = f"Item {i+1} - ${random.randint(10,500)}"
            draw.text((50, y), item, fill="black", font=font)
            y += 40

        noise = ["VAT Included", "Payment Terms: Net 30", "Thank you for your business"]
        for n in noise:
            draw.text((50, y), n, fill="gray", font=font)
            y += 40

        y += 40

    draw.text((50, y), f"Total: {data['total']}", fill="black", font=font)

    image.save(path)

# -----------------------------
# SAVE JSON
# -----------------------------
def save_json(data, path):

    gt = {
        "invoice_number": data["invoice_number"],
        "date": data["date"],
        "total": data["total"]
    }

    with open(path, "w") as f:
        json.dump(gt, f, indent=4)

# -----------------------------
# FINAL DATASET GENERATION
# -----------------------------
def generate_dataset(start=16, end=315):

    print("\nGenerating FINAL dataset...\n")

    for i in range(start, end + 1):

        img_name = f"test_invoice{i}.png"
        json_name = f"test_invoice{i}.json"

        img_path = os.path.join(IMAGE_FOLDER, img_name)
        json_path = os.path.join(GT_FOLDER, json_name)

        # SAFETY: fot not  overwrite existing files
        if os.path.exists(img_path) or os.path.exists(json_path):
            print(f"Skipping existing: {img_name}")
            continue

        data = generate_invoice_data(i)

        create_invoice_image(data, img_path)
        save_json(data, json_path)

        print(f"Created: {img_name} ({data['complexity']})")

    print("\nDONE — FINAL dataset ready!")

# -----------------------------
if __name__ == "__main__":
    generate_dataset()