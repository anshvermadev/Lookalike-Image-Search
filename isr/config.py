from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "caltech101-small"
INDEXES = ROOT / "indexes"
ARTIFACTS = ROOT / "artifacts"
MODEL_DIR = ROOT / "models" / "clip-vit-base-patch32"
MODEL_PATH = MODEL_DIR / "vision_model_quantized.onnx"
MODEL_REPO = "Xenova/clip-vit-base-patch32"
METHODS = {"clip": "CLIP semantic search", "histogram": "Color histogram"}
FEATURE_VERSIONS = {"histogram": "hsv-8x8x8-v1", "clip": "clip-vit-b32-onnx-int8-v1"}
LABELS = {"airplanes": "Airplanes", "butterfly": "Butterflies", "car_side": "Cars",
          "cellphone": "Phones", "chair": "Chairs", "cup": "Cups", "elephant": "Elephants",
          "laptop": "Laptops", "Motorbikes": "Motorbikes", "watch": "Watches"}


def label(category: str) -> str:
    return LABELS.get(category, category.replace("_", " ").title())
