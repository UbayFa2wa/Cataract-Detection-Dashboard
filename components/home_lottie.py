import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
ANIM_DIR = BASE_DIR / "assets" / "animations"


def load_lottiefile(filename: str):
    file_path = ANIM_DIR / filename
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


lottie_doctor = load_lottiefile("doctor.json")
lottie_data_analytics = load_lottiefile("data_analytics.json")
lottie_thinking = load_lottiefile("thinking.json")
