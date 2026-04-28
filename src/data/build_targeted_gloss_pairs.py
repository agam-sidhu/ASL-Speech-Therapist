"""Build targeted corrective gloss pairs from observed evaluation weaknesses."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"

MASTER_DATASET_PATH = DATA_DIR / "active" / "project_finetune_v2_v4_contrastive.json"
TARGETED_OUTPUT_PATH = DATA_DIR / "reports" / "data_pipeline_report.json"
MASTER_PLUS_TARGETED_PATH = DATA_DIR / "active" / "project_finetune_v2_v4_contrastive.json"
REPORT_OUTPUT_PATH = DATA_DIR / "reports" / "data_pipeline_report.json"

WORD_PATTERN = re.compile(r"[A-Za-z']+")

TARGETED_PAIRS: dict[str, list[tuple[str, str]]] = {
    "negation": [
        ("i cannot hear the teacher", "I HEAR TEACHER CAN NOT"),
        ("i did not finish my homework", "I NOT FINISH MY HOMEWORK"),
        ("she is not home today", "TODAY SHE HOME NOT"),
        ("we do not understand the homework", "WE NOT UNDERSTAND HOMEWORK"),
        ("he cannot come tomorrow", "TOMORROW HE COME CAN NOT"),
        ("my phone is not working", "MY PHONE WORK NOT"),
        ("i am not ready for class", "CLASS I READY NOT"),
        ("they did not bring the book", "THEY BRING BOOK NOT"),
        ("i never visited that museum", "I VISIT THAT MUSEUM NEVER"),
        ("the train is not here yet", "TRAIN HERE NOT YET"),
        ("i do not like this soup", "THIS SOUP I LIKE NOT"),
        ("she cannot find her keys", "HER KEY SHE FIND CAN NOT"),
        ("we are not late today", "TODAY WE LATE NOT"),
        ("i do not remember your name", "YOUR NAME I REMEMBER NOT"),
        ("he never studies at night", "NIGHT HE STUDY NEVER"),
        ("my computer is not charged", "MY COMPUTER CHARGE NOT"),
        ("i cannot stay long", "I STAY LONG CAN NOT"),
        ("they are not finished yet", "THEY FINISH NOT YET"),
        ("i do not want that one", "THAT ONE I WANT NOT"),
        ("the library is not open now", "LIBRARY OPEN NOT NOW"),
    ],
    "yes_no_question": [
        ("do you have time now", "YOU HAVE TIME NOW YOU"),
        ("are you free tomorrow", "YOU FREE TOMORROW YOU"),
        ("can we start class", "WE START CLASS CAN"),
        ("did she call you", "SHE CALL YOU SHE"),
        ("are they ready already", "THEY READY ALREADY THEY"),
        ("can you help me later", "LATER YOU HELP ME CAN"),
        ("do you know my teacher", "YOU KNOW MY TEACHER YOU"),
        ("is he home now", "HE HOME NOW HE"),
        ("did they finish the project", "THEY FINISH PROJECT THEY"),
        ("can i leave early", "I LEAVE EARLY CAN"),
        ("do you need water", "YOU NEED WATER YOU"),
        ("are we meeting today", "TODAY WE MEET WE"),
        ("can he drive tonight", "HE DRIVE TONIGHT CAN"),
        ("do you want more coffee", "YOU WANT MORE COFFEE YOU"),
        ("is she your doctor", "SHE YOUR DOCTOR SHE"),
        ("did he bring the tickets", "HE BRING TICKET HE"),
        ("can they stay after class", "AFTER CLASS THEY STAY CAN"),
        ("do you hear the music", "YOU HEAR MUSIC YOU"),
        ("are you busy this weekend", "THIS WEEKEND YOU BUSY YOU"),
        ("can we talk after dinner", "AFTER DINNER WE TALK CAN"),
    ],
    "time_fronting": [
        ("today the weather is nice", "TODAY WEATHER NICE"),
        ("today i learned a new sign", "TODAY I LEARN NEW SIGN"),
        ("tomorrow my mother visits me", "TOMORROW MY MOTHER VISIT ME"),
        ("yesterday we finished the lesson", "YESTERDAY WE FINISH LESSON"),
        ("tonight i study at home", "TONIGHT I STUDY HOME"),
        ("this morning the train was late", "THIS-MORNING TRAIN LATE"),
        ("after class i meet my tutor", "AFTER CLASS I MEET MY TUTOR"),
        ("next week my doctor appointment is monday", "NEXT-WEEK MY DOCTOR APPOINTMENT MONDAY"),
        ("later we practice together", "LATER WE PRACTICE TOGETHER"),
        ("every friday i work downtown", "EVERY-FRIDAY I WORK DOWNTOWN"),
        ("this afternoon she has a meeting", "THIS-AFTERNOON SHE HAVE MEETING"),
        ("tomorrow we review chapter five", "TOMORROW WE REVIEW CHAPTER FIVE"),
        ("last night my computer crashed", "LAST-NIGHT MY COMPUTER CRASH"),
        ("today my father cooks dinner", "TODAY MY FATHER COOK DINNER"),
        ("next month we move apartments", "NEXT-MONTH WE MOVE APARTMENT"),
    ],
    "family_profession_topic_comment": [
        ("my mother works at the hospital", "MY MOTHER HOSPITAL WORK"),
        ("my father teaches math", "MY FATHER TEACH MATH"),
        ("my sister is my interpreter", "MY SISTER MY INTERPRETER"),
        ("my brother is a nurse", "MY BROTHER NURSE"),
        ("our teacher is deaf", "OUR TEACHER DEAF"),
        ("my aunt is a lawyer", "MY AUNT LAWYER"),
        ("this man is my cousin", "THIS MAN MY COUSIN"),
        ("that woman is my boss", "THAT WOMAN MY BOSS"),
        ("my grandfather is a mechanic", "MY GRANDFATHER MECHANIC"),
        ("my roommate is hearing", "MY ROOMMATE HEARING"),
    ],
    "descriptive_reorder": [
        ("the red book is on the table", "TABLE ON RED BOOK"),
        ("my blue backpack is in the car", "CAR IN MY BLUE BACKPACK"),
        ("the meeting is in room five", "ROOM FIVE MEETING"),
        ("your coffee is on the desk", "DESK ON YOUR COFFEE"),
        ("the new student is from mexico", "NEW STUDENT MEXICO FROM"),
        ("the homework for math is difficult", "MATH HOMEWORK DIFFICULT"),
        ("the bus stop is near the library", "LIBRARY NEAR BUS STOP"),
        ("my favorite class is science", "MY FAVORITE CLASS SCIENCE"),
        ("the yellow jacket is mine", "YELLOW JACKET MINE"),
        ("the doctor office is downstairs", "DOCTOR OFFICE DOWNSTAIRS"),
    ],
}


def normalize_english(text: str) -> str:
    return " ".join(token.lower() for token in WORD_PATTERN.findall((text or "").lower()))


def normalize_gloss(text: str) -> str:
    return " ".join(str(text).strip().upper().split())


def load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def build_targeted_dataset() -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    master_entries = load_json(MASTER_DATASET_PATH)
    if not isinstance(master_entries, list):
        raise ValueError("Master dataset must be a JSON list.")

    existing_pairs = {(normalize_english(item["english"]), normalize_gloss(item["gloss"])) for item in master_entries}
    existing_english = {normalize_english(item["english"]) for item in master_entries}

    targeted_entries: list[dict[str, object]] = []
    seen_pairs: set[tuple[str, str]] = set()
    seen_english: set[str] = set()
    for category, pairs in TARGETED_PAIRS.items():
        expected = 20 if category in {"negation", "yes_no_question"} else 15 if category == "time_fronting" else 10
        if len(pairs) != expected:
            raise ValueError(f"Category {category} expected {expected} pairs, found {len(pairs)}.")

        for english_text, gloss_text in pairs:
            english = normalize_english(english_text)
            gloss = normalize_gloss(gloss_text)
            pair = (english, gloss)

            if english in existing_english:
                raise ValueError(f"Targeted English prompt already exists in master dataset: {english}")
            if pair in existing_pairs:
                raise ValueError(f"Targeted pair already exists in master dataset: {pair}")
            if english in seen_english:
                raise ValueError(f"Targeted English prompt duplicated internally: {english}")
            if pair in seen_pairs:
                raise ValueError(f"Targeted pair duplicated internally: {pair}")

            targeted_entries.append(
                {
                    "pair_id": f"targeted_{len(targeted_entries) + 1:04d}",
                    "english": english,
                    "gloss": gloss,
                    "source_kind": "targeted_failure_augmentation",
                    "source_files": [str(TARGETED_OUTPUT_PATH.relative_to(PROJECT_ROOT))],
                    "manual_category": category,
                    "notes": "Synthetic targeted augmentation created from observed grammar-challenge weaknesses.",
                }
            )
            seen_english.add(english)
            seen_pairs.add(pair)

    if len(targeted_entries) != 75:
        raise ValueError(f"Expected 75 targeted entries, found {len(targeted_entries)}.")

    master_plus_targeted: list[dict[str, object]] = []
    for index, entry in enumerate(master_entries + targeted_entries, start=1):
        merged = dict(entry)
        merged["pair_id"] = f"master_plus_{index:04d}"
        master_plus_targeted.append(merged)

    report = {
        "master_dataset": str(MASTER_DATASET_PATH.relative_to(PROJECT_ROOT)),
        "targeted_pair_count": len(targeted_entries),
        "targeted_category_counts": dict(Counter(entry["manual_category"] for entry in targeted_entries)),
        "master_plus_targeted_count": len(master_plus_targeted),
        "output_files": {
            "targeted": str(TARGETED_OUTPUT_PATH.relative_to(PROJECT_ROOT)),
            "master_plus_targeted": str(MASTER_PLUS_TARGETED_PATH.relative_to(PROJECT_ROOT)),
        },
    }
    return targeted_entries, master_plus_targeted, report


def main() -> None:
    print(
        json.dumps(
            {
                "status": "disabled",
                "reason": "Targeted pairs are archived for now and are not part of the active demo pipeline.",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
