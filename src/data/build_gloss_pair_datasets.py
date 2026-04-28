"""Build merged and augmented English->ASL gloss datasets."""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"

ORIGINAL_DATASET_FILES = [
    DATA_DIR / "asl_gloss_conversational.json",
]

GENERATED_OUTPUT_PATH = DATA_DIR / "active" / "project_finetune_v2_v4_contrastive.json"
MERGED_OUTPUT_PATH = DATA_DIR / "active" / "project_finetune_v2_v4_contrastive.json"
CONFLICT_OUTPUT_PATH = DATA_DIR / "reports" / "data_pipeline_report.json"
REPORT_OUTPUT_PATH = DATA_DIR / "reports" / "data_pipeline_report.json"

WORD_PATTERN = re.compile(r"[A-Za-z']+")


GENERATED_PAIRS: dict[str, list[tuple[str, str]]] = {
    "wh_question": [
        ("what time does class start", "CLASS START TIME WHAT"),
        ("what time do you leave home", "HOME LEAVE YOU TIME WHAT"),
        ("what chapter are we reading", "WE READ CHAPTER WHAT"),
        ("what bus do you take", "YOU TAKE BUS WHAT"),
        ("what room is the meeting in", "MEETING ROOM WHAT"),
        ("what movie do you want to watch", "YOU WANT WATCH MOVIE WHAT"),
        ("what snack do you want", "YOU WANT SNACK WHAT"),
        ("what color is your car", "YOUR CAR COLOR WHAT"),
        ("what book are you reading", "YOU READ BOOK WHAT"),
        ("what song is playing", "PLAY SONG WHAT"),
        ("where did you park the car", "YOU PARK CAR WHERE"),
        ("where is my notebook", "MY NOTEBOOK WHERE"),
        ("where should we sit", "WE SIT WHERE"),
        ("where can i charge my phone", "MY PHONE CHARGE WHERE"),
        ("where are your glasses", "YOUR GLASSES WHERE"),
        ("where do they study after class", "AFTER CLASS THEY STUDY WHERE"),
        ("when does the train arrive", "TRAIN ARRIVE WHEN"),
        ("when is your appointment", "YOUR APPOINTMENT WHEN"),
        ("when do you usually eat lunch", "YOU USUALLY EAT LUNCH WHEN"),
        ("when will the store open", "STORE OPEN WHEN"),
        ("why is he laughing", "HE LAUGH WHY"),
        ("why did they cancel class", "THEY CANCEL CLASS WHY"),
        ("why are you in a hurry", "YOU HURRY WHY"),
        ("why should i wait outside", "I WAIT OUTSIDE WHY"),
        ("how are we getting there", "WE GO-TO THERE HOW"),
        ("how long is the movie", "MOVIE HOW-LONG"),
        ("how many students are absent", "STUDENT ABSENT HOW-MANY"),
        ("how much homework do we have tonight", "TONIGHT HOMEWORK HOW-MUCH"),
        ("how often do you practice at home", "YOU PRACTICE HOME HOW-OFTEN"),
        ("how far is the station", "STATION HOW-FAR"),
    ],
    "time_fronting": [
        ("today i need to finish homework", "TODAY I NEED FINISH HOMEWORK"),
        ("today the lab is closed", "TODAY LAB CLOSED"),
        ("today we practice new signs", "TODAY WE PRACTICE NEW SIGN"),
        ("today my bus was late", "TODAY MY BUS LATE"),
        ("today she works at home", "TODAY SHE WORK HOME"),
        ("tomorrow we meet the advisor", "TOMORROW WE MEET ADVISOR"),
        ("tomorrow i pack my bag", "TOMORROW I PACK MY BAG"),
        ("tomorrow the office opens early", "TOMORROW OFFICE OPEN EARLY"),
        ("tomorrow my sister visits me", "TOMORROW MY SISTER VISIT ME"),
        ("tomorrow we review the chapter", "TOMORROW WE REVIEW CHAPTER"),
        ("yesterday i missed the bus", "YESTERDAY I MISS BUS"),
        ("yesterday the teacher was absent", "YESTERDAY TEACHER ABSENT"),
        ("yesterday i cooked dinner", "YESTERDAY I COOK DINNER"),
        ("yesterday we stayed home", "YESTERDAY WE STAY HOME"),
        ("yesterday my phone died", "YESTERDAY MY PHONE DIE"),
        ("later i will text you", "LATER I TEXT YOU"),
        ("later we discuss the project", "LATER WE DISCUSS PROJECT"),
        ("later he drives to the store", "LATER HE DRIVE STORE"),
        ("later i send the notes", "LATER I SEND NOTE"),
        ("later they call the doctor", "LATER THEY CALL DOCTOR"),
        ("now i need water", "NOW I NEED WATER"),
        ("now class starts", "NOW CLASS START"),
        ("now the store is busy", "NOW STORE BUSY"),
        ("now we can leave", "NOW WE LEAVE CAN"),
        ("now my laptop works", "NOW MY LAPTOP WORK"),
        ("every day she practices piano", "EVERY-DAY SHE PRACTICE PIANO"),
        ("every day we eat together", "EVERY-DAY WE EAT TOGETHER"),
        ("every day my hands hurt", "EVERY-DAY MY HAND HURT"),
        ("morning i walk my dog", "MORNING I WALK MY DOG"),
        ("night we study together", "NIGHT WE STUDY TOGETHER"),
    ],
    "negation": [
        ("i do not remember the address", "I NOT REMEMBER ADDRESS"),
        ("she is not available today", "TODAY SHE AVAILABLE NOT"),
        ("we cannot stay long", "WE STAY LONG CAN NOT"),
        ("they did not bring tickets", "THEY BRING TICKET NOT"),
        ("i never eat meat", "I NEVER EAT MEAT"),
        ("the printer is not working", "PRINTER WORK NOT"),
        ("i am not ready yet", "I READY NOT YET"),
        ("he does not drive", "HE DRIVE NOT"),
        ("my laptop is not charged", "MY LAPTOP CHARGE NOT"),
        ("we do not have enough chairs", "WE HAVE CHAIR ENOUGH NOT"),
        ("i cannot find my keys", "I FIND MY KEY CAN NOT"),
        ("she did not call me", "SHE CALL ME NOT"),
        ("they are not finished", "THEY FINISH NOT"),
        ("i do not want coffee", "I NOT WANT COFFEE"),
        ("the bus is not here yet", "BUS HERE NOT YET"),
        ("i never saw that sign before", "I SEE THAT SIGN NEVER BEFORE"),
        ("the teacher is not in the office", "TEACHER OFFICE NOT"),
        ("i cannot open the door", "I OPEN DOOR CAN NOT"),
        ("the battery is not full", "BATTERY FULL NOT"),
        ("we are not late", "WE LATE NOT"),
        ("i do not trust that answer", "I NOT TRUST THAT ANSWER"),
        ("she cannot hear the music", "SHE HEAR MUSIC CAN NOT"),
        ("they never practice at home", "THEY HOME PRACTICE NEVER"),
        ("i am not hungry anymore", "I HUNGRY NOT ANYMORE"),
        ("the water is not cold", "WATER COLD NOT"),
        ("we do not understand the instructions", "WE NOT UNDERSTAND INSTRUCTION"),
        ("he is not my partner", "HE MY PARTNER NOT"),
        ("i cannot sleep tonight", "TONIGHT I SLEEP CAN NOT"),
        ("the store is not open now", "STORE OPEN NOT NOW"),
        ("i did not finish reading", "I FINISH READ NOT"),
    ],
    "yes_no_question": [
        ("are you ready now", "YOU READY NOW YOU"),
        ("did he call today", "HE CALL TODAY HE"),
        ("can we start the meeting", "WE START MEETING CAN"),
        ("do they need help", "THEY NEED HELP THEY"),
        ("are you busy tonight", "YOU BUSY TONIGHT YOU"),
        ("did you lock the door", "YOU LOCK DOOR YOU"),
        ("can she come tomorrow", "SHE COME TOMORROW CAN"),
        ("are they home already", "THEY HOME ALREADY THEY"),
        ("do you want tea", "YOU WANT TEA YOU"),
        ("are we early", "WE EARLY WE"),
        ("can he drive now", "HE DRIVE NOW CAN"),
        ("do you see the screen", "YOU SEE SCREEN YOU"),
        ("is your phone okay", "YOUR PHONE OKAY YOU"),
        ("can they wait outside", "THEY WAIT OUTSIDE CAN"),
        ("do you remember me", "YOU REMEMBER ME YOU"),
        ("can i sit here", "I SIT HERE CAN"),
        ("did we miss the train", "WE MISS TRAIN WE"),
        ("are your hands cold", "YOUR HAND COLD YOU"),
        ("can you stay longer", "YOU STAY LONGER CAN"),
        ("does she know the address", "SHE KNOW ADDRESS SHE"),
        ("did they finish homework", "THEY FINISH HOMEWORK THEY"),
        ("is he your teacher", "HE YOUR TEACHER HE"),
        ("do i need a ticket", "I NEED TICKET I"),
        ("can we leave now", "WE LEAVE NOW CAN"),
        ("are you feeling better", "YOU FEEL BETTER YOU"),
        ("did she send the email", "SHE SEND EMAIL SHE"),
        ("can you repeat that slowly", "YOU REPEAT THAT SLOW CAN"),
        ("are they waiting outside", "THEY WAIT OUTSIDE THEY"),
        ("do you hear the alarm", "YOU HEAR ALARM YOU"),
        ("can he help us later", "HE HELP US LATER CAN"),
    ],
    "topic_comment_or_reordering": [
        ("i need the red folder", "RED FOLDER I NEED"),
        ("i understand your idea", "YOUR IDEA I UNDERSTAND"),
        ("we like that restaurant", "THAT RESTAURANT WE LIKE"),
        ("i enjoy math class", "MATH CLASS I ENJOY"),
        ("i want coffee now", "COFFEE I WANT NOW"),
        ("i know the answer", "ANSWER I KNOW"),
        ("i found my keys under the table", "MY KEY UNDER TABLE I FIND"),
        ("she already watched that movie", "THAT MOVIE SHE WATCH ALREADY"),
        ("my teacher recommended this book", "THIS BOOK MY TEACHER RECOMMEND"),
        ("i moved the blue chair", "BLUE CHAIR I MOVE"),
        ("i can answer your question", "YOUR QUESTION I ANSWER CAN"),
        ("i forgot the bus route", "BUS ROUTE I FORGET"),
        ("my left hand still hurts", "MY LEFT HAND STILL HURT"),
        ("i submitted the online form", "ONLINE FORM I SUBMIT"),
        ("i saved this address", "THIS ADDRESS I SAVE"),
        ("he locked the front door", "FRONT DOOR HE LOCK"),
        ("my sister's apartment is small", "MY SISTER APARTMENT SMALL"),
        ("i put your notebook on the desk", "YOUR NOTEBOOK DESK ON I PUT"),
        ("i already bought the train ticket", "TRAIN TICKET I BUY ALREADY"),
        ("i left my wallet in the car", "MY WALLET CAR IN I LEAVE"),
        ("i like this soup more", "THIS SOUP I LIKE MORE"),
        ("i missed the email attachment", "EMAIL ATTACHMENT I MISS"),
        ("i remember your name now", "YOUR NAME I REMEMBER NOW"),
        ("i finished that homework early", "THAT HOMEWORK I FINISH EARLY"),
        ("my doctor appointment is next month", "MY DOCTOR APPOINTMENT NEXT-MONTH"),
        ("we solve this problem together", "THIS PROBLEM WE SOLVE TOGETHER"),
        ("science is my favorite class", "MY FAVORITE CLASS SCIENCE"),
        ("i saw your jacket on the chair", "YOUR JACKET CHAIR ON I SEE"),
        ("i kept the parking ticket", "PARKING TICKET I KEEP"),
        ("i remember your coffee order", "YOUR COFFEE ORDER I REMEMBER"),
    ],
}


def normalize_english(text: str) -> str:
    """Normalize English text to the repo's lowercase training style."""
    return " ".join(token.lower() for token in WORD_PATTERN.findall((text or "").lower()))


def normalize_gloss(text: str) -> str:
    """Normalize gloss text to uppercase token sequences."""
    return " ".join(str(text).strip().upper().split())


def load_json_pairs(path: Path) -> list[dict[str, str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("data", [])
    if not isinstance(payload, list):
        raise ValueError(f"{path} is not a list-based pair dataset.")
    return [
        {"english": normalize_english(item["english"]), "gloss": normalize_gloss(item["gloss"])}
        for item in payload
        if isinstance(item, dict) and "english" in item and "gloss" in item
    ]


def build_generated_entries(existing_pairs: set[tuple[str, str]], existing_english: set[str]) -> list[dict[str, object]]:
    generated_entries: list[dict[str, object]] = []
    seen_pairs: set[tuple[str, str]] = set()
    seen_english: set[str] = set()

    for category, pairs in GENERATED_PAIRS.items():
        if len(pairs) != 30:
            raise ValueError(f"Category {category} does not contain 30 pairs.")
        for index, (english_text, gloss_text) in enumerate(pairs, start=1):
            english = normalize_english(english_text)
            gloss = normalize_gloss(gloss_text)
            pair = (english, gloss)

            if pair in existing_pairs:
                raise ValueError(f"Generated pair duplicates an original pair: {pair}")
            if pair in seen_pairs:
                raise ValueError(f"Generated pair duplicated internally: {pair}")
            if english in existing_english:
                raise ValueError(f"Generated English prompt already exists in original data: {english}")
            if english in seen_english:
                raise ValueError(f"Generated English prompt duplicated internally: {english}")

            generated_entries.append(
                {
                    "pair_id": f"gen_{len(generated_entries) + 1:04d}",
                    "english": english,
                    "gloss": gloss,
                    "source_kind": "generated_augmentation",
                    "source_files": [str(GENERATED_OUTPUT_PATH.relative_to(PROJECT_ROOT))],
                    "manual_category": category,
                    "notes": "Synthetic augmentation created for grammar-focused training expansion.",
                }
            )
            seen_pairs.add(pair)
            seen_english.add(english)

    if len(generated_entries) != 150:
        raise ValueError(f"Expected 150 generated entries, found {len(generated_entries)}.")
    return generated_entries


def build_original_entries() -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    by_pair: dict[tuple[str, str], dict[str, object]] = {}
    english_to_glosses: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    source_counts: dict[str, int] = {}
    duplicates_removed = 0

    for source_path in ORIGINAL_DATASET_FILES:
        records = load_json_pairs(source_path)
        source_counts[str(source_path.relative_to(PROJECT_ROOT))] = len(records)
        for record in records:
            pair = (record["english"], record["gloss"])
            source_key = str(source_path.relative_to(PROJECT_ROOT))

            if pair not in by_pair:
                by_pair[pair] = {
                    "english": record["english"],
                    "gloss": record["gloss"],
                    "source_kind": "original_curated",
                    "source_files": [source_key],
                }
            else:
                duplicates_removed += 1
                if source_key not in by_pair[pair]["source_files"]:
                    by_pair[pair]["source_files"].append(source_key)

            english_to_glosses[record["english"]][record["gloss"]].add(source_key)

    conflicts: list[dict[str, object]] = []
    conflict_group_for_pair: dict[tuple[str, str], str] = {}
    for conflict_index, (english, gloss_map) in enumerate(sorted(english_to_glosses.items()), start=1):
        if len(gloss_map) <= 1:
            continue
        conflict_id = f"conflict_{conflict_index:04d}"
        variants = []
        for gloss, source_files in sorted(gloss_map.items()):
            variants.append({"gloss": gloss, "source_files": sorted(source_files)})
            conflict_group_for_pair[(english, gloss)] = conflict_id
        conflicts.append(
            {
                "conflict_id": conflict_id,
                "english": english,
                "variant_count": len(variants),
                "variants": variants,
            }
        )

    original_entries: list[dict[str, object]] = []
    for entry_index, pair in enumerate(sorted(by_pair), start=1):
        entry = dict(by_pair[pair])
        entry["pair_id"] = f"orig_{entry_index:04d}"
        if pair in conflict_group_for_pair:
            entry["conflict_group"] = conflict_group_for_pair[pair]
        original_entries.append(entry)

    summary = {
        "source_counts": source_counts,
        "original_unique_pairs": len(original_entries),
        "exact_duplicates_removed": duplicates_removed,
        "conflict_group_count": len(conflicts),
    }
    return original_entries, conflicts, summary


def build_merged_entries(
    original_entries: list[dict[str, object]],
    generated_entries: list[dict[str, object]],
) -> list[dict[str, object]]:
    merged_entries: list[dict[str, object]] = []
    for entry_index, entry in enumerate(original_entries + generated_entries, start=1):
        merged = dict(entry)
        merged["pair_id"] = f"merged_{entry_index:04d}"
        merged_entries.append(merged)
    return merged_entries


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    from src.data.build_active_gloss_pipeline import build_active_datasets

    report = build_active_datasets()
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
