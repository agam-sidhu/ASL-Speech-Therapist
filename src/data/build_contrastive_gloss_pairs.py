"""Build contrastive English->ASL gloss pairs for semantic discrimination."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"

MASTER_DATASET_PATH = DATA_DIR / "active" / "project_finetune_v2_v4_contrastive.json"
CONTRASTIVE_OUTPUT_PATH = DATA_DIR / "active" / "project_finetune_v2_v4_contrastive.json"
MASTER_PLUS_CONTRASTIVE_PATH = DATA_DIR / "active" / "project_finetune_v2_v4_contrastive.json"
REPORT_OUTPUT_PATH = DATA_DIR / "reports" / "data_pipeline_report.json"

WORD_PATTERN = re.compile(r"[A-Za-z']+")

CONTRASTIVE_GROUPS: list[dict[str, object]] = [
    {
        "contrast_group": "family_doctor_identity",
        "manual_category": "family_people_professions",
        "contrast_axis": "family_member",
        "examples": [
            ("my cousin is a doctor", "MY COUSIN DOCTOR"),
            ("my father is a doctor", "MY FATHER DOCTOR"),
            ("my brother is a doctor", "MY BROTHER DOCTOR"),
            ("my sister is a doctor", "MY SISTER DOCTOR"),
            ("my friend is a doctor", "MY FRIEND DOCTOR"),
        ],
    },
    {
        "contrast_group": "family_teacher_identity",
        "manual_category": "family_people_professions",
        "contrast_axis": "family_member",
        "examples": [
            ("my mother is a teacher", "MY MOTHER TEACHER"),
            ("my cousin is a teacher", "MY COUSIN TEACHER"),
            ("my brother is a teacher", "MY BROTHER TEACHER"),
            ("my sister is a teacher", "MY SISTER TEACHER"),
            ("my friend is a teacher", "MY FRIEND TEACHER"),
        ],
    },
    {
        "contrast_group": "family_nurse_identity",
        "manual_category": "family_people_professions",
        "contrast_axis": "family_member",
        "examples": [
            ("my mother is a nurse", "MY MOTHER NURSE"),
            ("my father is a nurse", "MY FATHER NURSE"),
            ("my cousin is a nurse", "MY COUSIN NURSE"),
            ("my sister is a nurse", "MY SISTER NURSE"),
            ("my friend is a nurse", "MY FRIEND NURSE"),
        ],
    },
    {
        "contrast_group": "family_hospital_work",
        "manual_category": "family_people_professions",
        "contrast_axis": "family_member",
        "examples": [
            ("my cousin works at the hospital", "MY COUSIN HOSPITAL WORK"),
            ("my father works at the hospital", "MY FATHER HOSPITAL WORK"),
            ("my brother works at the hospital", "MY BROTHER HOSPITAL WORK"),
            ("my sister works at the hospital", "MY SISTER HOSPITAL WORK"),
            ("my friend works at the hospital", "MY FRIEND HOSPITAL WORK"),
        ],
    },
    {
        "contrast_group": "family_school_work",
        "manual_category": "family_people_professions",
        "contrast_axis": "family_member",
        "examples": [
            ("my mother works at the school", "MY MOTHER SCHOOL WORK"),
            ("my father works at the school", "MY FATHER SCHOOL WORK"),
            ("my brother works at the school", "MY BROTHER SCHOOL WORK"),
            ("my sister works at the school", "MY SISTER SCHOOL WORK"),
            ("my friend works at the school", "MY FRIEND SCHOOL WORK"),
        ],
    },
    {
        "contrast_group": "family_visit_tomorrow",
        "manual_category": "family_people_professions",
        "contrast_axis": "family_member",
        "examples": [
            ("my mother visits me tomorrow", "TOMORROW MY MOTHER VISIT ME"),
            ("my father visits me tomorrow", "TOMORROW MY FATHER VISIT ME"),
            ("my brother visits me tomorrow", "TOMORROW MY BROTHER VISIT ME"),
            ("my sister visits me tomorrow", "TOMORROW MY SISTER VISIT ME"),
            ("my friend visits me tomorrow", "TOMORROW MY FRIEND VISIT ME"),
        ],
    },
    {
        "contrast_group": "family_call_today",
        "manual_category": "family_people_professions",
        "contrast_axis": "family_member",
        "examples": [
            ("my mother calls me today", "TODAY MY MOTHER CALL ME"),
            ("my father calls me today", "TODAY MY FATHER CALL ME"),
            ("my brother calls me today", "TODAY MY BROTHER CALL ME"),
            ("my sister calls me today", "TODAY MY SISTER CALL ME"),
            ("my friend calls me today", "TODAY MY FRIEND CALL ME"),
        ],
    },
    {
        "contrast_group": "family_help_today",
        "manual_category": "family_people_professions",
        "contrast_axis": "family_member",
        "examples": [
            ("my mother helps me today", "TODAY MY MOTHER HELP ME"),
            ("my father helps me today", "TODAY MY FATHER HELP ME"),
            ("my brother helps me today", "TODAY MY BROTHER HELP ME"),
            ("my sister helps me today", "TODAY MY SISTER HELP ME"),
            ("my friend helps me today", "TODAY MY FRIEND HELP ME"),
        ],
    },
    {
        "contrast_group": "family_study_english",
        "manual_category": "family_people_professions",
        "contrast_axis": "family_member",
        "examples": [
            ("my mother studies english", "MY MOTHER STUDY ENGLISH"),
            ("my father studies english", "MY FATHER STUDY ENGLISH"),
            ("my brother studies english", "MY BROTHER STUDY ENGLISH"),
            ("my sister studies english", "MY SISTER STUDY ENGLISH"),
            ("my friend studies english", "MY FRIEND STUDY ENGLISH"),
        ],
    },
    {
        "contrast_group": "family_teach_math",
        "manual_category": "family_people_professions",
        "contrast_axis": "family_member",
        "examples": [
            ("my mother teaches math", "MY MOTHER TEACH MATH"),
            ("my cousin teaches math", "MY COUSIN TEACH MATH"),
            ("my brother teaches math", "MY BROTHER TEACH MATH"),
            ("my sister teaches math", "MY SISTER TEACH MATH"),
            ("my friend teaches math", "MY FRIEND TEACH MATH"),
        ],
    },
    {
        "contrast_group": "learn_new_sign_by_time",
        "manual_category": "time_daily_activity",
        "contrast_axis": "time_reference",
        "examples": [
            ("today i learn a new sign", "TODAY I LEARN NEW SIGN"),
            ("yesterday i learn a new sign", "YESTERDAY I LEARN NEW SIGN"),
            ("tomorrow i learn a new sign", "TOMORROW I LEARN NEW SIGN"),
            ("this morning i learn a new sign", "THIS-MORNING I LEARN NEW SIGN"),
            ("tonight i learn a new sign", "TONIGHT I LEARN NEW SIGN"),
        ],
    },
    {
        "contrast_group": "practice_asl_by_time",
        "manual_category": "time_daily_activity",
        "contrast_axis": "time_reference",
        "examples": [
            ("today i practice asl", "TODAY I PRACTICE ASL"),
            ("yesterday i practice asl", "YESTERDAY I PRACTICE ASL"),
            ("tomorrow i practice asl", "TOMORROW I PRACTICE ASL"),
            ("this morning i practice asl", "THIS-MORNING I PRACTICE ASL"),
            ("tonight i practice asl", "TONIGHT I PRACTICE ASL"),
        ],
    },
    {
        "contrast_group": "finish_homework_by_time",
        "manual_category": "time_daily_activity",
        "contrast_axis": "time_reference",
        "examples": [
            ("today i finish homework", "TODAY I FINISH HOMEWORK"),
            ("yesterday i finish homework", "YESTERDAY I FINISH HOMEWORK"),
            ("tomorrow i finish homework", "TOMORROW I FINISH HOMEWORK"),
            ("this afternoon i finish homework", "THIS-AFTERNOON I FINISH HOMEWORK"),
            ("tonight i finish homework", "TONIGHT I FINISH HOMEWORK"),
        ],
    },
    {
        "contrast_group": "go_to_class_by_time",
        "manual_category": "time_daily_activity",
        "contrast_axis": "time_reference",
        "examples": [
            ("today i go to class", "TODAY I GO CLASS"),
            ("yesterday i go to class", "YESTERDAY I GO CLASS"),
            ("tomorrow i go to class", "TOMORROW I GO CLASS"),
            ("this morning i go to class", "THIS-MORNING I GO CLASS"),
            ("every day i go to class", "EVERY-DAY I GO CLASS"),
        ],
    },
    {
        "contrast_group": "leave_home_by_time",
        "manual_category": "time_daily_activity",
        "contrast_axis": "time_reference",
        "examples": [
            ("today i leave home early", "TODAY I LEAVE HOME EARLY"),
            ("yesterday i leave home early", "YESTERDAY I LEAVE HOME EARLY"),
            ("tomorrow i leave home early", "TOMORROW I LEAVE HOME EARLY"),
            ("this morning i leave home early", "THIS-MORNING I LEAVE HOME EARLY"),
            ("every day i leave home early", "EVERY-DAY I LEAVE HOME EARLY"),
        ],
    },
    {
        "contrast_group": "study_library_by_time",
        "manual_category": "time_daily_activity",
        "contrast_axis": "time_reference",
        "examples": [
            ("today i study at the library", "TODAY I STUDY LIBRARY"),
            ("yesterday i study at the library", "YESTERDAY I STUDY LIBRARY"),
            ("tomorrow i study at the library", "TOMORROW I STUDY LIBRARY"),
            ("this afternoon i study at the library", "THIS-AFTERNOON I STUDY LIBRARY"),
            ("every day i study at the library", "EVERY-DAY I STUDY LIBRARY"),
        ],
    },
    {
        "contrast_group": "work_downtown_by_time",
        "manual_category": "time_daily_activity",
        "contrast_axis": "time_reference",
        "examples": [
            ("today i work downtown", "TODAY I WORK DOWNTOWN"),
            ("yesterday i work downtown", "YESTERDAY I WORK DOWNTOWN"),
            ("tomorrow i work downtown", "TOMORROW I WORK DOWNTOWN"),
            ("this morning i work downtown", "THIS-MORNING I WORK DOWNTOWN"),
            ("every day i work downtown", "EVERY-DAY I WORK DOWNTOWN"),
        ],
    },
    {
        "contrast_group": "wake_up_early_by_time",
        "manual_category": "time_daily_activity",
        "contrast_axis": "time_reference",
        "examples": [
            ("today i wake up early", "TODAY I WAKE-UP EARLY"),
            ("yesterday i wake up early", "YESTERDAY I WAKE-UP EARLY"),
            ("tomorrow i wake up early", "TOMORROW I WAKE-UP EARLY"),
            ("this morning i wake up early", "THIS-MORNING I WAKE-UP EARLY"),
            ("every day i wake up early", "EVERY-DAY I WAKE-UP EARLY"),
        ],
    },
    {
        "contrast_group": "cannot_verb_you",
        "manual_category": "sensory_understanding_communication",
        "contrast_axis": "verb_choice",
        "examples": [
            ("i cannot hear her", "I HEAR HER CAN NOT"),
            ("i cannot see you", "I SEE YOU CAN NOT"),
            ("i cannot understand you", "I UNDERSTAND YOU CAN NOT"),
            ("i cannot help you", "I HELP YOU CAN NOT"),
            ("i cannot call you", "I CALL YOU CAN NOT"),
        ],
    },
    {
        "contrast_group": "can_you_do_that",
        "manual_category": "sensory_understanding_communication",
        "contrast_axis": "verb_choice",
        "examples": [
            ("can you explain that", "THAT EXPLAIN YOU CAN"),
            ("can you repeat this", "THIS REPEAT YOU CAN"),
            ("can you show that", "THAT SHOW YOU CAN"),
            ("can you write that", "THAT WRITE YOU CAN"),
            ("can you sign that", "THAT SIGN YOU CAN"),
        ],
    },
    {
        "contrast_group": "do_you_know_information",
        "manual_category": "sensory_understanding_communication",
        "contrast_axis": "object_choice",
        "examples": [
            ("do you know my name", "YOU KNOW MY NAME YOU"),
            ("do you know her name", "YOU KNOW HER NAME YOU"),
            ("do you know his number", "YOU KNOW HIS NUMBER YOU"),
            ("do you know the address", "YOU KNOW ADDRESS YOU"),
            ("do you know the answer", "YOU KNOW ANSWER YOU"),
        ],
    },
    {
        "contrast_group": "i_verb_your_name",
        "manual_category": "sensory_understanding_communication",
        "contrast_axis": "verb_choice",
        "examples": [
            ("i know your name", "I KNOW YOUR NAME"),
            ("i remember your name", "I REMEMBER YOUR NAME"),
            ("i forget your name", "I FORGET YOUR NAME"),
            ("i spell your name", "I SPELL YOUR NAME"),
            ("i write your name", "I WRITE YOUR NAME"),
        ],
    },
    {
        "contrast_group": "did_she_verb_you",
        "manual_category": "sensory_understanding_communication",
        "contrast_axis": "verb_choice",
        "examples": [
            ("did she call him", "SHE CALL HIM SHE"),
            ("did she help you", "SHE HELP YOU SHE"),
            ("did she ask you", "SHE ASK YOU SHE"),
            ("did she text you", "SHE TEXT YOU SHE"),
            ("did she meet you", "SHE MEET YOU SHE"),
        ],
    },
    {
        "contrast_group": "do_you_verb_teacher",
        "manual_category": "sensory_understanding_communication",
        "contrast_axis": "verb_choice",
        "examples": [
            ("do you hear the teacher", "YOU HEAR TEACHER YOU"),
            ("do you see the teacher", "YOU SEE TEACHER YOU"),
            ("do you understand the teacher", "YOU UNDERSTAND TEACHER YOU"),
            ("do you know the teacher", "YOU KNOW TEACHER YOU"),
            ("do you remember the teacher", "YOU REMEMBER TEACHER YOU"),
        ],
    },
    {
        "contrast_group": "can_you_verb_me",
        "manual_category": "sensory_understanding_communication",
        "contrast_axis": "verb_choice",
        "examples": [
            ("can you text me", "YOU TEXT ME CAN"),
            ("can you call me", "YOU CALL ME CAN"),
            ("can you teach me", "YOU TEACH ME CAN"),
            ("can you explain to me", "YOU EXPLAIN ME CAN"),
            ("can you ask me", "YOU ASK ME CAN"),
        ],
    },
    {
        "contrast_group": "red_book_relation",
        "manual_category": "location_object_descriptive_relations",
        "contrast_axis": "relation",
        "examples": [
            ("the red book is under the table", "TABLE UNDER RED BOOK"),
            ("the red book is near the table", "TABLE NEAR RED BOOK"),
            ("the red book is behind the table", "TABLE BEHIND RED BOOK"),
            ("the red book is here", "RED BOOK HERE"),
            ("the red book is there", "RED BOOK THERE"),
        ],
    },
    {
        "contrast_group": "blue_backpack_relation",
        "manual_category": "location_object_descriptive_relations",
        "contrast_axis": "relation",
        "examples": [
            ("my blue backpack is under the chair", "MY BLUE BACKPACK CHAIR UNDER"),
            ("my blue backpack is near the chair", "MY BLUE BACKPACK CHAIR NEAR"),
            ("my blue backpack is behind the chair", "MY BLUE BACKPACK CHAIR BEHIND"),
            ("my blue backpack is here", "MY BLUE BACKPACK HERE"),
            ("my blue backpack is there", "MY BLUE BACKPACK THERE"),
        ],
    },
    {
        "contrast_group": "keys_relation",
        "manual_category": "location_object_descriptive_relations",
        "contrast_axis": "location_reference",
        "examples": [
            ("my keys are on the desk", "MY KEY DESK ON"),
            ("my keys are in the car", "MY KEY CAR IN"),
            ("my keys are under the chair", "MY KEY CHAIR UNDER"),
            ("my keys are near the phone", "MY KEY PHONE NEAR"),
            ("my keys are here", "MY KEY HERE"),
        ],
    },
    {
        "contrast_group": "computer_description",
        "manual_category": "location_object_descriptive_relations",
        "contrast_axis": "adjective",
        "examples": [
            ("my computer is new", "MY COMPUTER NEW"),
            ("my computer is old", "MY COMPUTER OLD"),
            ("my laptop is broken", "MY LAPTOP BROKEN"),
            ("my computer is big", "MY COMPUTER BIG"),
            ("my computer is small", "MY COMPUTER SMALL"),
        ],
    },
    {
        "contrast_group": "phone_description",
        "manual_category": "location_object_descriptive_relations",
        "contrast_axis": "adjective",
        "examples": [
            ("your phone is new", "YOUR PHONE NEW"),
            ("your phone is old", "YOUR PHONE OLD"),
            ("your phone is broken", "YOUR PHONE BROKEN"),
            ("your phone is big", "YOUR PHONE BIG"),
            ("your phone is small", "YOUR PHONE SMALL"),
        ],
    },
    {
        "contrast_group": "bathroom_near_place",
        "manual_category": "location_object_descriptive_relations",
        "contrast_axis": "reference_place",
        "examples": [
            ("the bathroom is near the office", "BATHROOM OFFICE NEAR"),
            ("the bathroom is near the library", "BATHROOM LIBRARY NEAR"),
            ("the bathroom is near the classroom", "BATHROOM CLASSROOM NEAR"),
            ("the bathroom is near the stairs", "BATHROOM STAIRS NEAR"),
            ("the bathroom is near the elevator", "BATHROOM ELEVATOR NEAR"),
        ],
    },
    {
        "contrast_group": "folder_color",
        "manual_category": "location_object_descriptive_relations",
        "contrast_axis": "color",
        "examples": [
            ("the folder is red", "FOLDER RED"),
            ("the folder is blue", "FOLDER BLUE"),
            ("the folder is green", "FOLDER GREEN"),
            ("the folder is yellow", "FOLDER YELLOW"),
            ("the folder is black", "FOLDER BLACK"),
        ],
    },
    {
        "contrast_group": "ready_polarity",
        "manual_category": "negation_yesno_contrastive",
        "contrast_axis": "negation_and_question_form",
        "examples": [
            ("are you ready", "YOU READY YOU"),
            ("are you not ready", "YOU READY NOT YOU"),
            ("you are ready", "YOU READY"),
            ("you are not ready", "YOU READY NOT"),
            ("are they ready", "THEY READY THEY"),
        ],
    },
    {
        "contrast_group": "have_time_tonight",
        "manual_category": "negation_yesno_contrastive",
        "contrast_axis": "negation_and_question_form",
        "examples": [
            ("do you have time tonight", "YOU HAVE TIME TONIGHT YOU"),
            ("do you not have time tonight", "YOU HAVE TIME TONIGHT NOT YOU"),
            ("you have time tonight", "YOU HAVE TIME TONIGHT"),
            ("you do not have time tonight", "YOU HAVE TIME TONIGHT NOT"),
            ("do they have time tonight", "THEY HAVE TIME TONIGHT THEY"),
        ],
    },
    {
        "contrast_group": "want_go_tomorrow",
        "manual_category": "negation_yesno_contrastive",
        "contrast_axis": "negation_and_question_form",
        "examples": [
            ("do you want to go tomorrow", "TOMORROW YOU WANT GO YOU"),
            ("do you not want to go tomorrow", "TOMORROW YOU WANT GO NOT YOU"),
            ("you want to go tomorrow", "TOMORROW YOU WANT GO"),
            ("you do not want to go tomorrow", "TOMORROW YOU WANT GO NOT"),
            ("does he want to go tomorrow", "TOMORROW HE WANT GO HE"),
        ],
    },
    {
        "contrast_group": "can_stay_after_class",
        "manual_category": "negation_yesno_contrastive",
        "contrast_axis": "negation_and_question_form",
        "examples": [
            ("can you stay after class", "AFTER CLASS YOU STAY CAN"),
            ("can you not stay after class", "AFTER CLASS YOU STAY NOT CAN"),
            ("you can stay after class", "AFTER CLASS YOU STAY CAN"),
            ("you cannot stay after class", "AFTER CLASS YOU STAY CAN NOT"),
            ("can she stay after class", "AFTER CLASS SHE STAY CAN"),
        ],
    },
    {
        "contrast_group": "know_answer_now",
        "manual_category": "negation_yesno_contrastive",
        "contrast_axis": "negation_and_question_form",
        "examples": [
            ("do you know the answer now", "YOU KNOW ANSWER NOW YOU"),
            ("do you not know the answer now", "YOU KNOW ANSWER NOW NOT YOU"),
            ("you know the answer now", "YOU KNOW ANSWER NOW"),
            ("you do not know the answer now", "YOU KNOW ANSWER NOW NOT"),
            ("does she know the answer now", "SHE KNOW ANSWER NOW SHE"),
        ],
    },
    {
        "contrast_group": "have_book_today",
        "manual_category": "negation_yesno_contrastive",
        "contrast_axis": "negation_and_question_form",
        "examples": [
            ("do you have the book today", "TODAY YOU HAVE BOOK YOU"),
            ("do you not have the book today", "TODAY YOU HAVE BOOK NOT YOU"),
            ("you have the book today", "TODAY YOU HAVE BOOK"),
            ("you do not have the book today", "TODAY YOU HAVE BOOK NOT"),
            ("does he have the book today", "TODAY HE HAVE BOOK HE"),
        ],
    },
    {
        "contrast_group": "understand_teacher_now",
        "manual_category": "negation_yesno_contrastive",
        "contrast_axis": "negation_and_question_form",
        "examples": [
            ("do you understand the teacher now", "YOU UNDERSTAND TEACHER NOW YOU"),
            ("do you not understand the teacher now", "YOU UNDERSTAND TEACHER NOW NOT YOU"),
            ("you understand the teacher now", "YOU UNDERSTAND TEACHER NOW"),
            ("you do not understand the teacher now", "YOU UNDERSTAND TEACHER NOW NOT"),
            ("do they understand the teacher now", "THEY UNDERSTAND TEACHER NOW THEY"),
        ],
    },
    {
        "contrast_group": "hear_me_now",
        "manual_category": "negation_yesno_contrastive",
        "contrast_axis": "negation_and_question_form",
        "examples": [
            ("can you hear me now", "YOU HEAR ME NOW CAN"),
            ("can you not hear me now", "YOU HEAR ME NOW NOT CAN"),
            ("you can hear me now", "YOU HEAR ME NOW CAN"),
            ("you cannot hear me now", "YOU HEAR ME NOW CAN NOT"),
            ("can she hear me now", "SHE HEAR ME NOW CAN"),
        ],
    },
]


def normalize_english(text: str) -> str:
    return " ".join(token.lower() for token in WORD_PATTERN.findall((text or "").lower()))


def normalize_gloss(text: str) -> str:
    return " ".join(str(text).strip().upper().split())


def load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def build_contrastive_dataset() -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    master_entries = load_json(MASTER_DATASET_PATH)
    if not isinstance(master_entries, list):
        raise ValueError("Master dataset must be a JSON list.")

    comparison_entries = list(master_entries)
    comparison_sources = [str(MASTER_DATASET_PATH.relative_to(PROJECT_ROOT))]

    existing_pairs = {(normalize_english(item["english"]), normalize_gloss(item["gloss"])) for item in comparison_entries}
    existing_english = {normalize_english(item["english"]) for item in comparison_entries}

    contrastive_entries: list[dict[str, object]] = []
    seen_pairs: set[tuple[str, str]] = set()
    seen_english: set[str] = set()
    category_counts: Counter[str] = Counter()
    axis_counts: Counter[str] = Counter()

    for group in CONTRASTIVE_GROUPS:
        contrast_group = str(group["contrast_group"])
        manual_category = str(group["manual_category"])
        contrast_axis = str(group["contrast_axis"])
        examples = list(group["examples"])
        if len(examples) != 5:
            raise ValueError(f"Contrast group {contrast_group} must contain 5 examples.")

        category_counts[manual_category] += len(examples)
        axis_counts[contrast_axis] += len(examples)

        for english_text, gloss_text in examples:
            english = normalize_english(english_text)
            gloss = normalize_gloss(gloss_text)
            pair = (english, gloss)

            if english in existing_english:
                raise ValueError(f"Contrastive English prompt already exists in comparison data: {english}")
            if pair in existing_pairs:
                raise ValueError(f"Contrastive pair already exists in comparison data: {pair}")
            if english in seen_english:
                raise ValueError(f"Contrastive English prompt duplicated internally: {english}")
            if pair in seen_pairs:
                raise ValueError(f"Contrastive pair duplicated internally: {pair}")

            contrastive_entries.append(
                {
                    "pair_id": f"contrastive_{len(contrastive_entries) + 1:04d}",
                    "english": english,
                    "gloss": gloss,
                    "source_kind": "contrastive_generated",
                    "source_files": [str(CONTRASTIVE_OUTPUT_PATH.relative_to(PROJECT_ROOT))],
                    "manual_category": manual_category,
                    "contrast_group": contrast_group,
                    "contrast_axis": contrast_axis,
                    "notes": "Synthetic contrastive augmentation created to improve semantic discrimination.",
                }
            )
            seen_english.add(english)
            seen_pairs.add(pair)

    if len(contrastive_entries) != 200:
        raise ValueError(f"Expected 200 contrastive entries, found {len(contrastive_entries)}.")

    master_plus_contrastive: list[dict[str, object]] = []
    duplicates_removed_when_merging = 0
    merged_seen_pairs: set[tuple[str, str]] = set()
    for entry in master_entries + contrastive_entries:
        pair = (normalize_english(str(entry["english"])), normalize_gloss(str(entry["gloss"])))
        if pair in merged_seen_pairs:
            duplicates_removed_when_merging += 1
            continue
        merged_seen_pairs.add(pair)
        merged = dict(entry)
        merged["pair_id"] = f"master_contrastive_{len(master_plus_contrastive) + 1:04d}"
        master_plus_contrastive.append(merged)

    report = {
        "master_dataset": str(MASTER_DATASET_PATH.relative_to(PROJECT_ROOT)),
        "validated_against": comparison_sources,
        "contrastive_pair_count": len(contrastive_entries),
        "contrastive_category_counts": dict(category_counts),
        "contrast_group_count": len(CONTRASTIVE_GROUPS),
        "average_contrast_group_size": round(len(contrastive_entries) / len(CONTRASTIVE_GROUPS), 2),
        "contrast_axis_counts": dict(axis_counts),
        "duplicates_removed_when_merging": duplicates_removed_when_merging,
        "master_plus_contrastive_count": len(master_plus_contrastive),
        "output_files": {
            "contrastive": str(CONTRASTIVE_OUTPUT_PATH.relative_to(PROJECT_ROOT)),
            "master_plus_contrastive": str(MASTER_PLUS_CONTRASTIVE_PATH.relative_to(PROJECT_ROOT)),
        },
    }
    return contrastive_entries, master_plus_contrastive, report


def main() -> None:
    from src.data.build_active_gloss_pipeline import build_active_datasets

    report = build_active_datasets()
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
