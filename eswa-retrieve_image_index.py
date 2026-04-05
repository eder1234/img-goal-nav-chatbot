import json
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


SYNONYMS = {
    "couch": "sofa",
    "canape": "sofa",
    "canapé": "sofa",
    "seat": "chair",
    "chairs": "chair",
    "beds": "bed",
    "windows": "window",
    "doors": "door",
    "television": "screen",
    "tv": "screen",
    "windowpane": "window",
    "pillows": "cushion",
    "sit down": "sit",
    "sit": "sit",
    "rest": "sit",
    "sleep": "sleep",
    "lie down": "sleep",
    "lie": "sleep",
    "watch television": "watch_tv",
    "watch tv": "watch_tv",
    "watch a movie": "watch_tv",
    "watch": "watch_tv",
    "look outside": "look_outside",
    "look out": "look_outside",
    "open": "open",
    "store": "store_items",
    "put away": "store_items",
    "put things": "store_items",
    "livingroom": "living room",
    "bedroom": "bedroom",
    "hotelroom": "hotel room",
}

INTENT_TO_OBJECTS = {
    "sit": {"chair", "sofa", "bench", "stool"},
    "watch_tv": {"screen", "sofa", "chair"},
    "sleep": {"bed", "sofa"},
    "look_outside": {"window"},
    "open": {"door", "window", "drawer", "cabinet"},
    "store_items": {"cabinet", "drawer", "shelf", "countertop"},
}

STOPWORDS = {
    "i", "want", "to", "the", "a", "an", "please", "would", "like", "me",
    "on", "in", "at", "for", "with", "and", "or", "near", "by", "my", "we",
    "go", "into", "from", "of", "is", "it", "this", "that", "there", "down",
}

PHRASE_CANDIDATES = [
    "coffee table",
    "abstract art",
    "wall lamp",
    "double door",
    "hotel room",
    "living room",
    "kitchen",
    "bedroom",
    "look outside",
    "watch tv",
    "sit down",
    "lie down",
]

SCENE_TYPES = {"hotel room", "living room", "kitchen", "bedroom"}


def normalize_text(text: str) -> str:
    return " ".join(text.lower().strip().split())


def canonicalize(text: str) -> str:
    t = normalize_text(text)
    return SYNONYMS.get(t, t)


def singularize_token(token: str) -> str:
    token = canonicalize(token)
    if token.endswith("ies") and len(token) > 3:
        return token[:-3] + "y"
    if token.endswith("s") and not token.endswith("ss") and len(token) > 3:
        return token[:-1]
    return token


def tokenize(text: str) -> List[str]:
    chars = []
    for ch in text.lower():
        if ch.isalnum() or ch in {"_", "-", " "}:
            chars.append(ch)
        else:
            chars.append(" ")
    raw = "".join(chars).split()
    tokens = []
    for tok in raw:
        tok = singularize_token(tok)
        if tok and tok not in STOPWORDS:
            tokens.append(tok)
    return tokens


def load_dataset(json_path: Path) -> Dict[str, Any]:
    return json.loads(json_path.read_text(encoding="utf-8"))


def infer_intents(user_text: str) -> Set[str]:
    t = normalize_text(user_text)
    intents = set()

    if "sit" in t or "seat" in t or "rest" in t:
        intents.add("sit")
    if "watch" in t and ("tv" in t or "television" in t or "screen" in t or "movie" in t):
        intents.add("watch_tv")
    if "sleep" in t or "lie down" in t or "lie" in t:
        intents.add("sleep")
    if "window" in t or "outside" in t or "look out" in t:
        intents.add("look_outside")
    if "open" in t:
        intents.add("open")
    if "store" in t or "put away" in t or "put things" in t:
        intents.add("store_items")

    direct = canonicalize(t)
    if direct in INTENT_TO_OBJECTS:
        intents.add(direct)

    return intents


def extract_query_terms(user_text: str) -> Set[str]:
    terms = set(tokenize(user_text))
    norm = normalize_text(user_text)

    for phrase in PHRASE_CANDIDATES:
        if phrase in norm:
            terms.add(phrase)

    return terms


def canonical_attr_set(values: List[str]) -> Set[str]:
    out = set()
    for v in values:
        vv = normalize_text(str(v))
        out.add(singularize_token(vv))
        out.add(canonicalize(vv.replace(" ", "_")))
        out.add(canonicalize(vv))
    return {x for x in out if x}


def object_match_score(
    obj: Dict[str, Any],
    intents: Set[str],
    query_terms: Set[str],
    is_clarification_turn: bool,
) -> Tuple[float, List[str]]:
    score = 0.0
    reasons = []

    obj_name_raw = normalize_text(str(obj.get("name", "")))
    obj_name = singularize_token(obj_name_raw)
    obj_attrs = canonical_attr_set([str(a) for a in obj.get("attributes", [])])
    obj_affs = set()

    for a in obj.get("affordances", []):
        aa = normalize_text(str(a)).replace(" ", "_")
        obj_affs.add(canonicalize(aa))
        obj_affs.add(canonicalize(aa.replace("_", " ")))

    # direct object mention
    if obj_name in query_terms:
        score += 4.0 if is_clarification_turn else 2.5
        reasons.append(f"object:{obj_name}")

    # phrase/object exact-ish mention
    for term in query_terms:
        if term == obj_name_raw:
            score += 3.0 if is_clarification_turn else 2.0
            reasons.append(f"object_phrase:{term}")
        elif term in obj_name_raw or obj_name_raw in term:
            score += 1.0
            reasons.append(f"object_partial:{term}")

    # attribute mention
    for term in query_terms:
        tt = canonicalize(term.replace(" ", "_"))
        if term in obj_attrs or tt in obj_attrs:
            score += 1.2 if is_clarification_turn else 0.7
            reasons.append(f"attribute:{term}")

    # intent/object compatibility
    for intent in intents:
        if obj_name in INTENT_TO_OBJECTS.get(intent, set()):
            score += 1.0
            reasons.append(f"intent_object:{intent}->{obj_name}")
        if intent in obj_affs:
            score += 1.3
            reasons.append(f"affordance:{intent}")

    return score, reasons


def scene_match_score(
    image: Dict[str, Any],
    query_terms: Set[str],
    is_clarification_turn: bool,
) -> Tuple[float, List[str]]:
    score = 0.0
    reasons = []

    summary = normalize_text(str(image.get("summary", "")))
    scene_attributes = {normalize_text(str(a)) for a in image.get("scene_attributes", [])}

    for term in query_terms:
        if term in SCENE_TYPES and term in summary:
            score += 1.0 if is_clarification_turn else 0.6
            reasons.append(f"summary_scene:{term}")
        elif term in summary:
            score += 0.3
            reasons.append(f"summary:{term}")

        if term in scene_attributes:
            score += 0.7 if is_clarification_turn else 0.4
            reasons.append(f"scene:{term}")

    return score, reasons


def image_score(
    image: Dict[str, Any],
    user_text: str,
    is_clarification_turn: bool,
) -> Dict[str, Any]:
    intents = infer_intents(user_text)
    query_terms = extract_query_terms(user_text)

    score = 0.0
    reasons: List[str] = []
    matched_objects: Set[str] = set()

    scene_score, scene_reasons = scene_match_score(image, query_terms, is_clarification_turn)
    score += scene_score
    reasons.extend(scene_reasons)

    objects = image.get("objects", [])
    for obj in objects:
        obj_score, obj_reasons = object_match_score(obj, intents, query_terms, is_clarification_turn)
        if obj_score > 0:
            score += obj_score
            reasons.extend(obj_reasons)
            matched_objects.add(singularize_token(str(obj.get("name", ""))))

    return {
        "index": image["index"],
        "filename": image["filename"],
        "summary": image.get("summary", ""),
        "matched_objects": sorted(matched_objects),
        "score_raw": score,
        "reasons": reasons,
    }


def normalize_scores(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not candidates:
        return candidates

    raw_scores = [c["score_raw"] for c in candidates]
    max_score = max(raw_scores)
    min_score = min(raw_scores)

    if max_score <= 0:
        for c in candidates:
            c["likeness"] = 0.0
        return candidates

    if max_score == min_score:
        n = len(candidates)
        for rank, c in enumerate(candidates):
            c["likeness"] = round(max(0.0, 1.0 - (rank / max(n, 1)) * 0.15), 4)
        return candidates

    denom = max_score - min_score
    for c in candidates:
        c["likeness"] = round((c["score_raw"] - min_score) / denom, 4)

    return candidates


def rank_candidates(
    dataset: Dict[str, Any],
    current_text: str,
    is_clarification_turn: bool,
) -> List[Dict[str, Any]]:
    ranked = []
    for image in dataset["images"]:
        scored = image_score(image, current_text, is_clarification_turn)
        if scored["score_raw"] > 0:
            ranked.append(scored)

    ranked.sort(key=lambda x: (-x["score_raw"], x["index"]))
    ranked = normalize_scores(ranked)
    return ranked


def build_clarification_question(candidates: List[Dict[str, Any]], top_k: int = 5) -> str:
    object_names = sorted({
        obj
        for cand in candidates[:top_k]
        for obj in cand["matched_objects"]
        if obj
    })

    if object_names:
        return "Ambiguous request. Which one do you mean: " + ", ".join(object_names) + "?"
    return "Ambiguous request. Please clarify the target object or room."


def is_clear_ranking(ranked: List[Dict[str, Any]]) -> bool:
    if len(ranked) <= 1:
        return True

    top = ranked[0]["score_raw"]
    second = ranked[1]["score_raw"]

    if top <= 0:
        return False

    if top == second:
        return False

    if second == 0 and top > 0:
        return True

    ratio = top / second
    gap = top - second

    return ratio >= 1.5 and gap >= 1.0


def run_dialog(
    dataset: Dict[str, Any],
    query: str,
    clarifications: List[str],
    max_tries: int,
) -> Dict[str, Any]:
    if max_tries < 1:
        raise ValueError("max_tries must be >= 1")

    turns = []
    final_ranked: List[Dict[str, Any]] = []
    ambiguity_absent = False

    current_text = query.strip()
    remaining_clarifications = [c.strip() for c in clarifications if c.strip()]

    for attempt in range(1, max_tries + 1):
        is_clarification_turn = attempt > 1
        ranked = rank_candidates(
            dataset=dataset,
            current_text=current_text,
            is_clarification_turn=is_clarification_turn,
        )
        final_ranked = ranked

        if not ranked:
            turns.append({
                "try": attempt,
                "query_used": current_text,
                "question": None,
                "status": "no_match",
            })
            break

        # If clarification exists, use it before declaring success,
        # unless ranking is already unique.
        if len(ranked) == 1:
            ambiguity_absent = True
            turns.append({
                "try": attempt,
                "query_used": current_text,
                "question": None,
                "status": "unique",
            })
            break

        if remaining_clarifications:
            question = build_clarification_question(ranked)
            clarification = remaining_clarifications.pop(0)

            turns.append({
                "try": attempt,
                "query_used": current_text,
                "question": question,
                "user_reply": clarification,
                "status": "clarified",
            })

            current_text = clarification
            continue

        if is_clear_ranking(ranked):
            ambiguity_absent = True
            turns.append({
                "try": attempt,
                "query_used": current_text,
                "question": None,
                "status": "clear_ranking",
            })
            break

        question = build_clarification_question(ranked)
        turns.append({
            "try": attempt,
            "query_used": current_text,
            "question": question,
            "status": "ambiguous",
        })
        break

    return {
        "query": query,
        "max_tries": max_tries,
        "tries_used": len(turns),
        "ambiguity_absent": ambiguity_absent,
        "dialog": turns,
        "ranked_candidates": [
            {
                "index": c["index"],
                "filename": c["filename"],
                "likeness": c["likeness"],
                "matched_objects": c["matched_objects"],
                "summary": c["summary"],
                "reasons": c["reasons"],
                "score_raw": c["score_raw"],
            }
            for c in final_ranked
        ],
    }


def print_result(result: Dict[str, Any], top_k: Optional[int]) -> None:
    print("=" * 80)
    print(f"Query: {result['query']}")
    print(f"Max tries: {result['max_tries']}")
    print(f"Tries used: {result['tries_used']}")
    print(f"Ambiguity absent: {result['ambiguity_absent']}")
    print("=" * 80)

    if result["dialog"]:
        print("Dialog trace:")
        for turn in result["dialog"]:
            print(f"- Try {turn['try']}: status={turn['status']}")
            print(f"  Query used: {turn['query_used']}")
            if turn.get("question"):
                print(f"  Question: {turn['question']}")
            if turn.get("user_reply"):
                print(f"  User reply: {turn['user_reply']}")
        print("=" * 80)

    ranked = result["ranked_candidates"]
    if top_k is not None:
        ranked = ranked[:top_k]

    if not ranked:
        print("No matching candidates.")
        return

    print("Ranked candidates:")
    for cand in ranked:
        print(
            f"index={cand['index']} | score_raw={cand['score_raw']:.4f} | "
            f"likeness={cand['likeness']:.4f} | "
            f"matched_objects={cand['matched_objects']} | filename={cand['filename']}"
        )
        print(f"  summary={cand['summary']}")
        print(f"  reasons={cand['reasons']}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Resolve a user command to ranked image indices using a prebuilt JSON dataset."
    )
    parser.add_argument("--json_path", required=True, help="Path to dataset JSON.")
    parser.add_argument("--query", required=True, help='User command, e.g. "I want to sit down".')
    parser.add_argument(
        "--clarifications",
        nargs="*",
        default=[],
        help='Optional clarification replies in order, e.g. sofa white living room',
    )
    parser.add_argument(
        "--max_tries",
        type=int,
        default=2,
        help="Maximum number of tries, including the initial query.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=None,
        help="Optional number of top ranked candidates to print.",
    )
    parser.add_argument(
        "--output_json",
        default=None,
        help="Optional path to save the ranked result as JSON.",
    )
    args = parser.parse_args()

    dataset = load_dataset(Path(args.json_path))
    result = run_dialog(
        dataset=dataset,
        query=args.query,
        clarifications=args.clarifications,
        max_tries=args.max_tries,
    )

    print_result(result, args.top_k)

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print("=" * 80)
        print(f"Saved result JSON to: {output_path}")


if __name__ == "__main__":
    main()
