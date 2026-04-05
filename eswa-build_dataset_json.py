import os
import re
import json
import base64
import argparse
from pathlib import Path
from typing import Any, Dict, List

import requests


GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

SYSTEM_PROMPT = """
You analyze one indoor scene image and return ONLY valid JSON.

Your task:
1. Identify objects that are clearly present.
2. Identify scene-level attributes.
3. Infer affordances only if visually plausible.

Return this exact JSON schema:
{
  "summary": "short scene description",
  "objects": [
    {
      "name": "canonical object name",
      "count": 1,
      "attributes": ["attribute1", "attribute2"],
      "affordances": ["sit", "watch_tv", "open", "store_items"]
    }
  ],
  "scene_attributes": ["luxurious", "indoor", "curved seating"]
}

Rules:
- Use concise canonical English labels.
- Do not invent hidden objects.
- Keep only high-confidence visible objects.
- Affordances must be practical and short verbs/verb_phrases.
- Return JSON only, no markdown.
""".strip()

CANONICAL_OBJECT_MAP = {
    "couch": "sofa",
    "settee": "sofa",
    "loveseat": "sofa",
    "armchair": "chair",
    "seat": "chair",
    "tv": "screen",
    "television": "screen",
    "monitor": "screen",
    "windowpane": "window",
    "pillow": "cushion",
    "pillows": "cushion",
    "lamp": "light",
    "lights": "light",
}

CANONICAL_ATTRIBUTE_MAP = {
    "luxury": "luxurious",
    "plush": "tufted",
    "metallic wall": "metallic",
    "curved couch": "curved seating",
    "indoor lounge": "lounge-like",
    "cinema": "entertainment-like",
}

DEFAULT_AFFORDANCE_TO_OBJECTS = {
    "sit": {"chair", "sofa", "bench", "stool", "seat"},
    "lie": {"sofa", "bed", "bench"},
    "watch_tv": {"screen", "tv", "monitor", "sofa", "chair"},
    "sleep": {"bed", "sofa"},
    "look_outside": {"window"},
    "store_items": {"cabinet", "shelf", "drawer"},
    "open": {"door", "window", "drawer", "cabinet"},
}


def encode_image_data_url(image_path: Path) -> str:
    suffix = image_path.suffix.lower()
    if suffix == ".png":
        mime = "image/png"
    elif suffix in {".jpg", ".jpeg"}:
        mime = "image/jpeg"
    elif suffix == ".webp":
        mime = "image/webp"
    else:
        raise ValueError(f"Unsupported image type: {image_path}")

    data = image_path.read_bytes()
    b64 = base64.b64encode(data).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def extract_json_from_text(text: str) -> Dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise ValueError(f"Model did not return valid JSON:\n{text}")
        return json.loads(match.group(0))


def normalize_token(token: str, mapping: Dict[str, str]) -> str:
    t = token.strip().lower()
    return mapping.get(t, t)


def normalize_object_entry(obj: Dict[str, Any]) -> Dict[str, Any]:
    name = normalize_token(str(obj.get("name", "")), CANONICAL_OBJECT_MAP)
    count = obj.get("count", 1)
    if not isinstance(count, int) or count < 1:
        count = 1

    attributes = obj.get("attributes", [])
    if not isinstance(attributes, list):
        attributes = []
    attributes = sorted({
        normalize_token(str(a), CANONICAL_ATTRIBUTE_MAP)
        for a in attributes
        if str(a).strip()
    })

    affordances = obj.get("affordances", [])
    if not isinstance(affordances, list):
        affordances = []
    affordances = sorted({
        str(a).strip().lower()
        for a in affordances
        if str(a).strip()
    })

    return {
        "name": name,
        "count": count,
        "attributes": attributes,
        "affordances": affordances,
    }


def enrich_affordances(objects: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    for obj in objects:
        obj_name = obj["name"]
        affs = set(obj.get("affordances", []))
        for aff, valid_objects in DEFAULT_AFFORDANCE_TO_OBJECTS.items():
            if obj_name in valid_objects:
                affs.add(aff)
        obj["affordances"] = sorted(affs)
    return objects


def normalize_record(raw: Dict[str, Any], image_index: int, filename: str) -> Dict[str, Any]:
    summary = str(raw.get("summary", "")).strip()

    objects = raw.get("objects", [])
    if not isinstance(objects, list):
        objects = []
    objects = [normalize_object_entry(o) for o in objects if isinstance(o, dict)]
    objects = [o for o in objects if o["name"]]
    objects = enrich_affordances(objects)

    scene_attributes = raw.get("scene_attributes", [])
    if not isinstance(scene_attributes, list):
        scene_attributes = []
    scene_attributes = sorted({
        normalize_token(str(a), CANONICAL_ATTRIBUTE_MAP)
        for a in scene_attributes
        if str(a).strip()
    })

    return {
        "index": image_index,
        "filename": filename,
        "summary": summary,
        "objects": objects,
        "scene_attributes": scene_attributes,
    }


def call_groq_vision(
    api_key: str,
    model: str,
    image_path: Path,
    timeout: int = 90,
) -> Dict[str, Any]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    image_data_url = encode_image_data_url(image_path)

    payload = {
        "model": model,
        "temperature": 1e-8,  # Groq maps 0 to a tiny positive value; use explicit small float.
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image using the required JSON schema."},
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                ],
            },
        ],
    }

    response = requests.post(GROQ_URL, headers=headers, json=payload, timeout=timeout)

    if not response.ok:
        raise RuntimeError(
            f"Groq error {response.status_code}\n"
            f"Response body:\n{response.text}\n"
            f"Request model: {model}\n"
            f"Image: {image_path.name}"
        )

    data = response.json()

    try:
        raw_text = data["choices"][0]["message"]["content"]
    except Exception as exc:
        raise RuntimeError(f"Unexpected Groq response format:\n{json.dumps(data, indent=2)}") from exc

    return extract_json_from_text(raw_text)


def build_dataset(
    image_dir: Path,
    api_key: str,
    model: str,
) -> Dict[str, Any]:
    image_paths = sorted([p for p in image_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}])
    if not image_paths:
        raise FileNotFoundError(f"No supported images found in: {image_dir}")

    records = []
    for idx, image_path in enumerate(image_paths, start=1):
        print(f"[INFO] Processing {idx}/{len(image_paths)}: {image_path.name}")
        raw = call_groq_vision(
            api_key=api_key,
            model=model,
            image_path=image_path,
        )
        rec = normalize_record(raw, image_index=idx, filename=image_path.name)
        records.append(rec)

    return {
        "dataset_dir": str(image_dir),
        "num_images": len(records),
        "model": model,
        "images": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build JSON metadata from images using Groq vision.")
    parser.add_argument("--image_dir", required=True, help="Directory containing images.")
    parser.add_argument("--output_json", required=True, help="Output JSON file.")
    parser.add_argument(
        "--model",
        default="meta-llama/llama-4-scout-17b-16e-instruct",
        help="Groq vision-capable model identifier.",
    )
    args = parser.parse_args()

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError("GROQ_API_KEY is not set.")

    dataset = build_dataset(
        image_dir=Path(args.image_dir),
        api_key=api_key,
        model=args.model,
    )

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(dataset, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[OK] Saved dataset JSON to: {output_path}")


if __name__ == "__main__":
    main()