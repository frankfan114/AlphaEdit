import json
import re


# =========================================================
#  Regex patterns
# =========================================================

# What / Where ... of X
WH_OF_PAT = re.compile(
    r"(?:What|Where)\s+(?:is|was)\s+.*?\s+of\s+([A-Z][A-Za-z0-9 .&'’\-\(\)]+)",
    re.UNICODE,
)

# of / for / by / in / from X
OF_PAT = re.compile(
    r"(?:of|for|by|in|from)\s+([A-Z][A-Za-z0-9 .&'’\-\(\)]+)",
    re.UNICODE,
)

# Leading entity (sentence starts with entity)
LEADING_ENTITY_PAT = re.compile(
    r"^([A-Z][A-Za-z0-9 .&'’\-\(\)]+?)"
    r"(?:\s+is|\s+was|\s+plays|\s+spoke|\s*,|\s*$)",
    re.UNICODE,
)

# X, something
COMMA_ENTITY_PAT = re.compile(
    r"^([A-Z][A-Za-z0-9 .&'’\-\(\)]+?),",
    re.UNICODE,
)

# Copula cleanup (CRITICAL FIX)
COPULA_CLEAN_PAT = re.compile(
    r"\b(is|was|are|were)\b.*$",
    re.IGNORECASE,
)


# =========================================================
#  Subject extraction utilities
# =========================================================

def clean_subject(subject: str) -> str:
    """
    Remove trailing copula verbs (is/was/are/were) and anything after them.
    """
    subject = subject.strip()
    subject = COPULA_CLEAN_PAT.sub("", subject)
    return subject.strip()


def extract_subject(prompt: str):
    """
    Extract semantic subject entity from a neighborhood prompt.
    Returns None if extraction is unreliable.
    """
    prompt = prompt.strip()

    # 1. What / Where ... of X
    m = WH_OF_PAT.search(prompt)
    if m:
        return clean_subject(m.group(1))

    # 2. of / for / by / in X
    m = OF_PAT.search(prompt)
    if m:
        return clean_subject(m.group(1))

    # 3. Leading entity
    m = LEADING_ENTITY_PAT.match(prompt)
    if m:
        return clean_subject(m.group(1))

    # 4. Entity before comma
    m = COMMA_ENTITY_PAT.match(prompt)
    if m:
        return clean_subject(m.group(1))

    return None


def make_template(prompt: str, subject: str):
    """
    Replace subject with {} exactly once.
    """
    esc = re.escape(subject)
    return re.sub(esc, "{}", prompt, count=1)


# =========================================================
#  Main conversion
# =========================================================

def convert_counterfact_neighborhood(input_path: str, output_path: str):
    with open(input_path, "r", encoding="utf-8") as f:
        cases = json.load(f)

    output = []

    for case in cases[:200]:  # Limit to first 200 cases
        case_id = case["case_id"]
        rr = case["requested_rewrite"]

        relation_id = rr["relation_id"]
        target_true = rr["target_true"]
        target_new = rr["target_new"]

        for idx, prompt in enumerate(case.get("neighborhood_prompts", [])):
            subject = extract_subject(prompt)
            if subject is None:
                continue

            template = make_template(prompt, subject)
            if "{}" not in template:
                continue

            # ID rule: concatenate case_id and neighborhood index
            known_id = int(f"{case_id}{idx}")

            output.append({
                "known_id": known_id,
                "case_id": case_id,
                "neighbor_index": idx,
                "subject": subject,
                "prompt": template,
                "relation_id": relation_id,
                "target_true": target_true,
                "target_new": target_new,
                "source": "neighborhood",
            })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"[OK] Wrote {len(output)} neighborhood probes → {output_path}")


# =========================================================
#  Entry
# =========================================================

if __name__ == "__main__":
    convert_counterfact_neighborhood(
        input_path="/rds/general/user/ff422/home/FYP/AlphaEdit/data/counterfact.json",
        output_path="/rds/general/user/ff422/home/FYP/AlphaEdit/data/counterfact_neighbor.json",
    )
