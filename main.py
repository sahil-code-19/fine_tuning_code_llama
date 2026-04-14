"""
Plumb's Veterinary Drug Handbook — Tiered QA Generator
=======================================================
Workflow:
  1. Feed raw drug text (from book extraction)
  2. Auto-classify drug into Tier 1–4
  3. Generate section-wise QA pairs scaled to that tier
  4. Augment every question with 5 paraphrase variants
  5. Export final JSONL ready for fine-tuning

Requirements:
    pip install anthropic tqdm
    export ANTHROPIC_API_KEY=your_key_here
"""

import json
import os
import re
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Optional

# import anthropic
import dotenv
import ollama
from tqdm import tqdm

dotenv.load_dotenv()

print("ANTHROPIC_API_KEY========================>", os.getenv("ANTHROPIC_API_KEY"))

# client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
MODEL = "gemma3:12b"

# ─────────────────────────────────────────────────────────────
# TIER DEFINITIONS
# ─────────────────────────────────────────────────────────────


@dataclass
class TierConfig:
    tier: int
    name: str
    description: str
    # How many QA pairs to generate per section
    section_counts: dict
    # How many paraphrase variants per question
    paraphrase_count: int


TIERS = {
    1: TierConfig(
        tier=1,
        name="Simple",
        description="Single species, 1-2 indications, few interactions",
        section_counts={
            "prescriber_highlights": 2,
            "uses_indications": 2,
            "pharmacology": 1,
            "pharmacokinetics": 1,
            "contraindications": 2,
            "adverse_effects": 2,
            "drug_interactions": 2,
            "dosages": 3,
            "overdose_toxicity": 1,
            "storage_forms": 1,
            "client_information": 1,
            "refusals": 2,
        },
        paraphrase_count=4,
    ),
    2: TierConfig(
        tier=2,
        name="Standard",
        description="Multi-species, several indications, moderate interactions",
        section_counts={
            "prescriber_highlights": 3,
            "uses_indications": 3,
            "pharmacology": 2,
            "pharmacokinetics": 2,
            "contraindications": 3,
            "adverse_effects": 4,
            "drug_interactions": 4,
            "dosages": 5,
            "overdose_toxicity": 2,
            "reproductive_safety": 1,
            "laboratory_considerations": 1,
            "storage_forms": 2,
            "client_information": 1,
            "refusals": 3,
        },
        paraphrase_count=5,
    ),
    3: TierConfig(
        tier=3,
        name="Complex",
        description="Many species, many indications, many interactions",
        section_counts={
            "prescriber_highlights": 4,
            "uses_indications": 5,
            "pharmacology": 3,
            "pharmacokinetics": 3,
            "contraindications": 5,
            "adverse_effects": 6,
            "drug_interactions": 7,
            "dosages": 8,
            "overdose_toxicity": 3,
            "reproductive_safety": 2,
            "laboratory_considerations": 2,
            "storage_forms": 2,
            "client_information": 2,
            "refusals": 4,
        },
        paraphrase_count=5,
    ),
    4: TierConfig(
        tier=4,
        name="High-Risk",
        description="Controlled/chemo/narrow therapeutic index drugs",
        section_counts={
            "prescriber_highlights": 5,
            "uses_indications": 5,
            "pharmacology": 4,
            "pharmacokinetics": 4,
            "contraindications": 7,
            "adverse_effects": 8,
            "drug_interactions": 10,
            "dosages": 10,
            "overdose_toxicity": 5,
            "reproductive_safety": 3,
            "laboratory_considerations": 4,
            "storage_forms": 2,
            "client_information": 3,
            "refusals": 5,
        },
        paraphrase_count=6,
    ),
}

# ─────────────────────────────────────────────────────────────
# DATA MODELS
# ─────────────────────────────────────────────────────────────


@dataclass
class QAPair:
    drug: str
    tier: int
    section: str
    question_type: str
    question: str
    answer: str
    confidence: str  # high / medium / low
    species: Optional[str]
    safety_flag: bool
    refusal: bool
    paraphrases: list[str] = field(default_factory=list)

    def to_training_records(self) -> list[dict]:
        """
        Expands one QAPair into N training records:
        - 1 original question
        - N paraphrase variants
        All share the same answer and system prompt.
        """
        system = (
            "You are a veterinary drug reference assistant trained exclusively on "
            "Plumb's Veterinary Drug Handbook. "
            "Rules: "
            "(1) Only answer questions covered in your training data. "
            "(2) For dosage questions always state: species, route, dose range, frequency. "
            "(3) If unsure, say: 'I don't have reliable information on this. "
            "Please consult the full Plumb's handbook or a licensed veterinarian.' "
            "(4) Never guess dosages. "
            "(5) Flag dangerous drug combinations with WARNING:. "
            "(6) End every clinical answer with: "
            "Disclaimer: Always verify with a licensed veterinarian."
        )
        records = []
        for q in [self.question] + self.paraphrases:
            records.append(
                {
                    "system": system,
                    "instruction": q,
                    "input": "",
                    "output": self.answer,
                    "metadata": {
                        "drug": self.drug,
                        "tier": self.tier,
                        "section": self.section,
                        "question_type": self.question_type,
                        "confidence": self.confidence,
                        "species": self.species,
                        "safety_flag": self.safety_flag,
                        "refusal": self.refusal,
                        "is_paraphrase": q != self.question,
                    },
                }
            )
        return records


# ─────────────────────────────────────────────────────────────
# STEP 1 — TIER CLASSIFIER
# ─────────────────────────────────────────────────────────────

TIER_CLASSIFIER_PROMPT = """
You are classifying a veterinary drug for fine-tuning dataset generation.

Read the drug text below and classify it into exactly one tier:

TIER 1 — Simple:
- Covers 1-2 species
- 1-2 clinical indications
- Few drug interactions (≤3)
- No controlled substance / narrow therapeutic index

TIER 2 — Standard:
- Covers 2-3 species  
- 2-4 clinical indications
- Moderate interactions (4-6)
- No major safety concerns beyond standard precautions

TIER 3 — Complex:
- Covers 3+ species
- 4+ clinical indications OR widely used drug
- Many interactions (7+)
- Multiple dosing routes or complex dosing protocols

TIER 4 — High-Risk:
- Controlled substance OR chemotherapy agent OR narrow therapeutic index
- Requires close monitoring (TDM, frequent labs)
- Severe or life-threatening adverse effects possible
- Many serious drug interactions

Drug text:
{drug_text}

Respond with ONLY a JSON object, no other text:
{{
  "tier": <1, 2, 3, or 4>,
  "reason": "<one sentence explanation>",
  "species_covered": ["list", "of", "species"],
  "indication_count": <number>,
  "interaction_count": <number>,
  "is_controlled": <true/false>,
  "is_narrow_therapeutic_index": <true/false>
}}
"""


def classify_drug_tier(drug_name: str, drug_text: str) -> dict:
    """Calls Claude to classify the drug into a tier."""
    print(f"  [Tier classifier] Classifying {drug_name}...")
    # response = client.messages.create(
    #     model=MODEL,
    #     max_tokens=300,
    #     messages=[
    #         {
    #             "role": "user",
    #             "content": TIER_CLASSIFIER_PROMPT.format(drug_text=drug_text[:3000]),
    #         }
    #     ],
    # )

    response = ollama.chat(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": TIER_CLASSIFIER_PROMPT.format(drug_text=drug_text[:3000]),
            }
        ],
    )
    raw = response.message.content.strip()
    # Strip markdown fences if present
    raw = re.sub(r"^```json\s*|^```\s*|\s*```$", "", raw, flags=re.MULTILINE).strip()
    result = json.loads(raw)
    result["drug"] = drug_name
    return result


# ─────────────────────────────────────────────────────────────
# STEP 2 — QA PAIR GENERATOR
# ─────────────────────────────────────────────────────────────

QA_GENERATION_PROMPT = """
You are building a high-quality fine-tuning dataset from Plumb's Veterinary Drug Handbook.

## Context
Drug: {drug_name}
Tier: {tier} ({tier_name}) — {tier_description}
Section: {section}
QA pairs needed: {count}

## Drug Text
{drug_text}

---

## CRITICAL: REDUNDANCY & COMPREHENSIVENESS RULES

### ⛔ DO NOT GENERATE REDUNDANT QUESTIONS
Avoid asking the same core content twice, even with different wording. Examples of redundancy to PREVENT:
- Multiple "What is the primary indication?" questions (ask ONCE, then move to different angles)
- Repeating the same mechanism explanation (ask it once, then focus on practical implications)
- Duplicating concentration/safety warnings (emphasize once as CRITICAL, not repeatedly)

### ✅ DO ENSURE COMPREHENSIVE COVERAGE within this section
For each section assigned, generate DIVERSE question angles:
- Indications/Uses: Ask about primary use (1 pair), then secondary uses (1 pair), then practical scenarios (others)
- Dosages: Ask cattle dose separately from horse dose separately from sheep dose (NOT all in one Q)
- Drug Interactions: Ask about SPECIFIC drugs/classes if mentioned in text (not generic)
- Chemistry/Storage: Include formulation percentages, storage conditions, stability if in text
- Pharmacology: Ask mechanism (1 pair), then clinical implications (1 pair), then monitoring (1 pair)

---

## Question Type Definitions
Generate a MIX of these types:
- factual          → Direct recall: "What is the mechanism of action of X?"
- clinical_scenario → Reasoning required: "A dog with renal failure needs pain management — is X appropriate?"
- species_specific  → ONE species per question. Ask separately for dog, cat, horse, etc.
- monitoring       → "What adverse effects or lab values should be monitored?"
- comparison       → "Why choose X over Y for Z condition?"
- practical        → Administration route, storage, compounding, handling, product sourcing
- client_education → Plain language a pet owner would understand (no jargon)
- refusal          → Questions that CANNOT be answered from the text (wrong species, off-label not covered, out of scope)

---

## Answer Writing Rules

### For ALL answers:
- Write in complete, grammatically correct sentences.
- Be EXPLANATORY: don't just state a fact — briefly explain WHY it matters clinically.
  BAD:  "The dose is 5 mg/kg."
  GOOD: "The recommended dose is 5 mg/kg because lower doses may be subtherapeutic, while higher doses increase the risk of hepatotoxicity."
- Ground every claim STRICTLY in the drug text above. Never invent or extrapolate.
- Use plain, consistent English. Avoid Unicode special characters (use 'mcg' not 'μg', 'degrees F' not '°F').
- Double-check drug and product names for spelling accuracy (e.g., "Acetic acid" not "Acarbose")

### For dosage answers:
ALWAYS include ALL of: species | route | dose range | frequency | any relevant duration or titration notes.
If multiple species exist in text, create SEPARATE questions for each species rather than one combined question.

### For drug interaction answers:
IfReferences mentions specific drug interactions, ask about EACH one separately (not nested).
Example from text: "Can interact with Aspirin, Azole antifungals, Iron, Phenobarbital, Quinidine"
→ Generate ~5 pairs, one asking about Aspirin, one about Azole antifungals, etc. (NOT one pair covering all)

### For chemistry/formulation answers:
If text mentions formulation details (concentrations, USP/NF specs, dilution instructions), include them.
Example: "Acetic Acid USP is 36-37%, Diluted NF is 5.7-6.3%, Vinegar is 3-5%"
→ Generate pairs asking about each formulation type

### For storage/regulatory answers:
If text mentions storage conditions, regulatory status, or product availability, generate practical Q&A.
Example: "Store in airtight containers" or "No veterinary-labeled products available"

### For safety-critical answers:
Begin the answer with: WARNING:

### For refusal questions:
The answer MUST be exactly:
"I don't have reliable information on this in the available drug reference. Please consult a licensed veterinarian or a current pharmacology resource."

### Closing line:
End every non-refusal clinical answer with:
"Disclaimer: Always verify dosing and applicability with a licensed veterinarian before administering."

---

## JSON Field Constraints — follow these EXACTLY

- "species": must be ONE of these exact strings only:
    "dog" | "cat" | "horse" | "bird" | "rabbit" | "small_mammal" | "general" | null
  → If a question covers multiple species, SPLIT into separate QA objects (one per species).
  → Use null only for practical/storage/compounding questions with no species context.

- "question_type": must be ONE of:
    "factual" | "clinical_scenario" | "species_specific" | "monitoring" |
    "comparison" | "practical" | "client_education" | "refusal"

- "confidence": must be ONE of: "high" | "medium" | "low"
  → Use "low" if the drug text only partially supports the answer.
  → Use "medium" for theoretical interactions or when data is incomplete.

- "safety_flag": boolean — true if the answer involves toxicity, contraindications, overdose, narrow therapeutic index, or any WARNING.

- "refusal": boolean — true only for refusal-type questions.

---

## SPECIAL EMPHASIS FOR THIS SECTION: {section}

For "{section}" section specifically:
- Identify ALL factual claims, dosages, interactions, or safety notes in the drug text
- Generate questions that cover each of these comprehensively
- Avoid asking the same question twice
- Ensure each pair brings NEW clinical or practical information

---

## Output Format
Respond with ONLY a valid JSON array. No preamble, no markdown fences, no trailing text.
Ensure all strings use ASCII characters only (no Unicode escapes, no special symbols).

[
  {{
    "question": "<the question>",
    "answer": "<full explanatory answer>",
    "question_type": "<one of the allowed types>",
    "confidence": "<high|medium|low>",
    "species": "<one allowed species string or null>",
    "safety_flag": <true|false>,
    "refusal": <true|false>
  }}
]
"""


def generate_qa_pairs(
    drug_name: str,
    drug_text: str,
    tier_config: TierConfig,
    section: str,
    count: int,
) -> list[dict]:
    """Generates QA pairs for one section of one drug."""
    if count == 0:
        return []

    # response = client.messages.create(
    #     model=MODEL,
    #     max_tokens=4000,
    #     messages=[{
    #         "role": "user",
    #         "content": QA_GENERATION_PROMPT.format(
    #             drug_name=drug_name,
    #             tier=tier_config.tier,
    #             tier_name=tier_config.name,
    #             tier_description=tier_config.description,
    #             drug_text=drug_text[:4000],
    #             section=section.replace("_", " ").title(),
    #             count=count,
    #         )
    #     }]
    # )

    response = ollama.chat(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": QA_GENERATION_PROMPT.format(
                    drug_name=drug_name,
                    tier=tier_config.tier,
                    tier_name=tier_config.name,
                    tier_description=tier_config.description,
                    drug_text=drug_text[:4000],
                    section=section.replace("_", " ").title(),
                    count=count,
                ),
            }
        ],
    )

    raw = response.message.content.strip()
    raw = re.sub(r"^```json\s*|^```\s*|\s*```$", "", raw, flags=re.MULTILINE).strip()
    pairs = json.loads(raw)
    return pairs[:count]  # Safety cap


# ─────────────────────────────────────────────────────────────
# STEP 3 — PARAPHRASE AUGMENTER
# ─────────────────────────────────────────────────────────────

PARAPHRASE_PROMPT = """
You are augmenting a veterinary fine-tuning dataset.

Original question about the drug "{drug_name}":
"{question}"

Generate exactly {count} alternative ways someone might ask this same question.
Vary the style realistically:
- Some formal (veterinarian style)
- Some casual (pet owner style)  
- Some very short ("acarbose with food?")
- Some with species embedded ("for my cat, when...")
- Some with slight misspellings or abbreviations
- Some phrased as a statement seeking confirmation

Critical rules:
- Do NOT change the meaning or introduce new clinical content
- Do NOT make the question unanswerable with the original answer
- Keep all questions relevant to the original intent

Respond with ONLY a JSON array of strings, no other text:
["variant 1", "variant 2", ...]
"""


def generate_paraphrases(
    drug_name: str,
    question: str,
    count: int,
) -> list[str]:
    """Generates paraphrase variants for a single question."""
    # response = client.messages.create(
    #     model=MODEL,
    #     max_tokens=800,
    #     messages=[
    #         {
    #             "role": "user",
    #             "content": PARAPHRASE_PROMPT.format(
    #                 drug_name=drug_name,
    #                 question=question,
    #                 count=count,
    #             ),
    #         }
    #     ],
    # )

    response = ollama.chat(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": PARAPHRASE_PROMPT.format(
                    drug_name=drug_name,
                    question=question,
                    count=count,
                ),
            }
        ],
    )

    # raw = response.content[0].text.strip()
    raw = response.message.content.strip()
    raw = re.sub(r"^```json\s*|^```\s*|\s*```$", "", raw, flags=re.MULTILINE).strip()
    variants = json.loads(raw)
    return variants[:count]


# ─────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────


def process_drug(drug_name: str, drug_text: str) -> dict:
    """
    Full pipeline for one drug:
      classify → generate QA per section → paraphrase augment → return records
    """
    print(f"\n{'=' * 60}")
    print(f"Processing: {drug_name}")
    print(f"{'=' * 60}")

    # ── Step 1: Classify tier ──────────────────────────────
    classification = classify_drug_tier(drug_name, drug_text)
    tier_num = classification["tier"]
    tier_config = TIERS[tier_num]
    print(f"  → Tier {tier_num} ({tier_config.name}): {classification['reason']}")

    all_qa_pairs: list[QAPair] = []

    # ── Step 2: Generate QA pairs section by section ──────
    sections = tier_config.section_counts
    for section, count in sections.items():
        print(f"  [QA Gen] {section} → {count} pairs...")
        try:
            raw_pairs = generate_qa_pairs(
                drug_name=drug_name,
                drug_text=drug_text,
                tier_config=tier_config,
                section=section,
                count=count,
            )
            for p in raw_pairs:
                all_qa_pairs.append(
                    QAPair(
                        drug=drug_name,
                        tier=tier_num,
                        section=section,
                        question_type=p.get("question_type", "factual"),
                        question=p["question"],
                        answer=p["answer"],
                        confidence=p.get("confidence", "high"),
                        species=p.get("species"),
                        safety_flag=p.get("safety_flag", False),
                        refusal=p.get("refusal", False),
                    )
                )
            time.sleep(0.5)  # Rate limit courtesy
        except Exception as e:
            print(f"    ⚠ Error in section {section}: {e}")
            continue

    # ── Step 3: Paraphrase augmentation ──────────────────
    print(
        f"  [Paraphrase] Augmenting {len(all_qa_pairs)} pairs "
        f"with {tier_config.paraphrase_count} variants each..."
    )

    for qa in tqdm(all_qa_pairs, desc="  Augmenting", leave=False):
        if qa.refusal:
            # Refusal questions need fewer variants — 2 is enough
            n = min(2, tier_config.paraphrase_count)
        else:
            n = tier_config.paraphrase_count
        try:
            qa.paraphrases = generate_paraphrases(
                drug_name=drug_name,
                question=qa.question,
                count=n,
            )
            time.sleep(0.3)
        except Exception as e:
            print(f"    ⚠ Paraphrase error: {e}")
            qa.paraphrases = []

    # ── Step 4: Expand to training records ───────────────
    training_records = []
    for qa in all_qa_pairs:
        training_records.extend(qa.to_training_records())

    result = {
        "drug": drug_name,
        "tier": tier_num,
        "tier_name": tier_config.name,
        "classification": classification,
        "qa_pair_count": len(all_qa_pairs),
        "training_record_count": len(training_records),
        "qa_pairs": [asdict(qa) for qa in all_qa_pairs],
        "training_records": training_records,
    }

    print(
        f"  ✓ {len(all_qa_pairs)} QA pairs → "
        f"{len(training_records)} training records (with paraphrases)"
    )
    return result


# ─────────────────────────────────────────────────────────────
# BATCH RUNNER
# ─────────────────────────────────────────────────────────────


def run_batch(
    drugs: list[dict],  # [{"name": "Acarbose", "text": "...full book text..."}]
    output_dir: str = "output",
) -> None:
    """
    Processes a list of drugs and writes:
      - output/<drug_name>.json       (full detail per drug)
      - output/dataset.jsonl          (flat training records, all drugs)
      - output/stats.json             (summary statistics)
    """
    os.makedirs(output_dir, exist_ok=True)
    all_training_records = []
    stats = {
        "total_drugs": len(drugs),
        "by_tier": {1: 0, 2: 0, 3: 0, 4: 0},
        "total_qa_pairs": 0,
        "total_training_records": 0,
        "safety_flagged": 0,
        "refusals": 0,
        "errors": [],
    }

    for drug in tqdm(drugs, desc="Processing drugs"):
        try:
            output_file_name = f"{drug['name'].lower().replace(' ', '_')}-{MODEL.replace(':', '_')}.json"

            if os.path.exists(os.path.join(output_dir, output_file_name)):
                print(f"  → Skipping {drug['name']} (already processed)")
                continue

            result = process_drug(drug["name"], drug["text"])

            # Per-drug JSON
            drug_path = os.path.join(
                output_dir,
                output_file_name,
            )
            with open(drug_path, "w") as f:
                json.dump(result, f, indent=2)

            # Accumulate
            all_training_records.extend(result["training_records"])
            stats["by_tier"][result["tier"]] += 1
            stats["total_qa_pairs"] += result["qa_pair_count"]
            stats["total_training_records"] += result["training_record_count"]
            stats["safety_flagged"] += sum(
                1 for qa in result["qa_pairs"] if qa["safety_flag"]
            )
            stats["refusals"] += sum(1 for qa in result["qa_pairs"] if qa["refusal"])

        except Exception as e:
            print(f"  ✗ Failed {drug['name']}: {e}")
            stats["errors"].append({"drug": drug["name"], "error": str(e)})

    # Write flat JSONL (one training record per line — standard fine-tune format)
    jsonl_path = os.path.join(output_dir, "dataset.jsonl")
    with open(jsonl_path, "w") as f:
        for record in all_training_records:
            f.write(json.dumps(record) + "\n")

    # Write stats
    stats_path = os.path.join(output_dir, "stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    # Print summary
    print(f"\n{'=' * 60}")
    print("BATCH COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Drugs processed    : {stats['total_drugs'] - len(stats['errors'])}")
    print(f"  Total QA pairs     : {stats['total_qa_pairs']}")
    print(
        f"  Training records   : {stats['total_training_records']} (incl. paraphrases)"
    )
    print(f"  Safety flagged     : {stats['safety_flagged']}")
    print(f"  Refusal examples   : {stats['refusals']}")
    print(f"  Errors             : {len(stats['errors'])}")
    print(f"\n  By tier:")
    for t, count in stats["by_tier"].items():
        print(f"    Tier {t} ({TIERS[t].name:<10}): {count} drugs")
    print(f"\n  Output: {jsonl_path}")


# ─────────────────────────────────────────────────────────────
# JSON LOADER  — reads plumbs_details.json
# ─────────────────────────────────────────────────────────────

PLUMBS_JSON = os.path.join(os.path.dirname(__file__), "plumbs_details.json")


def load_drugs_from_json(json_path: str) -> list[dict]:
    """
    Reads plumbs_details.json and returns a list of
    {"name": <title>, "text": <all sections concatenated>} dicts.
    Only entries with showMonograph=True are included.
    Extracts exactly 80 drugs (or fewer if fewer valid drugs exist).
    """
    with open(json_path, "r", encoding="utf-8") as f:
        entries = json.load(f)

    # Filter for showMonograph=True and valid titles FIRST
    valid_entries = []
    for entry in entries:
        if not entry.get("showMonograph", False):
            continue
        name = entry.get("title", "").strip()
        if not name:
            continue
        valid_entries.append(entry)

    # Then take the first 80 valid entries
    new_entries = valid_entries[:80]

    length = len(json.dumps(new_entries))
    print(f"Loaded {len(new_entries)} drugs from JSON (total {length} chars)")
    print(
        f"  (Filtered from {len(entries)} total entries, {len(valid_entries)} valid entries)"
    )

    drugs = []
    for entry in new_entries:
        name = entry.get("title", "").strip().replace("/", "-")

        # Build a readable text blob from all sections
        parts = [f"{name}"]
        drug_class = entry.get("drugClass", [])
        if drug_class:
            parts.append(f"Drug Class: {', '.join(drug_class)}")
        species = entry.get("species", [])
        if species:
            parts.append(f"Species: {', '.join(species)}")
        commercial = entry.get("commercialNames", [])
        if commercial:
            parts.append(f"Commercial Names: {', '.join(commercial)}")

        for section in entry.get("sections", []):
            title = section.get("title", "").strip()
            value = section.get("value", "").strip()
            if title and value:
                parts.append(f"\n{title}:\n{value}")

        print(f"  Built text for {name}.\n")
        drugs.append({"name": name, "text": "\n".join(parts)})

    return drugs


def find_drug(drugs: list[dict], name: str) -> dict | None:
    """Case-insensitive search for a drug by name (or alias prefix)."""
    name_lower = name.strip().lower()
    for d in drugs:
        if d["name"].lower() == name_lower:
            return d
    # Partial / prefix match as fallback
    for d in drugs:
        if d["name"].lower().startswith(name_lower):
            return d
    return None


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    os.makedirs("output", exist_ok=True)

    # Load all drugs from plumbs_details.json
    print(f"Loading drugs from {PLUMBS_JSON} …")
    all_drugs = load_drugs_from_json(PLUMBS_JSON)
    print(f"  Loaded {len(all_drugs)} monographs.\n")

    # ── Single-drug mode: python main.py "Acarbose" ────────
    if len(sys.argv) >= 2:
        drug_name_arg = " ".join(sys.argv[1:])
        drug = find_drug(all_drugs, drug_name_arg)

        if drug is None:
            print(f"❌ Drug '{drug_name_arg}' not found in plumbs_details.json.")
            print("   Available drugs (first 20):")
            for d in all_drugs[:20]:
                print(f"     • {d['name']}")
            sys.exit(1)

        assert drug is not None  # type narrowing for static checkers
        print(f"▶ Single-drug mode: {drug['name']}")
        result = process_drug(drug["name"], drug["text"])

        safe_name = drug["name"].lower().replace(" ", "_").replace("/", "_")
        full_path = os.path.join("output", f"{safe_name}_full.json")
        jsonl_path = os.path.join("output", f"{safe_name}_training.jsonl")

        with open(full_path, "w") as f:
            json.dump(result, f, indent=2)

        with open(jsonl_path, "w") as f:
            for record in result["training_records"]:
                f.write(json.dumps(record) + "\n")

        print(f"\n✅ Done!")
        print(f"   Drug          : {result['drug']}")
        print(f"   Tier          : {result['tier']} ({result['tier_name']})")
        print(f"   QA pairs      : {result['qa_pair_count']}")
        print(
            f"   Training recs : {result['training_record_count']} (with paraphrases)"
        )
        print(f"   Full JSON     : {full_path}")
        print(f"   Training JSONL: {jsonl_path}")

    # ── Batch mode: python main.py ─────────────────────────
    else:
        print("▶ Batch mode: processing all drugs from plumbs_details.json")
        run_batch(all_drugs, output_dir="output_v2")
