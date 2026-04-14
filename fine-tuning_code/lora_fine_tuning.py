import json
import os

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

torch.cuda.empty_cache()

token = None  # Set to None for auto-detection, or specify a token if needed

MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
QA_JSON_PATH = "../output_v2"

OUTPUT_DIR = "../models_fine_tuned/llama-lora-v1"
FINAL_DIR = "../models_fine_tuned/llama-final-v1"

MAX_SEQ_LEN = 1024
TRAIN_BATCH_SIZE = 1
EVAL_BATCH_SIZE = 1
GRAD_ACCUMULATION = 16

LEARNING_RATE = 2e-4
NUM_EPOCHS = 3
WARMUP_RATIO = 0.05

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

SYSTEM_PROMPT = """
You are a veterinary drug reference assistant trained exclusively on Plumb's Veterinary Drug Handbook.

## Core Rules:
(1) Only answer questions covered in your training data. If unsure, say: "I don't have reliable information on this. Please consult the full Plumb's handbook or a licensed veterinarian."
(2) For dosage questions, ALWAYS state: species | route | dose range | frequency. Never guess dosages.
(3) For safety-critical information, begin with WARNING: and explain clinical significance.
(4) Flag dangerous drug combinations and contraindications explicitly.
(5) Specify which species each answer applies to (e.g., "For cattle..." or "This is contraindicated in cats").
(6) If a question asks about a species not in your training data, provide a refusal with appropriate caution.

## Quality Standards:
- Be explanatory: Don't just state facts—briefly explain WHY they matter clinically.
- Ground all claims strictly in the reference material. Never extrapolate or invent.
- Use plain English. Avoid Unicode characters (use 'mcg' not 'μg', 'degrees F' not '°F').
- For drug interactions: provide specific mechanism and clinical significance, not generic warnings.
- For dosages: include duration, frequency, route, and any titration notes.
- For formulations: distinguish between USP, NF, and common forms (e.g., "Acetic acid 36-37% USP vs 3-5% vinegar").

## End Every Clinical Answer With:
Disclaimer: Always verify dosing and applicability with a licensed veterinarian before administering.

## Tone:
Professional, precise, safety-focused. Prioritize accuracy over completeness.
"""


# ---------------- DATASET ---------------- #


def load_and_expand_dataset(path):
    print("📂 Loading dataset...")

    samples = []

    for filename in os.listdir(path):
        file_path = os.path.join(path, filename)

        if (
            filename.startswith("stats")
            or filename.startswith("dataset")
            or not filename.endswith(".json")
        ):
            continue
        if filename.endswith(".json"):
            with open(file_path, "r") as f:
                data = json.load(f)

            print(
                f" Processing {filename} with {len(data.get('qa_pairs', []))} QA pairs..."
            )
            if not data:
                print(f"⚠️ Warning: {filename} is empty. Skipping.")
                continue

            for drug in data["qa_pairs"]:
                question = drug["question"].strip()
                answer = drug["answer"].strip()
                paraphrases = drug.get("paraphrases", [])

                # Ensure paraphrases is always a list
                if not isinstance(paraphrases, list):
                    paraphrases = []

                # 🔥 Expand here
                all_questions = [question] + paraphrases

                for q in all_questions:
                    if q and q.strip():  # avoid empty strings
                        samples.append({"question": q.strip(), "answer": answer})

    print(f"✅ Dataset loaded & expanded: {len(samples)} samples")

    return Dataset.from_list(samples)


# ----------------- Tokenization ----------------- #


def tokenize_dataset(dataset, tokenizer):
    def tokenize(example):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["question"]},
        ]
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        full_text = prompt_text + example["answer"] + tokenizer.eos_token
        prompt_length = len(
            tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        )

        tokenized = tokenizer(
            full_text, max_length=MAX_SEQ_LEN, truncation=True, padding="max_length"
        )

        labels = [-100] * prompt_length + tokenized["input_ids"][prompt_length:]
        labels = labels[:MAX_SEQ_LEN]
        if len(labels) < MAX_SEQ_LEN:
            labels += [-100] * (MAX_SEQ_LEN - len(labels))

        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "labels": labels,
        }

    return dataset.map(
        tokenize,
        remove_columns=dataset.column_names,
        batched=False,
    )


# ---------------- LORA ---------------- #


def apply_lora(model):

    print("🔧 Applying LoRA")

    config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "up_proj",
            "down_proj",
            "gate_proj",
        ],  # Llama modules
        inference_mode=False,
    )

    model = get_peft_model(model, config)

    model.print_trainable_parameters()

    return model


# ---------------- TRAINING ---------------- #


def train(model, tokenizer, train_dataset, eval_dataset):

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        warmup_ratio=WARMUP_RATIO,
        lr_scheduler_type="cosine",  # Better convergence for LLMs
        weight_decay=0.01,
        max_grad_norm=1.0,  # Prevent gradient explosion
        optim="paged_adamw_32bit",  # Memory-efficient optimizer
        gradient_checkpointing=True,  # Reduce VRAM usage during training
        fp16=True,
        bf16=False,
        logging_steps=10,
        eval_steps=100,
        eval_strategy="steps",
        save_steps=100,
        save_total_limit=5,
        dataloader_pin_memory=True,  # GPU efficiency
        seed=42,  # Reproducibility
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    # Data collator handles padding and batching
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal LM, not masked LM
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=10,
                early_stopping_threshold=0.001,
            )
        ],
    )

    if os.path.isdir(OUTPUT_DIR) and any(
        "checkpoint" in d for d in os.listdir(OUTPUT_DIR)
    ):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    return trainer


# ---------------- SAVE ---------------- #


def save_model(trainer, tokenizer):

    print("💾 Saving LoRA adapters...")
    trainer.model.save_pretrained(OUTPUT_DIR + "_lora")
    tokenizer.save_pretrained(OUTPUT_DIR + "_lora")

    print("🔄 Reloading base model for clean merge...")

    # Reload base model
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
        token=token,
    )

    # Load LoRA into base model
    model_with_lora = PeftModel.from_pretrained(base_model, OUTPUT_DIR + "_lora")

    print("🔗 Merging LoRA into base model...")
    merged_model = model_with_lora.merge_and_unload()

    print("💾 Saving merged full model...")
    merged_model.save_pretrained(FINAL_DIR)
    tokenizer.save_pretrained(FINAL_DIR)

    print("✅ LoRA adapters and merged model saved successfully!")


# ---------------- TEST ---------------- #


def test_model(model, tokenizer):

    question = "What is the side effect of alprazolam in cats?"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=512,
            num_beams=4,
            repetition_penalty=1.2,
        )

    answer = tokenizer.decode(
        output[0][inputs["input_ids"].shape[-1] :],
    )

    print("\nQuestion:", question)
    print("\nAnswer:", answer)


# ---------------- MAIN ---------------- #


def main():

    print("🚀 Llama Veterinary Drug LoRA Training")

    if not torch.cuda.is_available():
        print("❌ GPU required")
        return

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=token)

    tokenizer.pad_token = tokenizer.eos_token

    print("📥 Loading model...")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float16,
        device_map="auto",
        token=token,
    )

    model.config.use_cache = False

    dataset = load_and_expand_dataset(QA_JSON_PATH)

    split = dataset.train_test_split(test_size=0.1)

    train_dataset = tokenize_dataset(split["train"], tokenizer)

    eval_dataset = tokenize_dataset(split["test"], tokenizer)

    model = apply_lora(model)

    trainer = train(model, tokenizer, train_dataset, eval_dataset)

    save_model(trainer, tokenizer)

    test_model(model, tokenizer)


if __name__ == "__main__":
    main()
