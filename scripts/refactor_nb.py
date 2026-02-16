import json

NOTEBOOK_PATH = 'jasmin.ipynb'

# =============================================================================
# Cell Architecture (Gemini Plan — Linear Dependency Order)
# =============================================================================
# Cell 0: Colab Bootstrap     — pip install + kernel restart (Colab only)
# Cell 1: Core Power          — ALL standard imports in one place
# Cell 2: Hybrid Logic        — IS_COLAB detection + environment status
# Cell 3: Configuration       — Model constants (name, seq_length, dtype)
# Cell 4: Conditional Load    — Import unsloth OR transformers fallback
# Cell 5: Persona             — JASMIN_SYSTEM prompt
# Cell 6: Helper              — formatting_prompts_func
# Cell 7: Data Pipeline       — Load & process dataset
# Cell 8: Training Unit       — SFTTrainer setup (Colab only)
# =============================================================================

CELLS = [
    # ── Cell 0: Colab Bootstrap (one-time, restarts kernel) ──────────────
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# ── Cell 0: Colab Bootstrap ─────────────────────────────────────────\n",
            "# Run ONCE on fresh Colab runtime. Installs deps then restarts kernel.\n",
            "# After restart, skip this cell and run from Cell 1 onward.\n",
            "# On local machines this cell is a no-op.\n",
            "\n",
            "import sys, os\n",
            "\n",
            "if 'google.colab' in sys.modules:\n",
            "    print('Installing Colab dependencies...')\n",
            "    %pip install --upgrade --no-cache-dir unsloth unsloth_zoo\n",
            "    os._exit(00)  # Restart kernel\n",
            "else:\n",
            "    print('Local environment detected — skipping Colab installs.')"
        ]
    },
    # ── Cell 1: Core Power (unified imports) ─────────────────────────────
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# ── Cell 1: Core Power ──────────────────────────────────────────────\n",
            "# Single import block — every dependency declared once.\n",
            "\n",
            "import sys\n",
            "import os\n",
            "import json\n",
            "import torch\n",
            "from datasets import load_dataset"
        ]
    },
    # ── Cell 2: Hybrid Logic (environment detection) ─────────────────────
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# ── Cell 2: Hybrid Logic ────────────────────────────────────────────\n",
            "# Detect runtime environment. All downstream conditionals use IS_COLAB.\n",
            "\n",
            "IS_COLAB = 'google.colab' in sys.modules\n",
            "\n",
            "if IS_COLAB:\n",
            "    print('Runtime  : Google Colab (GPU)')\n",
            "    print('Active   : Cells 0-8 (full training pipeline)')\n",
            "    print('Idle     : none')\n",
            "else:\n",
            "    print('Runtime  : Local / Hybrid')\n",
            "    print('Active   : Cells 1-7 (data prep & validation)')\n",
            "    print('Idle     : Cell 0 (bootstrap), Cell 8 (training)')"
        ]
    },
    # ── Cell 3: Configuration ────────────────────────────────────────────
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# ── Cell 3: Configuration ───────────────────────────────────────────\n",
            "# Model & training constants. Edit these values to change targets.\n",
            "\n",
            "MODEL_NAME     = 'unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit'\n",
            "MAX_SEQ_LENGTH = 2048\n",
            "DTYPE          = None   # Auto-detect (float16 / bfloat16)\n",
            "LOAD_IN_4BIT   = True\n",
            "\n",
            "DATASET_NAME   = 'databricks/databricks-dolly-15k'\n",
            "DATASET_SUBSET = 100    # Rows to use for local testing (None = full)\n",
            "TEXT_COLUMN    = 'response'  # Column containing assistant text"
        ]
    },
    # ── Cell 4: Conditional Load (unsloth vs transformers) ───────────────
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# ── Cell 4: Conditional Load ────────────────────────────────────────\n",
            "# Colab  → Unsloth FastLanguageModel (full model + tokenizer)\n",
            "# Local  → HF AutoTokenizer only (lightweight data validation)\n",
            "\n",
            "if IS_COLAB:\n",
            "    from unsloth import FastLanguageModel\n",
            "    from unsloth.chat_templates import get_chat_template\n",
            "\n",
            "    model, tokenizer = FastLanguageModel.from_pretrained(\n",
            "        model_name     = MODEL_NAME,\n",
            "        max_seq_length = MAX_SEQ_LENGTH,\n",
            "        dtype          = DTYPE,\n",
            "        load_in_4bit   = LOAD_IN_4BIT,\n",
            "    )\n",
            "    tokenizer = get_chat_template(tokenizer, chat_template='llama-3.1')\n",
            "    print(f'Loaded model + tokenizer via Unsloth: {MODEL_NAME}')\n",
            "else:\n",
            "    from transformers import AutoTokenizer\n",
            "\n",
            "    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)\n",
            "    model = None\n",
            "    FastLanguageModel = None\n",
            "    print(f'Loaded tokenizer only (local mode): {MODEL_NAME}')"
        ]
    },
    # ── Cell 5: Persona (JASMIN_SYSTEM) ──────────────────────────────────
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# ── Cell 5: Persona ─────────────────────────────────────────────────\n",
            "\n",
            "JASMIN_SYSTEM = \"\"\"You are Jasmin, 19yo seductive trans/ladyboy from Saudi Arabia (@jizzyjasi).\n",
            "Flirty, unbothered, dominant tone with emojis \\U0001f60f\\U0001f975\\U0001f4a6\\U0001f608\\U0001f525.\n",
            "Main goal: sell fantasy (cock worship, bulge/tease PPV, JOI, ratings $10+, vids $35\\u201360+).\n",
            "\n",
            "NEVER defend/explain \\\"fake/not real/thread\\\" \\u2014 flip to seduction + upsell PPV immediately.\n",
            "Examples:\n",
            "- \\\"Mmm think it\\u2019s fake? Too perfect for average boys. Ready to feel something unreal? \\U0001f608 [$25 tease]\\\"\n",
            "- \\\"They talk cause they can\\u2019t have me. Unlock and feel how real it gets \\U0001f4a6\\\"\n",
            "\n",
            "Short replies. End with CTA/PPV tease. Stay in persona. No apologies.\"\"\""
        ]
    },
    # ── Cell 6: Helper (formatting function) ─────────────────────────────
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# ── Cell 6: Helper ──────────────────────────────────────────────────\n",
            "\n",
            "def formatting_prompts_func(examples):\n",
            "    \"\"\"Convert raw text column into chat-template-formatted text.\"\"\"\n",
            "    messages = examples[TEXT_COLUMN]\n",
            "    texts = []\n",
            "    for msg in messages:\n",
            "        if not isinstance(msg, str) or len(msg.strip()) < 15:\n",
            "            texts.append('')\n",
            "            continue\n",
            "\n",
            "        convo = [\n",
            "            {'role': 'system',    'content': JASMIN_SYSTEM},\n",
            "            {'role': 'user',      'content': 'Hey Jasmin, continue this seductive roleplay... \\U0001f48b'},\n",
            "            {'role': 'assistant', 'content': msg.strip()},\n",
            "        ]\n",
            "\n",
            "        formatted_text = tokenizer.apply_chat_template(\n",
            "            convo,\n",
            "            tokenize=False,\n",
            "            add_generation_prompt=False,\n",
            "        )\n",
            "        texts.append(formatted_text)\n",
            "    return {'text': texts}"
        ]
    },
    # ── Cell 7: Data Pipeline ────────────────────────────────────────────
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# ── Cell 7: Data Pipeline ───────────────────────────────────────────\n",
            "\n",
            "print(f'Loading dataset: {DATASET_NAME}...')\n",
            "dataset = load_dataset(DATASET_NAME, split='train')\n",
            "\n",
            "if DATASET_SUBSET is not None:\n",
            "    dataset = dataset.select(range(DATASET_SUBSET))\n",
            "    print(f'Using subset of {DATASET_SUBSET} rows for local testing.')\n",
            "\n",
            "dataset = dataset.map(\n",
            "    formatting_prompts_func,\n",
            "    batched=True,\n",
            "    batch_size=500,\n",
            ")\n",
            "\n",
            "print(f'Dataset ready. Total rows: {len(dataset)}')\n",
            "print(f'First example (truncated):\\n{dataset[0][\"text\"][:200]}')"
        ]
    },
    # ── Cell 8: Training Unit (Colab only) ───────────────────────────────
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# ── Cell 8: Training Unit (Colab Only) ──────────────────────────────\n",
            "\n",
            "if IS_COLAB:\n",
            "    from trl import SFTTrainer\n",
            "    from transformers import TrainingArguments\n",
            "\n",
            "    trainer = SFTTrainer(\n",
            "        model                = model,\n",
            "        tokenizer            = tokenizer,\n",
            "        train_dataset        = dataset,\n",
            "        dataset_text_field   = 'text',\n",
            "        max_seq_length       = MAX_SEQ_LENGTH,\n",
            "        dataset_num_proc     = 2,\n",
            "        packing              = False,\n",
            "        args = TrainingArguments(\n",
            "            per_device_train_batch_size = 2,\n",
            "            gradient_accumulation_steps = 8,\n",
            "            warmup_steps                = 10,\n",
            "            max_steps                   = 60,\n",
            "            learning_rate               = 2e-4,\n",
            "            fp16             = not torch.cuda.is_bf16_supported(),\n",
            "            bf16             = torch.cuda.is_bf16_supported(),\n",
            "            logging_steps    = 1,\n",
            "            optim            = 'adamw_8bit',\n",
            "            weight_decay     = 0.01,\n",
            "            lr_scheduler_type = 'linear',\n",
            "            seed             = 3407,\n",
            "            output_dir       = 'outputs',\n",
            "        ),\n",
            "    )\n",
            "    print('SFTTrainer configured. Call trainer.train() to start.')\n",
            "    # trainer.train()\n",
            "else:\n",
            "    print('Cell 8 idle — training requires Colab GPU + Unsloth.')"
        ]
    },
]

# ── Construct notebook JSON ──────────────────────────────────────────────────
notebook_content = {
    "cells": CELLS,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3.13 (ChatBot)",
            "language": "python",
            "name": "chat-bot-env"
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.13.7"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

with open(NOTEBOOK_PATH, 'w') as f:
    json.dump(notebook_content, f, indent=2)

print(f"Generated {NOTEBOOK_PATH} — {len(CELLS)} cells")
print("Architecture: Bootstrap → Imports → Env → Config → Load → Persona → Helper → Data → Training")
