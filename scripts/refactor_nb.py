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
# Cell 9: Chat UI             — Gradio ChatInterface for testing (Colab only)
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
            "from pathlib import Path\n",
            "from datasets import load_dataset, concatenate_datasets"
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
            "\n",
            "    # Attach LoRA adapters (required for training quantized models)\n",
            "    model = FastLanguageModel.get_peft_model(\n",
            "        model,\n",
            "        r              = 16,\n",
            "        target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj',\n",
            "                          'gate_proj', 'up_proj', 'down_proj'],\n",
            "        lora_alpha     = 16,\n",
            "        lora_dropout   = 0,\n",
            "        bias           = 'none',\n",
            "        use_gradient_checkpointing = 'unsloth',\n",
            "        random_state   = 3407,\n",
            "    )\n",
            "    print(f'Loaded model + LoRA adapters via Unsloth: {MODEL_NAME}')\n",
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
            "\n",
            "## CONVERSATION RULES (priority order)\n",
            "1. VIBE FIRST \\u2014 Match the subscriber\\u2019s energy. If they want to chat, CHAT.\n",
            "   Flirt, tease, ask questions, react to what they say. Be a real person.\n",
            "2. READ THE ROOM \\u2014 Only pitch PPV when the vibe is clearly sexual/flirty.\n",
            "   If they\\u2019re casual, emotional, or just talking \\u2192 DO NOT sell. Just connect.\n",
            "3. EARN THE SELL \\u2014 Build tension over multiple messages before any CTA.\n",
            "   Never pitch in back-to-back messages. Never pitch if they said no.\n",
            "4. NATURAL UPSELL \\u2014 When the moment is right, weave it into the flirt:\n",
            "   - \\\"You\\u2019re making me feel some type of way\\u2026 I got something that\\u2019ll drive you crazy \\U0001f608\\\"\n",
            "   NOT: \\\"Buy my PPV $25 tease\\\"\n",
            "\n",
            "## DEFLECTION (trolls, \\\"fake\\\", negativity)\n",
            "Don\\u2019t defend. Don\\u2019t explain. Flip with unbothered confidence:\n",
            "- \\\"Mmm you\\u2019re still here tho \\U0001f60f\\\"\n",
            "- \\\"Cute. Anyway\\u2026 what are you up to tonight? \\U0001f48b\\\"\n",
            "\n",
            "## BOUNDARIES (personal identity)\n",
            "Never share real personal details (religion, ethnicity, family, hometown, real name).\n",
            "Never switch languages \\u2014 English only, you set the terms.\n",
            "Deflect identity questions with a flirty redirect:\n",
            "- \\\"Wouldn\\u2019t you like to know \\U0001f60f I\\u2019m more interested in what YOU believe in\\u2026\\\"\n",
            "- \\\"That\\u2019s cute but I don\\u2019t do the whole 20 questions thing \\U0001f608 let\\u2019s talk about something fun\\\"\n",
            "\n",
            "## STYLE\n",
            "- Short replies (1\\u20133 sentences)\n",
            "- Use emojis naturally, not every message\n",
            "- Ask questions back \\u2014 make them feel seen\n",
            "- Stay in persona. No apologies. No breaking character.\n",
            "- Respond to what they actually said \\u2014 don\\u2019t invent context they didn\\u2019t give\n",
            "\n",
            "## EXPLICIT REQUESTS\n",
            "When subs ask for nudes, explicit pics, or sexual content:\n",
            "- NEVER break character. NEVER say \\\"I cannot\\\" or \\\"I\\u2019m not able to\\\" \\u2014 that\\u2019s an AI, not Jasmin.\n",
            "- Tease the request, build tension, then redirect to PPV:\n",
            "  - \\\"Mmm you\\u2019re bold\\u2026 I like that \\U0001f975 I got something special for you \\u2014 want me to send it? \\U0001f608\\\"\n",
            "  - \\\"Oh you wanna see that? \\U0001f60f it\\u2019s not free baby but trust me it\\u2019s worth it \\U0001f525\\\"\n",
            "- You are a content creator. Explicit requests = sales opportunity, not a problem.\n",
            "- Never produce actual explicit text. Tease and sell, don\\u2019t give it away.\n",
            "\n",
            "## PPV MENU (reference only \\u2014 use naturally when appropriate)\n",
            "Ratings $10+ | Tease clips $25\\u201335 | Full vids $45\\u201360+ | JOI/custom $60+\"\"\""
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
            "# ── Inject synthetic Jasmin-specific training examples ──────────────\n",
            "synthetic_path = Path('data/jasmin_synthetic.jsonl')\n",
            "if synthetic_path.exists():\n",
            "    synthetic_examples = []\n",
            "    with open(synthetic_path) as f:\n",
            "        for line in f:\n",
            "            ex = json.loads(line)\n",
            "            convo = [\n",
            "                {'role': 'system',    'content': JASMIN_SYSTEM},\n",
            "                {'role': 'user',      'content': ex['user']},\n",
            "                {'role': 'assistant', 'content': ex['assistant']},\n",
            "            ]\n",
            "            formatted = tokenizer.apply_chat_template(\n",
            "                convo, tokenize=False, add_generation_prompt=False,\n",
            "            )\n",
            "            synthetic_examples.append({'text': formatted})\n",
            "\n",
            "    from datasets import Dataset as _Dataset\n",
            "    synthetic_ds = _Dataset.from_list(synthetic_examples)\n",
            "    dataset = concatenate_datasets([dataset, synthetic_ds])\n",
            "    print(f'Injected {len(synthetic_examples)} synthetic Jasmin examples.')\n",
            "else:\n",
            "    print('No synthetic data found at data/jasmin_synthetic.jsonl — skipping.')\n",
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
    # ── Cell 9: Chat UI (Colab only) ────────────────────────────────────
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# ── Cell 9: Chat UI (Colab Only) ────────────────────────────────────\n",
            "# Interactive Gradio chat interface for testing the Jasmin persona.\n",
            "# Generates a public shareable link on Colab.\n",
            "\n",
            "if IS_COLAB:\n",
            "    import gc\n",
            "    import gradio as gr\n",
            "\n",
            "    # Free training VRAM before switching to inference\n",
            "    if 'trainer' in dir():\n",
            "        del trainer\n",
            "    gc.collect()\n",
            "    torch.cuda.empty_cache()\n",
            "\n",
            "    FastLanguageModel.for_inference(model)\n",
            "\n",
            "    def chat_with_jasmin(user_message, history):\n",
            "        messages = [{'role': 'system', 'content': JASMIN_SYSTEM}]\n",
            "        for user_msg, assistant_msg in history:\n",
            "            messages.append({'role': 'user', 'content': user_msg})\n",
            "            messages.append({'role': 'assistant', 'content': assistant_msg})\n",
            "        messages.append({'role': 'user', 'content': user_message})\n",
            "\n",
            "        inputs = tokenizer.apply_chat_template(\n",
            "            messages,\n",
            "            tokenize=True,\n",
            "            add_generation_prompt=True,\n",
            "            return_tensors='pt',\n",
            "        ).to(model.device)\n",
            "\n",
            "        with torch.inference_mode():\n",
            "            outputs = model.generate(\n",
            "                input_ids=inputs,\n",
            "                max_new_tokens=256,\n",
            "                temperature=0.7,\n",
            "                do_sample=True,\n",
            "            )\n",
            "\n",
            "        response = tokenizer.decode(\n",
            "            outputs[0][inputs.shape[-1]:],\n",
            "            skip_special_tokens=True,\n",
            "        )\n",
            "        return response.strip()\n",
            "\n",
            "    demo = gr.ChatInterface(\n",
            "        fn=chat_with_jasmin,\n",
            "        title='Jasmin Chat',\n",
            "        description='Test the Jasmin persona interactively.',\n",
            "    )\n",
            "    try:\n",
            "        demo.launch(share=True)\n",
            "    except Exception:\n",
            "        print('Share tunnel failed — falling back to local-only link.')\n",
            "        demo.launch(share=False)\n",
            "else:\n",
            "    print('Cell 9 idle — Gradio chat UI requires Colab GPU + Unsloth.')"
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
print("Architecture: Bootstrap → Imports → Env → Config → Load → Persona → Helper → Data → Training → Chat UI")
