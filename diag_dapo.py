"""Throwaway diagnostic for the DAPO empty-prompt crash. Run on the pod:
    uv run python diag_dapo.py
CPU-only, no model weights — just the tokenizer + the real prompt rows.
"""
import transformers, tokenizers
from transformers import AutoTokenizer
from train.dapo import build_prompt_dataset

print("transformers:", transformers.__version__, "| tokenizers:", tokenizers.__version__)

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
print("tokenizer class:", type(tok).__name__)

rows = build_prompt_dataset("runs/eval_aime_train_*.json")
print("rows:", len(rows))

# scan every row for an empty tokenization
empties = []
for i, r in enumerate(rows):
    txt = tok.apply_chat_template(r["prompt"], tokenize=False, add_generation_prompt=True)
    nids = len(tok(txt)["input_ids"]) if txt else 0
    if nids == 0:
        empties.append(i)
print("rows that tokenize to EMPTY:", len(empties), empties[:5])

r0 = rows[0]
print("\n--- row 0 ---")
print("roles:", [m["role"] for m in r0["prompt"]])
print("system len:", len(r0["prompt"][0]["content"]))
print("user len:", len(r0["prompt"][1]["content"]))
txt0 = tok.apply_chat_template(r0["prompt"], tokenize=False, add_generation_prompt=True)
print("apply_chat_template(tokenize=False) text len:", len(txt0) if txt0 else 0)
print("text head:", repr(txt0[:120]) if txt0 else "EMPTY")

# (A) the CURRENT two-step path in train/dapo.py
enc_two = tok([txt0] * 6, return_tensors="pt", padding=True)
print("\n(A) CURRENT two-step  input_ids shape:", tuple(enc_two["input_ids"].shape))

# (B) the CANONICAL single-step path (candidate fix)
enc_one = tok.apply_chat_template(
    r0["prompt"], add_generation_prompt=True, return_tensors="pt", return_dict=True,
)
print("(B) CANONICAL one-step input_ids shape:", tuple(enc_one["input_ids"].shape))
