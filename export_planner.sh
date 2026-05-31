#!/usr/bin/env bash
# Export the DAPO-trained Planner to Ollama for the "after" eval.
#
#   merged bf16 HF model  ->  f16 GGUF (llama.cpp)  ->  Ollama model (quantized
#   to Q4_K_M to MATCH the baseline qwen3:8b, so before/after differ only by
#   training, not by quantization).
#
# Run on the pod:  bash export_planner.sh
set -euo pipefail
cd /workspace/agentflow-pro

MERGED=artifacts/planner-dapo-merged
GGUF=artifacts/planner-f16.gguf
MODEL=agentflow-planner

echo "==> 0. sanity: the merged model must exist"
test -d "$MERGED" || { echo "ERROR: $MERGED not found — did DAPO finish and save?"; exit 1; }
ls -lh "$MERGED" | head

echo "==> 1. llama.cpp converter"
[ -d /workspace/llama.cpp ] || git clone --depth 1 https://github.com/ggml-org/llama.cpp /workspace/llama.cpp
uv pip install -q -r /workspace/llama.cpp/requirements/requirements-convert_hf_to_gguf.txt

echo "==> 2. convert merged bf16 model -> f16 GGUF (this reads the 8B; ~2-5 min)"
uv run python /workspace/llama.cpp/convert_hf_to_gguf.py "$MERGED" --outfile "$GGUF" --outtype f16
ls -lh "$GGUF"

echo "==> 3. ensure ollama is installed + serving"
command -v ollama >/dev/null 2>&1 || curl -fsSL https://ollama.com/install.sh | sh
pgrep -x ollama >/dev/null 2>&1 || { nohup ollama serve > /workspace/ollama.log 2>&1 & sleep 5; }

echo "==> 4. pull stock qwen3:8b (only to inherit its chat template + stop params)"
ollama pull qwen3:8b

echo "==> 5. build Modelfile: keep qwen3:8b's TEMPLATE/PARAMS, swap in our weights"
ollama show qwen3:8b --modelfile | grep -vE '^FROM ' > Modelfile.tmpl
printf 'FROM ./%s\n' "$GGUF" | cat - Modelfile.tmpl > Modelfile

echo "==> 6. create '$MODEL', quantized to Q4_K_M (matches the baseline's quant)"
ollama create "$MODEL" --quantize q4_K_M -f Modelfile

echo "==> 7. smoke test (should print a short JSON-ish reply, no errors)"
ollama run "$MODEL" "Reply with ONLY this JSON: {\"thought\":\"hi\",\"action\":\"answer\",\"action_input\":\"ok\"}"

echo
echo "==> DONE. '$MODEL' is registered in Ollama. Next: run the re-eval (see chat)."
