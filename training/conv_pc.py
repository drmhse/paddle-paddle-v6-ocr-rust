#!/usr/bin/env python3
"""Convert chat-format data_id/*.jsonl -> prompt/completion data_id_pc/*.jsonl
(for models without a chat template, e.g. Supra-50M). Prompt string MUST match
eval_any.py's no-chat-template fallback: f"{system}\n\n{user}\n"."""
import json, os
os.makedirs("data_id_pc", exist_ok=True)
for split in ("train", "valid"):
    with open(f"data_id/{split}.jsonl") as fin, open(f"data_id_pc/{split}.jsonl", "w") as fout:
        for ln in fin:
            m = json.loads(ln)["messages"]
            sys_, usr, asst = m[0]["content"], m[1]["content"], m[2]["content"]
            fout.write(json.dumps({"prompt": f"{sys_}\n\n{usr}\n", "completion": asst},
                                  ensure_ascii=False) + "\n")
print("wrote data_id_pc/{train,valid}.jsonl")
