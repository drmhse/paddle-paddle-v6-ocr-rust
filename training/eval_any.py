#!/usr/bin/env python3
"""Generic held-out eval.  Usage: python eval_any.py <gen_module> <model> <adapter|none> [N]"""
import json, random, sys, importlib
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

gen = importlib.import_module(sys.argv[1])               # gen_data | gen_id
MODEL = sys.argv[2]
adapter = None if sys.argv[3] == "none" else sys.argv[3]
N = int(sys.argv[4]) if len(sys.argv) > 4 else 60

random.seed(4242)                                        # held-out seed (unseen)
data = [(ex["messages"][:2], json.loads(ex["messages"][2]["content"]))
        for ex in (gen.make_example() for _ in range(N))]

model, tok = load(MODEL, adapter_path=adapter)
sampler = make_sampler(temp=0.0)

def prompt_for(msgs):
    if getattr(tok, "chat_template", None):
        return tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    # fallback for tokenizers без chat template (e.g. Supra-50M)
    sys_, usr = msgs[0]["content"], msgs[1]["content"]
    return f"{sys_}\n\n{usr}\n"

import re
def parse(out, keys):
    try:
        return {k: json.loads(out[out.index("{"):]).get(k) for k in keys}
    except Exception:
        d = {}
        for k in keys:
            m = re.search(rf'"{re.escape(k)}"\s*:\s*("(?:[^"\\]|\\.)*"|null)', out)
            d[k] = json.loads(m.group(1)) if m else None
        return d

hits = {f: 0 for f in gen.FIELDS}; full = jok = 0
for msgs, truth in data:
    raw = generate(model, tok, prompt=prompt_for(msgs), max_tokens=256, sampler=sampler, verbose=False)
    try: json.loads(raw); jok += 1
    except Exception: pass
    pred = parse(raw, gen.FIELDS)
    ok = True
    for f in gen.FIELDS:
        if pred.get(f) == truth.get(f): hits[f] += 1
        else: ok = False
    full += ok

tag = "LORA" if adapter else "BASE"
print(f"\n=== {tag} ({sys.argv[1]}, {MODEL.split('/')[-1]}) — {N} held-out ===")
print(f"valid JSON: {jok}/{N}   full-record exact: {full}/{N} ({100*full/N:.0f}%)")
for f in gen.FIELDS:
    print(f"  {f:20s} {hits[f]:2d}/{N}")
print(f"micro-avg field accuracy: {100*sum(hits.values())/(N*len(gen.FIELDS)):.1f}%")
