#!/usr/bin/env python3
"""
Understanding sidecar: browser -> upload image -> ppocr-server OCR (kenya_id)
-> Qwen2.5-0.5B + LoRA extraction -> structured JSON, rendered on a page.

Run:  uvicorn sidecar:app --host 127.0.0.1 --port 8099   (ppocr-server on :8080)
"""
import io, json, time, re
import requests
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse

import gen_id
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

OCR_URL = "http://127.0.0.1:8080/v1/ocr"
BASE = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER = "./adapters_id"

def fields_from_text(text, keys):
    """First "key": value per schema key — robust to looping / unclosed JSON."""
    out = {}
    for k in keys:
        m = re.search(rf'"{re.escape(k)}"\s*:\s*("(?:[^"\\]|\\.)*"|null)', text)
        out[k] = (json.loads(m.group(1)) if m else None)
    return out

def first_json_object(text):
    depth = 0; start = None
    for i, c in enumerate(text):
        if c == "{":
            if depth == 0: start = i
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0 and start is not None:
                try: return json.loads(text[start:i+1])
                except Exception: return None
    return None

print("loading extraction model (Qwen2.5-0.5B + LoRA)...")
MODEL, TOK = load(BASE, adapter_path=ADAPTER)
SAMPLER = make_sampler(temp=0.0)
print("ready.")

app = FastAPI(title="ID understanding sidecar")

PAGE = """<!doctype html><meta charset=utf-8><title>Kenya ID extractor</title>
<style>
 body{font:15px system-ui;margin:40px auto;max-width:760px;color:#222}
 h1{font-size:20px} .drop{border:2px dashed #bbb;border-radius:10px;padding:30px;text-align:center;color:#666}
 img{max-width:320px;border-radius:8px;margin-top:14px;display:none}
 pre{background:#0f111a;color:#d7e0ff;padding:16px;border-radius:8px;overflow:auto}
 .row{display:flex;gap:24px;flex-wrap:wrap} .col{flex:1;min-width:280px}
 .muted{color:#888;font-size:13px} button{font:15px system-ui;padding:8px 16px;border-radius:8px;border:1px solid #ccc;cursor:pointer}
</style>
<h1>Kenya National ID — OCR + understanding</h1>
<p class=muted>Upload an ID image. Pipeline: ppocr-server (kenya_id) &rarr; Qwen2.5-0.5B+LoRA &rarr; JSON. All local.</p>
<div class=drop><input type=file id=f accept="image/*"> <button onclick="go()">Extract</button></div>
<img id=prev>
<div class=row>
 <div class=col><h3>Extracted fields</h3><pre id=out>&mdash;</pre></div>
 <div class=col><h3>Raw OCR text</h3><pre id=ocr>&mdash;</pre><p class=muted id=timing></p></div>
</div>
<script>
const f=document.getElementById('f'),prev=document.getElementById('prev');
f.onchange=()=>{if(f.files[0]){prev.src=URL.createObjectURL(f.files[0]);prev.style.display='block';}};
async function go(){
 if(!f.files[0]){alert('pick an image');return;}
 document.getElementById('out').textContent='extracting…';
 const fd=new FormData();fd.append('image',f.files[0]);
 const r=await fetch('/extract',{method:'POST',body:fd});const j=await r.json();
 document.getElementById('out').textContent=JSON.stringify(j.fields,null,2);
 document.getElementById('ocr').textContent=j.ocr_text||'(none)';
 document.getElementById('timing').textContent=`OCR ${j.ocr_ms} ms · understanding ${j.llm_ms} ms · ${j.num_lines} lines`;
}
</script>"""

@app.get("/", response_class=HTMLResponse)
def home(): return PAGE

@app.post("/extract")
async def extract(image: UploadFile = File(...)):
    raw = await image.read()
    t0 = time.time()
    ocr = requests.post(OCR_URL, params={"mode": "kenya_id",
                        "det_model": "PP-OCRv6_medium_det",
                        "rec_model": "PP-OCRv6_medium_rec"}, data=raw,
                        headers={"Content-Type": "application/octet-stream"}, timeout=60)
    ocr_ms = int((time.time() - t0) * 1000)
    if ocr.status_code != 200:
        return JSONResponse({"error": f"ocr failed: {ocr.status_code}", "detail": ocr.text}, status_code=502)
    oj = ocr.json(); ocr_text = oj.get("text", "")

    msgs = [{"role": "system", "content": gen_id.SYSTEM},
            {"role": "user", "content": f"OCR lines:\n{ocr_text}\n\nReturn the ID fields as JSON."}]
    prompt = TOK.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    t1 = time.time()
    out = generate(MODEL, TOK, prompt=prompt, max_tokens=220, sampler=SAMPLER, verbose=False)
    llm_ms = int((time.time() - t1) * 1000)

    obj = first_json_object(out)
    fields = {k: obj.get(k) for k in gen_id.FIELDS} if obj else fields_from_text(out, gen_id.FIELDS)
    return {"fields": fields, "ocr_text": ocr_text,
            "num_lines": oj.get("num_lines"), "ocr_ms": ocr_ms, "llm_ms": llm_ms}

@app.get("/health")
def health(): return {"ok": True}
