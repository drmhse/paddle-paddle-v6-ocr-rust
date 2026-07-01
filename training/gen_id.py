#!/usr/bin/env python3
"""
Synthetic National-ID training data v2 — models REAL ppocr-server OCR noise.

Grounded in an actual kenya_id OCR dump:
  values appear BEFORE their labels; labels cluster together; headers/labels
  are garbled (REPURLICOFKENYA, DATEOEAIRTH, IDNUMBER); dates lose dots
  (25.051993, 14102011); stray junk lines appear (G, HOLDER'S SIGN., signature).

Target policy:
  - names / district / place / numbers / sex : copy the value VERBATIM as OCR'd
    (a 0.5B can't safely un-garble proper nouns — downstream can fuzzy-correct).
  - dates : NORMALIZED to DD.MM.YYYY (a learnable digit-regroup transform).
The hard skill being taught is FIELD ASSOCIATION under a scrambled layout.
"""
import json, random, os
random.seed(11)

SYSTEM = (
    "You extract fields from the OCR text of a Kenyan National ID card. "
    "The text is noisy and labels may be garbled or separated from their "
    "values (values can appear before the labels, and labels can be grouped "
    "together). Use the field meaning to assign each value: a 3-word ALL-CAPS "
    "personal name is FULL NAMES; 8-9 digit numbers are the serial/ID numbers; "
    "MALE/FEMALE is SEX; a place name is district/place of issue. Copy text "
    "values verbatim. Normalize dates to DD.MM.YYYY. Use null for a field that "
    "is genuinely absent. Output one JSON object, nothing else."
)

FIELDS = ["serial_number", "id_number", "full_names", "date_of_birth",
          "sex", "district_of_birth", "place_of_issue", "date_of_issue"]

LABELS = {
    "serial_number": "SERIAL NUMBER", "id_number": "ID NUMBER",
    "full_names": "FULL NAMES", "date_of_birth": "DATE OF BIRTH", "sex": "SEX",
    "district_of_birth": "DISTRICT OF BIRTH", "place_of_issue": "PLACE OF ISSUE",
    "date_of_issue": "DATE OF ISSUE",
}

FIRST = ["MARY","JOHN","GRACE","DAVID","FAITH","SAMUEL","ESTHER","JAMES","ANNE",
         "JOSEPH","ROSE","DANIEL","LUCY","BRIAN","MERCY","KEVIN","JOAN","DENNIS",
         "PURITY","VICTOR","MIKE","PETER","JANE","ABDUL","HASSAN","NAOMI"]
MID   = ["WANJIRU","OUMA","CHEROTICH","MUTHONI","WAFULA","AKINYI","KIPLAGAT",
         "ADHIAMBO","WAMBUI","OTIENO","CHEPKEMOI","NAFULA","WEKESA","JEROP",
         "MUKAMI","ONYANGO","KIPNGENO","ATIENO","KIPLIMO","NJERI","OMAR"]
LAST  = ["KAMAU","ODHIAMBO","KORIR","MWANGI","BARASA","OCHIENG","NJUGUNA","RUTO",
         "OWINO","GITHINJI","MAINA","KIPROTICH","WANYAMA","MUTISO","KEMBOI",
         "OMONDI","WAMBUA","CHELIMO","CHUMBA","ALI","HUSSEIN"]
DISTRICTS = ["NANDI SOUTH","TINDIRET","KISUMU EAST","KIAMBU","MACHAKOS",
             "UASIN GISHU","BUNGOMA","IMENTI NORTH","NAKURU","KAKAMEGA","NYERI",
             "KILIFI","TRANS NZOIA","KERICHO","BOMET","MIGORI","HOMA BAY",
             "KAJIADO","NAROK","VIHIGA","BUSIA","EMBU"]
JUNK = ["G","HOLDER'S SIGN.","Nitue","Sign","2AFE","O","R","Safe","af","M"]

SUBS = {"O":"0","I":"T","L":"T","B":"R","F":"E","S":"5","U":"J","R":"A",
        "D":"0","G":"6","E":"A","T":"I","N":"H","C":"O"}

def garble(s, p=0.12, drop_space=0.35):
    out = []
    for ch in s:
        if ch == " ":
            if random.random() < drop_space:
                continue
            out.append(" "); continue
        u = ch.upper()
        if u in SUBS and random.random() < p:
            out.append(SUBS[u])
        else:
            out.append(ch)
    return "".join(out)

def gtruth_date(y0, y1):
    return f"{random.randint(1,28):02d}.{random.randint(1,12):02d}.{random.randint(y0,y1)}"

def ocr_date(norm):                       # "25.05.1993" -> noisy display form
    d, m, y = norm.split(".")
    forms = [f"{d}.{m}.{y}", f"{d}.{m}{y}", f"{d}{m}{y}", f"{d} {m} {y}"]
    return random.choice(forms)

def label(f):                             # garbled label, maybe with ':' and no space
    base = LABELS[f]
    s = garble(base, p=0.18) if random.random() < 0.6 else base
    if random.random() < 0.35:
        s = s.replace(" ", "")
    if random.random() < 0.3:
        s += ":"
    return s

def make_example():
    dob_year = random.randint(1965, 2004)
    name_clean = f"{random.choice(FIRST)} {random.choice(MID)} {random.choice(LAST)}"
    dob_norm = gtruth_date(dob_year, dob_year)
    iss_norm = gtruth_date(min(dob_year + 18, 2022), 2022)

    serial = str(random.randint(100000000, 999999999))
    idnum  = str(random.randint(10000000, 39999999))
    name_ocr = garble(name_clean, p=0.14, drop_space=0.45)   # value as OCR'd
    dist_ocr = garble(random.choice(DISTRICTS), p=0.12, drop_space=0.15)
    place_ocr = garble(random.choice(DISTRICTS), p=0.12, drop_space=0.15)
    sex = random.choice(["MALE", "FEMALE"])

    truth = {
        "serial_number": serial, "id_number": idnum, "full_names": name_ocr,
        "date_of_birth": dob_norm, "sex": sex, "district_of_birth": dist_ocr,
        "place_of_issue": place_ocr, "date_of_issue": iss_norm,
    }
    for f in ["district_of_birth", "place_of_issue", "serial_number"]:
        if random.random() < 0.1:
            truth[f] = None

    lines = []
    def junk():
        if random.random() < 0.4:
            lines.append(random.choice(JUNK))

    if random.random() < 0.9:
        lines.append(garble("JAMHURI YA KENYA", p=0.2))
        lines.append(garble("REPUBLIC OF KENYA", p=0.2))

    mode = random.random()
    if mode < 0.55:
        if truth["serial_number"]: lines.append(truth["serial_number"])
        lines.append(idnum)
        lines.append(label("serial_number"))
        lines.append(label("id_number"))
        lines.append(label("full_names"))
        lines.append(name_ocr); junk()
        rest = ["date_of_birth", "sex", "district_of_birth", "place_of_issue", "date_of_issue"]
    elif mode < 0.8:
        lines.append(label("serial_number"))
        if truth["serial_number"]: lines.append(truth["serial_number"])
        lines.append(idnum); lines.append(label("id_number"))
        lines.append(label("full_names")); lines.append(name_ocr); junk()
        rest = ["date_of_birth", "sex", "district_of_birth", "place_of_issue", "date_of_issue"]
    else:
        rest = list(FIELDS)

    for f in rest:
        if truth.get(f) is None and f in ("district_of_birth", "place_of_issue", "serial_number"):
            continue
        lines.append(label(f))
        if f in ("date_of_birth", "date_of_issue"):
            lines.append(ocr_date(truth[f]))
        elif f == "serial_number":
            lines.append(serial)
        elif f == "id_number":
            lines.append(idnum)
        else:
            lines.append(truth[f] if truth[f] else "")
        junk()

    if random.random() < 0.6:
        lines.append(random.choice(["HOLDER'S SIGN.", "Sign", "Nitue"]))
        if random.random() < 0.5:
            lines.append(garble(name_clean.split()[0], p=0.4))

    user = "OCR lines:\n" + "\n".join(lines) + "\n\nReturn the ID fields as JSON."
    target = {k: truth.get(k) for k in FIELDS}
    return {"messages": [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user},
        {"role": "assistant", "content": json.dumps(target, ensure_ascii=False)},
    ]}

def main():
    os.makedirs("data_id", exist_ok=True)
    with open("data_id/train.jsonl", "w") as f:
        for _ in range(1600):
            f.write(json.dumps(make_example(), ensure_ascii=False) + "\n")
    with open("data_id/valid.jsonl", "w") as f:
        for _ in range(160):
            f.write(json.dumps(make_example(), ensure_ascii=False) + "\n")
    print("wrote data_id/train.jsonl (1600) + data_id/valid.jsonl (160)")

if __name__ == "__main__":
    main()
