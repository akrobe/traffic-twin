# tools/parse_ctrl.py
import re, sys, csv, statistics as st

if len(sys.argv) < 3:
    print("usage: python3 tools/parse_ctrl.py <input_log.txt> <output_csv>")
    sys.exit(1)

inp, outp = sys.argv[1], sys.argv[2]
rows = []
rx = re.compile(
    r'\[CTRL\] tick\s+(\d+)\s+\|\s+slices\s+(\d+)/(\d+)\s+\|\s+preds=(\d+)'
    r'\s+\|\s+top0=\d+\s+\|\s+miss-ratio=([0-9.]+)'
)

with open(inp) as f:
    for ln in f:
        m = rx.search(ln)
        if m:
            tick, got, P, preds, miss = m.groups()
            rows.append({
                "tick": int(tick),
                "slices_got": int(got),
                "predictors": int(P),
                "preds": int(preds),
                "miss_ratio": float(miss),
            })

if not rows:
    print("no CTRL lines parsed from", inp)
    sys.exit(2)

with open(outp, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader()
    w.writerows(rows)

preds_list = [r["preds"] for r in rows]
bp_frac = sum(1 for p in preds_list if p < max(preds_list)) / len(preds_list)
print("ticks:", len(rows))
print("P:", rows[0]["predictors"])
print("mean preds:", round(st.mean(preds_list), 2))
print("median preds:", st.median(preds_list))
print("final miss-ratio:", rows[-1]["miss_ratio"])
print("back-pressure ticks fraction:", f"{bp_frac:.2f}")