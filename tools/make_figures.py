#!/usr/bin/env python3
import sys, re, statistics as st, csv, pathlib
import matplotlib.pyplot as plt

def load_rows(path):
    pat = re.compile(r'\[CTRL\]\s*tick\s+(\d+)\s*\|\s*slices\s+(\d+)/(\d+)\s*\|\s*preds=(\d+)\s*\|\s*top0=\d+\s*\|\s*miss-ratio=([0-9.]+)')
    rows=[]
    for line in open(path, 'r', errors='ignore'):
        m = pat.search(line)
        if m:
            t, got, P, preds, miss = m.groups()
            rows.append((int(t), int(got), int(P), int(preds), float(miss)))
    return rows

def make_line_fig(x, y, xlabel, ylabel, outpng):
    plt.figure()
    plt.plot(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(outpng, dpi=150)
    plt.close()

def main():
    outdir = pathlib.Path("docs/figures")
    outdir.mkdir(parents=True, exist_ok=True)

    f4 = "results/dist-250ms-work1200-4P.txt"
    f6 = "results/dist-250ms-work1200-6P.txt"

    r4 = load_rows(f4)
    r6 = load_rows(f6)

    # 4) preds per tick 4P
    x4 = [r[0] for r in r4]
    y4 = [r[3] for r in r4]
    make_line_fig(x4, y4, "tick", "preds", outdir/"fig-preds-4P.png")

    # 5) preds per tick 6P
    x6 = [r[0] for r in r6]
    y6 = [r[3] for r in r6]
    make_line_fig(x6, y6, "tick", "preds", outdir/"fig-preds-6P.png")

    # 6) miss-ratio over ticks for the same 4P run
    m4 = [r[4] for r in r4]
    make_line_fig(x4, m4, "tick", "miss-ratio", outdir/"fig-missratio.png")

    # 8) scaling: mean preds vs number of predictors using 4P vs 6P runs
    mean4 = st.mean(y4) if y4 else 0
    mean6 = st.mean(y6) if y6 else 0
    plt.figure()
    plt.plot([4,6], [mean4, mean6], marker='o')
    plt.xticks([4,6])
    plt.xlabel("number of predictors")
    plt.ylabel("mean preds")
    plt.tight_layout()
    plt.savefig(outdir/"fig-scaling.png", dpi=150)
    plt.close()

    # 7) summary table CSV
    rows = []
    def summarize(tag, rows_):
        preds = [r[3] for r in rows_]
        bp_frac = sum(1 for p in preds if p < max(preds)) / len(preds) if preds else 0
        return dict(
            run=tag,
            ticks=len(rows_),
            P=rows_[0][2] if rows_ else "",
            mean_preds=round(st.mean(preds),1) if preds else 0,
            median_preds=round(st.median(preds),1) if preds else 0,
            final_miss_ratio=rows_[-1][4] if rows_ else 0.0,
            backpressure_fraction=round(bp_frac,2)
        )

    rows.append(summarize("4P work1200", r4))
    rows.append(summarize("6P work1200", r6))

    with open(outdir/"fig-summary-table.csv","w",newline="") as f:
        w=csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

if __name__ == "__main__":
    main()