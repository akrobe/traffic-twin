#!/usr/bin/env python3
import re, csv, sys, pathlib, statistics as st
import matplotlib.pyplot as plt

FIGDIR = pathlib.Path("docs/figures"); FIGDIR.mkdir(parents=True, exist_ok=True)

CTRL_RE = re.compile(r'\[CTRL\]\s*tick\s+(\d+)\s*\|\s*slices\s+(\d+)/(\d+)\s*\|\s*preds=(\d+)\s*\|\s*top0=\d+\s*\|\s*miss-ratio=([0-9.]+)')
SMP_RE  = re.compile(r'tick\s+(\d+)\s*\|\s*preds=(\d+)\s*\|\s*top\[0\]=\d+\s*\|\s*queues\s+IA:(\d+)\s+AP:(\d+)\s+PC:(\d+)')

def parse_ctrl(path):
    rows=[]
    for line in open(path, 'r', errors='ignore'):
        m=CTRL_RE.search(line)
        if m:
            t, got, P, preds, miss = m.groups()
            rows.append((int(t), int(got), int(P), int(preds), float(miss)))
    if not rows:
        raise RuntimeError(f"no CTRL lines in {path}")
    return rows

def parse_smp(path):
    rows=[]
    for line in open(path, 'r', errors='ignore'):
        m=SMP_RE.search(line)
        if m:
            t, preds, ia, ap, pc = m.groups()
            rows.append((int(t), int(preds), int(ia), int(ap), int(pc)))
    if not rows:
        raise RuntimeError(f"no SMP lines in {path}")
    return rows

def save_csv(rows, out_csv, header):
    with open(out_csv, "w", newline="") as f:
        w=csv.writer(f); w.writerow(header); w.writerows(rows)

def plot_preds(path, out_png, title):
    rows=parse_ctrl(path)
    ticks=[r[0] for r in rows]
    preds=[r[3] for r in rows]
    plt.figure()
    plt.plot(ticks, preds, marker='o')
    plt.xlabel("tick")
    plt.ylabel("preds")
    plt.title(title)
    plt.grid(True, linewidth=0.3)
    plt.tight_layout()
    plt.savefig(FIGDIR/out_png, dpi=160)
    plt.close()

def plot_miss(path, out_png, title):
    rows=parse_ctrl(path)
    ticks=[r[0] for r in rows]
    miss=[r[4] for r in rows]
    plt.figure()
    plt.plot(ticks, miss, marker='o')
    plt.xlabel("tick")
    plt.ylabel("miss-ratio")
    plt.title(title)
    plt.grid(True, linewidth=0.3)
    plt.tight_layout()
    plt.savefig(FIGDIR/out_png, dpi=160)
    plt.close()

def plot_scaling(paths, labels, out_png):
    means=[]
    xs=[]
    for p,label in zip(paths, labels):
        rows=parse_ctrl(p)
        preds=[r[3] for r in rows]
        means.append(st.mean(preds))
        xs.append(label)
    plt.figure()
    plt.plot(xs, means, marker='o')
    plt.xlabel("predictor ranks (P)")
    plt.ylabel("mean preds")
    plt.title("Scaling: mean preds vs predictor ranks")
    plt.grid(True, linewidth=0.3)
    plt.tight_layout()
    plt.savefig(FIGDIR/out_png, dpi=160)
    plt.close()

def plot_smp_queues(path, out_png):
    rows=parse_smp(path)
    t=[r[0] for r in rows]
    ia=[r[2] for r in rows]; ap=[r[3] for r in rows]; pc=[r[4] for r in rows]
    plt.figure()
    plt.plot(t, ia, label="IA")
    plt.plot(t, ap, label="AP")
    plt.plot(t, pc, label="PC")
    plt.xlabel("tick")
    plt.ylabel("queue depth")
    plt.title("SMP queues over ticks")
    plt.legend()
    plt.grid(True, linewidth=0.3)
    plt.tight_layout()
    plt.savefig(FIGDIR/out_png, dpi=160)
    plt.close()

def draw_deadline(out_png):
    # simple conceptual timeline
    import numpy as np
    T=1.0
    slices=[("ingest",0.05,0.20),("aggregate",0.25,0.40),("predict",0.42,0.85),("control",0.86,0.98)]
    plt.figure(figsize=(6,1.8))
    for name,s,e in slices:
        plt.hlines(1, s, e, linewidth=12)
        plt.text((s+e)/2, 1.02, name, ha='center', va='bottom', fontsize=9)
    # deadline line
    plt.vlines(T, 0.7, 1.3, linestyles='dashed')
    plt.text(T, 0.68, "deadline", ha='center', va='top', fontsize=9)
    # late region example
    plt.hlines(1, 1.05, 1.20, colors='r', linewidth=12)
    plt.text(1.125, 0.95, "drop-late", ha='center', va='top', fontsize=8, color='r')
    plt.xlim(0,1.25); plt.ylim(0.7,1.3); plt.yticks([])
    plt.xlabel("time (seconds)")
    plt.tight_layout()
    plt.savefig(FIGDIR/out_png, dpi=160, bbox_inches='tight')
    plt.close()

def write_summary_csv(rows4, rows6, out_csv):
    def stats(rows):
        preds=[r[3] for r in rows]
        miss=[r[4] for r in rows]
        P=rows[0][2]
        bp_frac=sum(1 for p in preds if p < max(preds))/len(preds)
        return dict(P=P, mean=st.mean(preds), median=st.median(preds), bp_frac=bp_frac, final_miss=miss[-1])
    s4=stats(rows4); s6=stats(rows6)
    with open(FIGDIR/out_csv, "w", newline="") as f:
        w=csv.writer(f)
        w.writerow(["run","predictor_ranks_P","mean_preds","median_preds","bp_fraction","final_miss_ratio"])
        w.writerow(["work1200_4P", s4["P"], f'{s4["mean"]:.1f}', f'{s4["median"]:.1f}', f'{s4["bp_frac"]:.2f}', f'{s4["final_miss"]:.2f}'])
        w.writerow(["work1200_6P", s6["P"], f'{s6["mean"]:.1f}', f'{s6["median"]:.1f}', f'{s6["bp_frac"]:.2f}', f'{s6["final_miss"]:.2f}'])

def main():
    # Inputs you’ve already created
    f4 = "results/dist-250ms-work1200-4P.txt"
    f6 = "results/dist-250ms-work1200-6P.txt"
    f4b = "results/dist-1s-work0-4P.txt"
    smp = "results/smp.txt"

    # Plots for report
    plot_preds(f4,  "fig-preds-4P.png", "preds per tick — 4P under load")
    plot_preds(f6,  "fig-preds-6P.png", "preds per tick — 6P under load")
    plot_miss(f4,   "fig-missratio.png","miss-ratio over ticks — 4P under load")
    plot_scaling([f4, f6], labels=[1,3], out_png="fig-scaling.png")  # P in your logs: 4 MPI -> P=1, 6 MPI -> P=3
    plot_smp_queues(smp, "fig-smp-queues.png")
    draw_deadline("fig-deadline.png")

    # Summary table
    rows4=parse_ctrl(f4); rows6=parse_ctrl(f6)
    write_summary_csv(rows4, rows6, "fig-summary-table.csv")

    # Also keep CSV exports of series for archival
    save_csv(rows4, FIGDIR/"fig-preds-4P.csv", ["tick","got","P","preds","miss_ratio"])
    save_csv(rows6, FIGDIR/"fig-preds-6P.csv", ["tick","got","P","preds","miss_ratio"])

    print("wrote figures to", FIGDIR)

if __name__ == "__main__":
    main()
