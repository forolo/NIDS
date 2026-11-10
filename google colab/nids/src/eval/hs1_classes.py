"""HS1 - Rappel par classe d'attaque et par graine : nombre de graines atteignant le seuil (par défaut 0,99).

Lit les prédictions du test (data/embeddings/pred{motif}_test.npy, écrites par train_temporal.py --save-preds) et les étiquettes
(data/graphs/test.npz). Numpy seul.

Usage (depuis la racine du projet) :
    python -m src.eval.hs1_classes --pattern '_tl_f10_s{}' --seeds 0,1,2,3,4,5,6,7,8,9 --tag _tl_f10
"""
import argparse
import json
from pathlib import Path

import numpy as np

from src.eval.confusion_analysis import confusion


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph-dir", default="data/graphs")
    ap.add_argument("--emb-dir", default="data/embeddings")
    ap.add_argument("--pattern", required=True)
    ap.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9")
    ap.add_argument("--threshold", type=float, default=0.99)
    ap.add_argument("--split", default="test")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    gd, ed = Path(args.graph_dir), Path(args.emb_dir)
    meta = json.loads((gd / "meta.json").read_text(encoding="utf-8"))
    names = [k for k, _ in sorted(meta["label_id"].items(), key=lambda t: t[1])]
    K = len(names)
    y = np.load(gd / f"{args.split}.npz")["y"].astype(np.int64)
    seeds = [int(s) for s in args.seeds.split(",")]
    rec = np.full((len(seeds), K), np.nan)
    for i, s in enumerate(seeds):
        p = np.load(ed / f"pred{args.pattern.format(s)}_{args.split}.npy").astype(np.int64)
        cm = confusion(y, p, K)
        for k in range(K):
            if cm[k].sum():
                rec[i, k] = cm[k, k] / cm[k].sum()
    sup = np.bincount(y, minlength=K)
    L = ["# HS1 - rappel par classe d'attaque et par graine", "",
         f"- ensemble : {args.split} ; motif : {args.pattern} ; graines : {seeds} ; seuil : {args.threshold}", "",
         "| Classe | flux | " + " | ".join(f"s{s}" for s in seeds) + " | moyenne | graines >= seuil |",
         "| --- | --- | " + " | ".join(["---"] * len(seeds)) + " | --- | --- |"]
    for k in sorted(range(K), key=lambda k: -np.nanmean(rec[:, k]) if sup[k] else 0):
        if k == names.index("BENIGN") or not sup[k]:
            continue
        L.append(f"| {names[k]} | {sup[k]} | " + " | ".join(f"{v:.3f}" for v in rec[:, k]) + f" | {np.nanmean(rec[:, k]):.4f} | {int((rec[:, k] >= args.threshold).sum())} / {len(seeds)} |")
    rp = Path(f"reports/hs1_classes{args.tag}.md")
    rp.parent.mkdir(exist_ok=True)
    rp.write_text("\n".join(L), encoding="utf-8")
    print(f"Rapport écrit : {rp.resolve()}")


if __name__ == "__main__":
    main()
