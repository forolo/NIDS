"""Étape 6 (analyse) - Rappel par classe d'attaque de plusieurs prédictions côte à côte.

Pour chaque étiquette de prédiction (data/embeddings/pred{étiquette}_{ensemble}.npy), donne le rappel exact par classe d'attaque
et les résumés F1 macro / rappel macro (toutes les classes d'attaque, puis celles ayant au moins --min-support flux).
Numpy seul.

Usage (depuis la racine du projet) :
    python -m src.eval.compare_runs --tags _tl_f10_s0 _tl_f10_s2 _dec_e2_h0 --labels e0h0 e2h2 e2h0 --tag _dec
"""
import argparse
import json
from pathlib import Path

import numpy as np

from src.eval.confusion_analysis import confusion, f1_from


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph-dir", default="data/graphs")
    ap.add_argument("--emb-dir", default="data/embeddings")
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--labels", nargs="+", default=None)
    ap.add_argument("--split", default="test")
    ap.add_argument("--min-support", type=int, default=100)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    labels = args.labels or args.tags
    if len(labels) != len(args.tags):
        raise SystemExit("--labels doit avoir autant d'éléments que --tags")

    gd, ed = Path(args.graph_dir), Path(args.emb_dir)
    meta = json.loads((gd / "meta.json").read_text(encoding="utf-8"))
    names = [k for k, _ in sorted(meta["label_id"].items(), key=lambda t: t[1])]
    K = len(names)
    y = np.load(gd / f"{args.split}.npz")["y"].astype(np.int64)
    sup = np.bincount(y, minlength=K)
    att = [k for k in range(1, K) if sup[k] > 0]
    big = [k for k in att if sup[k] >= args.min_support]
    F = []
    for t in args.tags:
        p = np.load(ed / f"pred{t}_{args.split}.npy").astype(np.int64)
        cm = confusion(y, p, K)
        F.append({k: f1_from(cm, k) for k in att})

    head = "| " + " | ".join(["Classe", "support"] + labels) + " |"
    sep = "| " + " | ".join(["---"] * (2 + len(labels))) + " |"
    L = [f"# Comparaison de {len(args.tags)} prédictions ({args.split})", "",
         "- " + " ; ".join(f"{l} = {t}" for l, t in zip(labels, args.tags)), "",
         "## Résumés", "", "| Métrique | " + " | ".join(labels) + " |", "| --- | " + " | ".join("---" for _ in labels) + " |"]
    for name, ks, i in ((f"F1 macro, {len(att)} classes", att, 2), (f"rappel macro, {len(att)} classes", att, 0),
                        (f"F1 macro, {len(big)} classes (≥ {args.min_support} flux)", big, 2), (f"rappel macro, {len(big)} classes", big, 0)):
        L.append(f"| {name} | " + " | ".join(f"{np.mean([f[k][i] for k in ks]):.4f}" for f in F) + " |")
    L += ["", "## Rappel exact par classe d'attaque", "", head, sep]
    L += [f"| {names[k]} | {sup[k]} | " + " | ".join(f"{f[k][0]:.3f}" for f in F) + " |" for k in att]
    rp = Path(f"reports/compare_runs{args.tag}.md")
    rp.parent.mkdir(parents=True, exist_ok=True)
    rp.write_text("\n".join(L), encoding="utf-8")
    print(f"Rapport écrit : {rp.resolve()}")


if __name__ == "__main__":
    main()
