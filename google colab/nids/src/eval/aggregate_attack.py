"""Étape 6 (analyse) - Métriques centrées sur les classes d'attaque, agrégées sur plusieurs graines.

Pour chaque graine, à partir de data/embeddings/pred{tag}_{split}.npy (train_temporal.py --save-preds) :
  - F1 macro et rappel macro sur les classes d'attaque présentes (toutes, puis celles ayant au moins --min-support flux) ;
  - détection binaire (flux d'attaque non prédits BENIGN), taux de fausses alertes (BENIGN prédits attaque) ;
  - rappel exact par classe d'attaque.
Puis moyenne, écart-type (ddof=1), min et max sur les graines. Numpy seul.

Usage (depuis la racine du projet) :
    python -m src.eval.aggregate_attack --pattern _tl_f10_s{} --seeds 0 1 2 3 4 --min-support 100 --tag _tl_f10
"""
import argparse
import json
from pathlib import Path

import numpy as np

from src.eval.confusion_analysis import confusion, f1_from


def ms(v):
    v = np.asarray(v, dtype=float)
    return f"{v.mean():.4f} ± {v.std(ddof=1) if len(v) > 1 else float('nan'):.4f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph-dir", default="data/graphs")
    ap.add_argument("--emb-dir", default="data/embeddings")
    ap.add_argument("--pattern", required=True, help="étiquette de prédiction avec {} pour la graine, ex. _tl_f10_s{}")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--split", default="test")
    ap.add_argument("--min-support", type=int, default=100)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    gd, ed = Path(args.graph_dir), Path(args.emb_dir)
    meta = json.loads((gd / "meta.json").read_text(encoding="utf-8"))
    names = [k for k, _ in sorted(meta["label_id"].items(), key=lambda t: t[1])]
    K = len(names)
    y = np.load(gd / f"{args.split}.npz")["y"].astype(np.int64)
    sup = np.bincount(y, minlength=K)
    att = [k for k in range(1, K) if sup[k] > 0]
    big = [k for k in att if sup[k] >= args.min_support]
    small = [k for k in att if k not in big]

    R, REC = [], []
    for s in args.seeds:
        p = np.load(ed / f"pred{args.pattern.format(s)}_{args.split}.npy").astype(np.int64)
        cm = confusion(y, p, K)
        f = {k: f1_from(cm, k) for k in att}
        a = y > 0
        R.append(dict(seed=s,
                      f1_all=np.mean([f[k][2] for k in att]), rec_all=np.mean([f[k][0] for k in att]),
                      f1_big=np.mean([f[k][2] for k in big]), rec_big=np.mean([f[k][0] for k in big]),
                      det=float((p[a] != 0).mean()), missed=int((p[a] == 0).sum()),
                      fa=float((p[~a] != 0).mean()), fa_n=int((p[~a] != 0).sum())))
        REC.append([f[k][0] for k in att])
    REC = np.array(REC)

    col = lambda key: [r[key] for r in R]
    L = [f"# Métriques sur les classes d'attaque, {len(args.seeds)} graines ({args.split})", "",
         f"- étiquettes : {args.pattern} ; graines : {args.seeds} ; classes d'attaque présentes : {len(att)} ; "
         f"classes avec au moins {args.min_support} flux : {len(big)} ; classes rares : "
         + ", ".join(f"{names[k]} ({sup[k]})" for k in small), "",
         "## Résumé (moyenne ± écart-type sur les graines)", "",
         "| Métrique | valeur |", "| --- | --- |",
         f"| F1 macro, {len(att)} classes d'attaque | {ms(col('f1_all'))} |",
         f"| rappel macro, {len(att)} classes d'attaque | {ms(col('rec_all'))} |",
         f"| F1 macro, {len(big)} classes avec au moins {args.min_support} flux | {ms(col('f1_big'))} |",
         f"| rappel macro, {len(big)} classes avec au moins {args.min_support} flux | {ms(col('rec_big'))} |",
         f"| détection binaire | {ms(col('det'))} |",
         f"| taux de fausses alertes | {ms(col('fa'))} |", "",
         "## Par graine", "",
         f"| graine | F1 macro ({len(att)}) | rappel macro ({len(att)}) | F1 macro ({len(big)}) | rappel macro ({len(big)}) | détection binaire | attaques manquées | fausses alertes (taux) |", "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    L += [f"| {r['seed']} | {r['f1_all']:.4f} | {r['rec_all']:.4f} | {r['f1_big']:.4f} | {r['rec_big']:.4f} | {r['det']:.4f} | {r['missed']} | {r['fa_n']} ({r['fa']:.5f}) |" for r in R]
    L += ["", "## Rappel exact par classe d'attaque", "", "| Classe | support | moyenne ± écart-type | min | max |", "| --- | --- | --- | --- | --- |"]
    L += [f"| {names[k]} | {sup[k]} | {ms(REC[:, i])} | {REC[:, i].min():.3f} | {REC[:, i].max():.3f} |" for i, k in enumerate(att)]
    rp = Path(f"reports/attack_seeds{args.tag}.md")
    rp.parent.mkdir(parents=True, exist_ok=True)
    rp.write_text("\n".join(L), encoding="utf-8")
    print(f"Rapport écrit : {rp.resolve()}")


if __name__ == "__main__":
    main()
