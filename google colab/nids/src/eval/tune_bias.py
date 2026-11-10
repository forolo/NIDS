"""Étape 6 (affinage) - Décalage du seuil BENIGN choisi sur la validation, appliqué au test.

Principe : on retranche delta au logit de BENIGN avant l'argmax. delta > 0 rend la tête plus sensible
(moins d'attaques classées BENIGN, plus de fausses alertes). Un seul paramètre, aucun réentraînement.

Sélection de delta : sur les flux ÉTIQUETÉS de validation seulement (même tirage que l'entraînement :
--label-fraction, --seed + 1), en maximisant le F1 macro des classes d'attaque sous la contrainte
« taux de fausses alertes sur BENIGN <= --max-fpr ». Le test n'intervient pas dans le choix ; sa courbe
est rapportée pour contrôle.

Entrées : data/embeddings/logit{tag}_{val,test}.npy (train_temporal.py --save-logits) ; data/graphs.
Numpy seul.

Usage (depuis la racine du projet) :
    python -m src.eval.tune_bias --logit-tag _tl_f10_s0 --label-fraction 0.1 --seed 0 --max-fpr 0.001
"""
import argparse
import json
from pathlib import Path

import numpy as np

from src.eval.confusion_analysis import confusion, f1_from
from src.models.baseline_xgb import stratified_mask


def metrics(y, logit, delta, K):
    lg = logit.copy()
    lg[:, 0] -= delta
    p = lg.argmax(1)
    cm = confusion(y, p, K)
    att = [k for k in range(1, K) if cm[k].sum() > 0]
    f1 = float(np.mean([f1_from(cm, k)[2] for k in att]))
    rec = float(np.mean([f1_from(cm, k)[0] for k in att]))
    missed = int(sum(cm[k, 0] for k in att))
    tot = int(sum(cm[k].sum() for k in att))
    fp = int(cm[0, 1:].sum())
    return {"f1": f1, "rec": rec, "det": 1 - missed / tot, "missed": missed, "fp": fp,
            "fpr": fp / cm[0].sum(), "cm": cm}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph-dir", default="data/graphs")
    ap.add_argument("--emb-dir", default="data/embeddings")
    ap.add_argument("--logit-tag", required=True)
    ap.add_argument("--label-fraction", type=float, default=0.1)
    ap.add_argument("--min-labels-per-class", type=int, default=0,
                    help="plancher d'étiquettes par classe (0 = protocole d'origine : seulement ceil(fraction x effectif))")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-fpr", type=float, default=0.001)
    ap.add_argument("--delta-max", type=float, default=6.0)
    ap.add_argument("--delta-step", type=float, default=0.25)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    gd, ed = Path(args.graph_dir), Path(args.emb_dir)
    meta = json.loads((gd / "meta.json").read_text(encoding="utf-8"))
    names = [k for k, _ in sorted(meta["label_id"].items(), key=lambda t: t[1])]
    K = len(names)
    yv = np.load(gd / "val.npz")["y"].astype(np.int64)
    yt = np.load(gd / "test.npz")["y"].astype(np.int64)
    lv = np.load(ed / f"logit{args.logit_tag}_val.npy")
    lt = np.load(ed / f"logit{args.logit_tag}_test.npy")
    lab = stratified_mask(yv, args.label_fraction, args.seed + 1, min_per_class=args.min_labels_per_class)
    deltas = np.arange(0.0, args.delta_max + 1e-9, args.delta_step)

    rows, best = [], None
    for dl in deltas:
        mv = metrics(yv[lab], lv[lab], dl, K)
        mt = metrics(yt, lt, dl, K)
        rows.append((dl, mv, mt))
        if mv["fpr"] <= args.max_fpr and (best is None or mv["f1"] > best[1]["f1"] + 1e-12):
            best = (dl, mv, mt)
    if best is None:
        best = rows[0]
    dl0, mv0, mt0 = rows[0][0], rows[0][1], rows[0][2]
    dls, mvs, mts = best

    L = ["# Décalage du seuil BENIGN choisi sur la validation", "",
         f"- logits : {args.logit_tag} ; flux étiquetés de validation utilisés : {int(lab.sum())} (fraction {args.label_fraction}, graine {args.seed + 1}) ; contrainte : fausses alertes <= {args.max_fpr} sur la validation étiquetée",
         f"- delta choisi sur la validation : {dls:.2f} (0 = tête inchangée)", "",
         "## Résultat sur le test", "",
         "| | delta | F1 macro attaques | rappel macro attaques | détection binaire | attaques manquées | fausses alertes | taux de fausses alertes |",
         "| --- | --- | --- | --- | --- | --- | --- | --- |",
         f"| tête inchangée | 0.00 | {mt0['f1']:.4f} | {mt0['rec']:.4f} | {mt0['det']:.4f} | {mt0['missed']} | {mt0['fp']} | {mt0['fpr']:.5f} |",
         f"| delta choisi | {dls:.2f} | {mts['f1']:.4f} | {mts['rec']:.4f} | {mts['det']:.4f} | {mts['missed']} | {mts['fp']} | {mts['fpr']:.5f} |", "",
         "## Rappel exact par classe d'attaque sur le test (tête inchangée -> delta choisi)", "", "| Classe | support | avant | après | manquée avant | manquée après |", "| --- | --- | --- | --- | --- | --- |"]
    for k in range(1, K):
        sup = mt0["cm"][k].sum()
        if sup:
            L.append(f"| {names[k]} | {sup} | {mt0['cm'][k, k] / sup:.3f} | {mts['cm'][k, k] / sup:.3f} | {mt0['cm'][k, 0] / sup:.3f} | {mts['cm'][k, 0] / sup:.3f} |")
    L += ["", "## Courbe complète (validation étiquetée et test)", "",
          "| delta | val F1 att. | val fausses alertes (taux) | val détection | test F1 att. | test fausses alertes (taux) | test détection | test manquées |",
          "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for dl, mv, mt in rows:
        L.append(f"| {dl:.2f} | {mv['f1']:.4f} | {mv['fpr']:.5f} | {mv['det']:.4f} | {mt['f1']:.4f} | {mt['fpr']:.5f} | {mt['det']:.4f} | {mt['missed']} |")
    rp = Path(f"reports/tune_bias{args.tag}.md")
    rp.parent.mkdir(parents=True, exist_ok=True)
    rp.write_text("\n".join(L), encoding="utf-8")
    print(f"Rapport écrit : {rp.resolve()} ; delta choisi : {dls:.2f}")


if __name__ == "__main__":
    main()
