"""EF2 (inférence sur nœuds inconnus) - Résultats du test selon la nouveauté des hôtes, sans réentraînement.

Un hôte est « connu » s'il apparaît (source ou destination) dans l'ensemble de référence (--ref train, ou trainval).
Chaque flux du test est classé selon ses deux extrémités :
  connus : source et destination connues ; un inconnu : une seule extrémité inconnue ; deux inconnus : les deux inconnues.
Pour chaque graine, sur les prédictions déjà écrites (data/embeddings/pred{motif}_test.npy) : détection des attaques (non BENIGN),
fausses alarmes, F1 / rappel moyens sur les classes d'attaque présentes dans la strate (et sur celles ayant au moins --min-support flux
dans la strate), puis rappel par classe et par strate. Numpy seul.

Usage (depuis la racine du projet) :
    python -m src.eval.ef2_novelty --pattern '_tl_f10_s{}' --seeds 0,1,2,3,4,5,6,7,8,9 --tag _tl_f10
"""
import argparse
import json
from pathlib import Path

import numpy as np

from src.eval.confusion_analysis import confusion
from src.realtime.replay import attack_summary

STRATA = ["connus", "un inconnu", "deux inconnus"]


def strata_of(src, dst, known):
    k = known[src].astype(int) + known[dst].astype(int)
    return 2 - k  # 0 : connus, 1 : un inconnu, 2 : deux inconnus


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph-dir", default="data/graphs")
    ap.add_argument("--emb-dir", default="data/embeddings")
    ap.add_argument("--pattern", required=True)
    ap.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9")
    ap.add_argument("--ref", choices=["train", "trainval"], default="train")
    ap.add_argument("--min-support", type=int, default=100)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    gd, ed = Path(args.graph_dir), Path(args.emb_dir)
    meta = json.loads((gd / "meta.json").read_text(encoding="utf-8"))
    names = [k for k, _ in sorted(meta["label_id"].items(), key=lambda t: t[1])]
    K, b = len(names), names.index("BENIGN")
    t = np.load(gd / "test.npz")
    y, src, dst = t["y"].astype(np.int64), t["src"], t["dst"]
    sets = ["train"] + (["val"] if args.ref == "trainval" else [])
    ref_hosts = np.unique(np.concatenate([np.concatenate([np.load(gd / f"{s}.npz")["src"], np.load(gd / f"{s}.npz")["dst"]]) for s in sets]))
    n_hosts = int(max(src.max(), dst.max(), ref_hosts.max())) + 1
    known = np.zeros(n_hosts, bool)
    known[ref_hosts] = True
    st = strata_of(src, dst, known)
    test_hosts = np.unique(np.concatenate([src, dst]))
    seeds = [int(s) for s in args.seeds.split(",")]
    preds = [np.load(ed / f"pred{args.pattern.format(s)}_test.npy").astype(np.int64) for s in seeds]
    L = ["# EF2 - résultats du test selon la nouveauté des hôtes (sans réentraînement)", "",
         f"- référence : {'+'.join(sets)} ; motif : {args.pattern} ; graines : {seeds}",
         f"- hôtes du test inconnus de la référence : {int((~known[test_hosts]).sum())} sur {len(test_hosts)}",
         "- strate d'un flux : nombre de ses extrémités (source, destination) absentes de la référence", "",
         "## Effectifs", "", "| strate | flux | flux d'attaque | part des flux du test |", "| --- | --- | --- | --- |"]
    for s in range(3):
        m = st == s
        L.append(f"| {STRATA[s]} | {int(m.sum())} | {int((m & (y != b)).sum())} | {m.mean():.4f} |")
    L += ["", "## Mesures par strate (moyenne ± écart type sur les graines)", "",
          "| strate | détection des attaques | fausses alarmes | rappel moyen (classes présentes) | F1 moyen (classes présentes) | classes présentes | "
          f"F1 moyen (classes ≥ {args.min_support} flux dans la strate) | classes ≥ seuil |", "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    ms = lambda a: f"{np.mean(a):.4f} ± {np.std(a):.4f}"
    for s in range(3):
        m = st == s
        if not m.any():
            continue
        R = [attack_summary(y[m], p[m], names, args.min_support) for p in preds]
        L.append(f"| {STRATA[s]} | {ms([r['det'] for r in R])} | {ms([r['fa'] for r in R])} | {ms([r['rec_att'] for r in R])} | "
                 f"{ms([r['f1_att'] for r in R])} | {R[0]['n_att']} | {ms([r['f1_big'] for r in R]) if R[0]['n_big'] else 'n/a'} | {R[0]['n_big']} |")
    L += ["", "## Rappel par classe et par strate (moyenne sur les graines ; flux de la classe entre parenthèses)", "",
          "| classe | " + " | ".join(STRATA) + " |", "| --- | --- | --- | --- |"]
    sup = np.stack([np.bincount(y[st == s], minlength=K) for s in range(3)])
    rec = np.full((len(preds), 3, K), np.nan)
    for i, p in enumerate(preds):
        for s in range(3):
            m = st == s
            if m.any():
                cm = confusion(y[m], p[m], K)
                for k in range(K):
                    if cm[k].sum():
                        rec[i, s, k] = cm[k, k] / cm[k].sum()
    for k in range(K):
        if k == b or not sup[:, k].sum():
            continue
        L.append(f"| {names[k]} | " + " | ".join(f"{np.nanmean(rec[:, s, k]):.3f} ({sup[s, k]})" if sup[s, k] else "— (0)" for s in range(3)) + " |")
    L += ["", "## Limites", "",
          "- Les strates ne sont pas équilibrées en classes : comparer les rappels classe par classe plutôt que les moyennes globales.",
          "- Le F1 d'une strate utilise les flux de cette strate seulement (précision calculée dans la strate).",
          "- Un hôte « inconnu » de l'entraînement peut apparaître dans la validation (utilisée pour choisir les réglages) : voir --ref trainval.",
          "- Aucun réentraînement : les graines sont celles du pipeline ; l'encodeur n'a pas d'identité d'hôte en entrée (degrés log1p seulement)."]
    rp = Path(f"reports/ef2_novelty{args.tag}.md")
    rp.parent.mkdir(exist_ok=True)
    rp.write_text("\n".join(L), encoding="utf-8")
    print(f"Rapport écrit : {rp.resolve()}")


if __name__ == "__main__":
    main()
