"""Étape 6 (analyse) - Vote majoritaire sur plusieurs graines (prédictions déjà sauvegardées, sans entraînement).

Pour chaque flux : classe la plus prédite parmi les graines (égalité : classe prédite par la première des graines à égalité,
dans l'ordre de --seeds). Calcule les mêmes métriques sur les classes d'attaque que aggregate_attack.py, les compare à la
moyenne des graines prises une à une, et donne une courbe de détection binaire : alerte si au moins m graines signalent
le flux comme attaque (m = 1 à nombre de graines).
Attention : ce système exécute plusieurs modèles par fenêtre ; son coût de calcul est multiplié par le nombre de graines.
Numpy seul.

Usage (depuis la racine du projet) :
    python -m src.eval.vote --pattern '_tl_f10_s{}' --seeds 0 1 2 3 4 --min-support 100 --tag _tl_f10 --save-tag _vote5
"""
import argparse
import json
from pathlib import Path

import numpy as np

from src.eval.confusion_analysis import confusion, f1_from


def plurality(P, K):
    S, N = P.shape
    c = np.zeros((N, K), dtype=np.int64)
    for p in P:
        np.add.at(c, (np.arange(N), p), 1)
    tied = c == c.max(1, keepdims=True)
    out = np.full(N, -1, dtype=np.int64)
    for p in P:
        take = (out < 0) & tied[np.arange(N), p]
        out[take] = p[take]
    return out


def metrics(y, p, K, att, big):
    cm = confusion(y, p, K)
    f = {k: f1_from(cm, k) for k in att}
    a = y > 0
    return dict(f1_all=np.mean([f[k][2] for k in att]), rec_all=np.mean([f[k][0] for k in att]),
                f1_big=np.mean([f[k][2] for k in big]), rec_big=np.mean([f[k][0] for k in big]),
                det=float((p[a] != 0).mean()), missed=int((p[a] == 0).sum()),
                fa=float((p[~a] != 0).mean()), fa_n=int((p[~a] != 0).sum()), rec=[f[k][0] for k in att])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph-dir", default="data/graphs")
    ap.add_argument("--emb-dir", default="data/embeddings")
    ap.add_argument("--pattern", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--split", default="test")
    ap.add_argument("--min-support", type=int, default=100)
    ap.add_argument("--save-tag", default="", help="si renseigné, écrit pred{save-tag}_{split}.npy (utilisable par confusion_analysis)")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    gd, ed = Path(args.graph_dir), Path(args.emb_dir)
    meta = json.loads((gd / "meta.json").read_text(encoding="utf-8"))
    names = [k for k, _ in sorted(meta["label_id"].items(), key=lambda t: t[1])]
    K, S = len(names), len(args.seeds)
    y = np.load(gd / f"{args.split}.npz")["y"].astype(np.int64)
    P = np.stack([np.load(ed / f"pred{args.pattern.format(s)}_{args.split}.npy").astype(np.int64) for s in args.seeds])
    sup = np.bincount(y, minlength=K)
    att = [k for k in range(1, K) if sup[k] > 0]
    big = [k for k in att if sup[k] >= args.min_support]

    pv = plurality(P, K)
    single = [metrics(y, P[i], K, att, big) for i in range(S)]
    vote = metrics(y, pv, K, att, big)
    if args.save_tag:
        np.save(ed / f"pred{args.save_tag}_{args.split}.npy", pv.astype(np.int8))

    def row(label, key, fmt="{:.4f}"):
        v = np.array([m[key] for m in single], dtype=float)
        return f"| {label} | {v.mean():.4f} ± {v.std(ddof=1):.4f} | {v.min():.4f} | {v.max():.4f} | {fmt.format(vote[key])} |"

    L = [f"# Vote majoritaire sur {S} graines ({args.split})", "",
         f"- étiquettes : {args.pattern} ; graines : {args.seeds} ; classes d'attaque : {len(att)} ; classes avec au moins {args.min_support} flux : {len(big)}",
         f"- coût de calcul : {S} modèles par fenêtre, soit environ {S} fois celui d'un seul modèle",
         f"- égalité : classe prédite par la première des graines à égalité, dans l'ordre {args.seeds}", "",
         "## Comparaison avec les graines prises une à une", "",
         "| Métrique | graines seules (moyenne ± écart-type) | min | max | vote |", "| --- | --- | --- | --- | --- |",
         row(f"F1 macro, {len(att)} classes", "f1_all"), row(f"rappel macro, {len(att)} classes", "rec_all"),
         row(f"F1 macro, {len(big)} classes (≥ {args.min_support} flux)", "f1_big"), row(f"rappel macro, {len(big)} classes", "rec_big"),
         row("détection binaire", "det"), row("attaques manquées", "missed", "{:d}"), row("fausses alertes (flux)", "fa_n", "{:d}"),
         row("taux de fausses alertes", "fa", "{:.5f}"), "",
         "## Rappel exact par classe d'attaque", "", "| Classe | support | graines seules (moyenne) | min | max | vote |", "| --- | --- | --- | --- | --- | --- |"]
    R = np.array([m["rec"] for m in single])
    L += [f"| {names[k]} | {sup[k]} | {R[:, i].mean():.3f} | {R[:, i].min():.3f} | {R[:, i].max():.3f} | {vote['rec'][i]:.3f} |" for i, k in enumerate(att)]
    a = y > 0
    flag = (P != 0).sum(0)
    L += ["", "## Alerte si au moins m graines signalent le flux comme attaque", "", "| m | attaques manquées | détection binaire | fausses alertes | taux de fausses alertes |", "| --- | --- | --- | --- | --- |"]
    for m in range(1, S + 1):
        al = flag >= m
        L.append(f"| {m} | {int((~al[a]).sum())} | {al[a].mean():.4f} | {int(al[~a].sum())} | {al[~a].mean():.5f} |")
    rp = Path(f"reports/vote{args.tag}.md")
    rp.parent.mkdir(parents=True, exist_ok=True)
    rp.write_text("\n".join(L), encoding="utf-8")
    print(f"Rapport écrit : {rp.resolve()}")


if __name__ == "__main__":
    main()
