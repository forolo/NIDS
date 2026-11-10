"""Étape 6 (affinage) - Diagnostic d'une classe : flux manqués (prédits BENIGN) contre flux détectés (prédits dans la classe).

Trois mesures, sur un ensemble donné (--split, test par défaut), à partir de pred{tag}_{split}.npy :
  1. variables brutes : écart-type, min et max sur les flux de la classe ; nombre de variables à écart-type quasi nul ;
     nombre de vecteurs de variables brutes (arrondis à 4 décimales) présents à la fois chez les manqués et chez les détectés ;
     écarts de moyenne en unités brutes (normalisées) entre manqués et détectés, avec l'écart-type de la classe ;
  2. variables de graphe et de temps (src/models/graph_features.py) : moyenne chez les détectés, les manqués, le reste de la classe
     et un échantillon de BENIGN ;
  3. taux de détection selon le nombre de flux de la classe dans la fenêtre, et selon le nombre de flux de la classe de la même
     source dans la fenêtre.
Numpy seul.

Usage (depuis la racine du projet) :
    python -m src.eval.diagnose_class --pred-tag _tl_f10_s0 --class-name "Bot" --tag _bot
"""
import argparse
import json
from pathlib import Path

import numpy as np

from src.models import graph_features

BINS = [(1, 1), (2, 2), (3, 5), (6, 10), (11, 100), (101, 10**9)]


def rate_by_count(cnt_per_flow, det, mis):
    rows = []
    for lo, hi in BINS:
        m = (cnt_per_flow >= lo) & (cnt_per_flow <= hi)
        n = int(m.sum())
        if n:
            rows.append((f"{lo}" if lo == hi else (f"{lo} à {hi}" if hi < 10**9 else f">= {lo}"), n, int(det[m].sum()), int(mis[m].sum())))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph-dir", default="data/graphs")
    ap.add_argument("--emb-dir", default="data/embeddings")
    ap.add_argument("--pred-tag", required=True)
    ap.add_argument("--class-name", default="Bot")
    ap.add_argument("--split", default="test")
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    gd, ed = Path(args.graph_dir), Path(args.emb_dir)
    meta = json.loads((gd / "meta.json").read_text(encoding="utf-8"))
    names = [k for k, _ in sorted(meta["label_id"].items(), key=lambda t: t[1])]
    k = names.index(args.class_name)
    z = np.load(gd / f"{args.split}.npz")
    y = z["y"].astype(np.int64)
    p = np.load(ed / f"pred{args.pred_tag}_{args.split}.npy").astype(np.int64)
    X = z["edge_attr"].astype(np.float64)
    ptr = z["win_ptr"]
    wid = np.repeat(np.arange(len(ptr) - 1), np.diff(ptr))
    cls = y == k
    det = cls & (p == k)
    mis = cls & (p == 0)
    L = [f"# Diagnostic de la classe « {args.class_name} » ({args.split})", "",
         f"- prédictions : {args.pred_tag} ; flux de la classe : {int(cls.sum())} ; détectés (prédits dans la classe) : {int(det.sum())} ; "
         f"manqués (prédits BENIGN) : {int(mis.sum())} ; classés ailleurs : {int((cls & ~det & ~mis).sum())}", ""]

    # 1. variables brutes
    Xc = X[cls]
    sd = Xc.std(0)
    L += ["## 1. Variables brutes", "",
          f"- variables dont l'écart-type sur la classe est < 0.001 : {int((sd < 1e-3).sum())} sur {X.shape[1]}",
          f"- vecteurs distincts (arrondis à 4 décimales) parmi les flux de la classe : {len(np.unique(np.round(Xc, 4), axis=0))} sur {len(Xc)}"]
    if det.sum() and mis.sum():
        rd, rm = np.round(X[det], 4), np.round(X[mis], 4)
        ud, um = np.unique(rd, axis=0), np.unique(rm, axis=0)
        both = np.unique(np.concatenate([ud, um]), axis=0)
        shared = len(ud) + len(um) - len(both)
        key = lambda a: {tuple(r) for r in a}
        kd = key(rd)
        L += [f"- vecteurs distincts chez les détectés : {len(ud)} ; chez les manqués : {len(um)} ; présents dans les deux groupes : {shared}",
              f"- flux manqués dont le vecteur de variables brutes est identique à celui d'au moins un flux détecté : {sum(tuple(r) in kd for r in rm)} sur {len(rm)}", ""]
        d = np.abs(X[mis].mean(0) - X[det].mean(0))
        order = np.argsort(-d)[:args.top]
        L += ["Variables brutes à plus grand écart de moyenne entre manqués et détectés (unités normalisées) :", "",
              "| Variable | moyenne détectés | moyenne manqués | écart absolu | écart-type de la classe | min de la classe | max de la classe |", "| --- | --- | --- | --- | --- | --- | --- |"]
        L += [f"| {meta['features'][j]} | {X[det][:, j].mean():.4f} | {X[mis][:, j].mean():.4f} | {d[j]:.4f} | {sd[j]:.4f} | {Xc[:, j].min():.3f} | {Xc[:, j].max():.3f} |" for j in order]
    else:
        L.append("- pas assez de flux détectés et manqués pour comparer")

    # 2. variables de graphe
    G = graph_features.compute(z["src"], z["dst"], z["dport"], z["win_ptr"], z["win_start"], z["win_file"]).astype(np.float64)
    ben = np.flatnonzero(y == 0)[::20]
    rest = cls & ~det & ~mis
    L += ["", "## 2. Variables de graphe et de temps (moyennes ; degrés et comptages en log1p)", "",
          f"| Variable | détectés ({int(det.sum())}) | manqués ({int(mis.sum())}) | autres de la classe ({int(rest.sum())}) | BENIGN (1 flux sur 20) |", "| --- | --- | --- | --- | --- |"]
    f = lambda m, j: f"{G[m][:, j].mean():.3f}" if m.sum() else "-"
    for j, n in enumerate(graph_features.NAMES):
        L.append(f"| {n} | {f(det, j)} | {f(mis, j)} | {f(rest, j)} | {G[ben][:, j].mean():.3f} |")

    # 3. taux selon le nombre de flux de la classe
    n_win = np.bincount(wid[cls], minlength=len(ptr) - 1)
    c_win = n_win[wid]
    pair = wid.astype(np.int64) * 10**7 + z["src"].astype(np.int64)
    u, inv = np.unique(pair[cls], return_inverse=True)
    n_src = np.zeros(len(y), dtype=np.int64)
    n_src[np.flatnonzero(cls)] = np.bincount(inv.reshape(-1))[inv.reshape(-1)]
    for title, c in (("par nombre de flux de la classe dans la fenêtre", c_win), ("par nombre de flux de la classe de la même source dans la fenêtre", n_src)):
        L += ["", f"## 3. Détection {title}", "", "| nombre | flux | détectés | manqués (BENIGN) | part détectée |", "| --- | --- | --- | --- | --- |"]
        for lab, n, a, b in rate_by_count(c[cls], det[cls], mis[cls]):
            L.append(f"| {lab} | {n} | {a} | {b} | {a / n:.3f} |")
    rp = Path(f"reports/diagnose{args.tag}.md")
    rp.parent.mkdir(parents=True, exist_ok=True)
    rp.write_text("\n".join(L), encoding="utf-8")
    print(f"Rapport écrit : {rp.resolve()}")


if __name__ == "__main__":
    main()
