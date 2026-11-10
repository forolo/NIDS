"""Étape 6 (affinage) - Les flux manqués d'une classe (test) existent-ils dans train ?

À partir de pred{tag}_{split}.npy (train_temporal.py --save-preds) :
  1. distance euclidienne (63 variables brutes) de chaque flux manqué du test, et d'un échantillon de flux détectés du test,
     au flux de train le plus proche de la classe et au flux de train BENIGN le plus proche (échantillon de BENIGN) ;
     part des flux dont le voisin le plus proche (classe ou BENIGN échantillonné) est de la classe ;
  2. pour les variables qui séparent le mieux manqués et détectés : part des flux de la classe dans train plus proches de la
     moyenne des manqués que de celle des détectés ;
  3. tableau des fenêtres de la classe dans train, val et test : flux, rappel exact, part des flux dont la source était
     présente la minute précédente (src_vue_prec), degré sortant précédent moyen (log1p).
Numpy seul.

Usage (depuis la racine du projet) :
    python -m src.eval.compare_train_class --pred-tag _tl_f10_s0 --class-name "DoS Slowhttptest" --tag _slowhttp
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.models import graph_features

J_PREV, J_OUT_PREV = graph_features.NAMES.index("src_vue_prec"), graph_features.NAMES.index("deg_sortant_src_prec")


def nearest(A, B, chunk=64):
    """Distance euclidienne de chaque ligne de A à la ligne la plus proche de B."""
    A, B = A.astype(np.float32), B.astype(np.float32)
    b2 = (B * B).sum(1)
    out = np.empty(len(A), dtype=np.float32)
    for i in range(0, len(A), chunk):
        a = A[i:i + chunk]
        d2 = (a * a).sum(1)[:, None] + b2[None, :] - 2.0 * a @ B.T
        out[i:i + chunk] = np.sqrt(np.maximum(d2.min(1), 0.0))
    return out


def q(v):
    return " / ".join(f"{np.quantile(v, t):.3f}" for t in (0.25, 0.5, 0.75)) if len(v) else "-"


def windows(z, y, p, G, k):
    ptr = z["win_ptr"]
    wid = np.repeat(np.arange(len(ptr) - 1), np.diff(ptr))
    idx = np.flatnonzero(y == k)
    rows = []
    for w in np.unique(wid[idx]):
        m = idx[wid[idx] == w]
        ts = datetime.fromtimestamp(int(z["win_start"][w]), tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
        rows.append((int(z["win_file"][w]), ts, len(m), float((p[m] == k).mean()), float(G[m, J_PREV].mean()), float(G[m, J_OUT_PREV].mean())))
    return sorted(rows, key=lambda r: (r[0], r[1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph-dir", default="data/graphs")
    ap.add_argument("--emb-dir", default="data/embeddings")
    ap.add_argument("--pred-tag", required=True)
    ap.add_argument("--class-name", default="DoS Slowhttptest")
    ap.add_argument("--benign-sample", type=int, default=200000)
    ap.add_argument("--detected-sample", type=int, default=300)
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    gd, ed = Path(args.graph_dir), Path(args.emb_dir)
    meta = json.loads((gd / "meta.json").read_text(encoding="utf-8"))
    names = [k for k, _ in sorted(meta["label_id"].items(), key=lambda t: t[1])]
    k = names.index(args.class_name)
    Z = {s: np.load(gd / f"{s}.npz") for s in ("train", "val", "test")}
    Y = {s: Z[s]["y"].astype(np.int64) for s in Z}
    P = {s: np.load(ed / f"pred{args.pred_tag}_{s}.npy").astype(np.int64) for s in Z}
    rng = np.random.default_rng(0)
    Xtr, Xte = Z["train"]["edge_attr"], Z["test"]["edge_attr"]
    tr_cls = Xtr[Y["train"] == k]
    ben_idx = np.flatnonzero(Y["train"] == 0)
    ben = Xtr[rng.choice(ben_idx, size=min(args.benign_sample, len(ben_idx)), replace=False)]
    det = np.flatnonzero((Y["test"] == k) & (P["test"] == k))
    mis = np.flatnonzero((Y["test"] == k) & (P["test"] == 0))
    L = [f"# Classe « {args.class_name} » : flux manqués du test comparés à train", "",
         f"- prédictions : {args.pred_tag} ; flux de la classe dans train : {len(tr_cls)} ; flux BENIGN de train échantillonnés : {len(ben)} sur {len(ben_idx)} ; "
         f"test : {len(det)} détectés, {len(mis)} manqués"]
    if len(tr_cls) == 0 or len(det) == 0 or len(mis) == 0:
        L += ["", "Pas assez de flux pour comparer."]
    else:
        det_s = rng.choice(det, size=min(args.detected_sample, len(det)), replace=False)
        L += ["", "## 1. Distance au plus proche flux de train (variables brutes ; quartiles 25 % / 50 % / 75 %)", "",
              "| Groupe du test | flux | distance au plus proche flux de la classe | distance au plus proche flux BENIGN (échantillon) | part dont le voisin le plus proche est de la classe |", "| --- | --- | --- | --- | --- |"]
        for name, idx in (("détectés (échantillon)", det_s), ("manqués", mis)):
            dc, db = nearest(Xte[idx], tr_cls), nearest(Xte[idx], ben)
            L.append(f"| {name} | {len(idx)} | {q(dc)} | {q(db)} | {(dc < db).mean():.3f} |")
        sd = Xte[Y["test"] == k].std(0) + 1e-9
        d = np.abs(Xte[mis].mean(0) - Xte[det].mean(0)) / sd
        top = np.argsort(-d)[:args.top]
        L += ["", "## 2. Flux de la classe dans train plus proches de la moyenne des manqués que de celle des détectés", "",
              "| Variable | moyenne détectés (test) | moyenne manqués (test) | moyenne classe (train) | part des flux de classe de train plus proches des manqués |", "| --- | --- | --- | --- | --- |"]
        for j in top:
            a, b = Xte[det][:, j].mean(), Xte[mis][:, j].mean()
            share = (np.abs(tr_cls[:, j] - b) < np.abs(tr_cls[:, j] - a)).mean()
            L.append(f"| {meta['features'][j]} | {a:.3f} | {b:.3f} | {tr_cls[:, j].mean():.3f} | {share:.3f} |")
    L += ["", "## 3. Fenêtres de la classe (src_vue_prec : part des flux de la classe dont la source était présente la minute précédente)", ""]
    for s in ("train", "val", "test"):
        z = Z[s]
        G = graph_features.compute(z["src"], z["dst"], z["dport"], z["win_ptr"], z["win_start"], z["win_file"])
        rows = windows(z, Y[s], P[s], G, k)
        L += [f"### {s}", "", "| fichier | minute (UTC) | flux de la classe | rappel exact | src_vue_prec | deg_sortant_src_prec (log1p) |", "| --- | --- | --- | --- | --- | --- |"]
        L += [f"| {r[0]} | {r[1]} | {r[2]} | {r[3]:.3f} | {r[4]:.3f} | {r[5]:.3f} |" for r in rows] or ["| - | - | - | - | - | - |"]
        L.append("")
    rp = Path(f"reports/compare_train{args.tag}.md")
    rp.parent.mkdir(parents=True, exist_ok=True)
    rp.write_text("\n".join(L), encoding="utf-8")
    print(f"Rapport écrit : {rp.resolve()}")


if __name__ == "__main__":
    main()
