"""Étape 6 (affinage) - Où et pourquoi une classe d'attaque est manquée.

Pour une classe donnée (--class-name), à partir des prédictions de la tête (pred{tag}_{ensemble}.npy,
train_temporal.py --save-preds) et des graphes :
  - par ensemble (train, val, test) : nombre de fenêtres où la classe apparaît, flux par fenêtre, rappel ;
  - pour val et test : tableau des fenêtres (fichier, minute UTC, flux de la classe, détectés, manqués vers BENIGN,
    classés ailleurs) ;
  - sur le test : les caractéristiques qui séparent le plus les flux manqués des flux détectés (différence de moyennes
    divisée par l'écart-type de la classe), avec la moyenne des flux BENIGN en repère.
Numpy seul.

Usage (depuis la racine du projet) :
    python -m src.eval.inspect_class --pred-tag _tl_f10_s0 --class-name "DoS Slowhttptest"
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def window_table(z, y, p, k):
    ptr = z["win_ptr"]
    wid = np.repeat(np.arange(len(ptr) - 1), np.diff(ptr))
    idx = np.flatnonzero(y == k)
    rows = []
    for w in np.unique(wid[idx]):
        m = idx[wid[idx] == w]
        a, b = ptr[w], ptr[w + 1]
        ts = datetime.fromtimestamp(int(z["win_start"][w]), tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
        rows.append((int(z["win_file"][w]), ts, int(b - a), len(m), int((p[m] == k).sum()),
                     int((p[m] == 0).sum()), int(((p[m] != k) & (p[m] != 0)).sum())))
    return sorted(rows, key=lambda r: (r[0], r[1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph-dir", default="data/graphs")
    ap.add_argument("--emb-dir", default="data/embeddings")
    ap.add_argument("--pred-tag", required=True)
    ap.add_argument("--class-name", default="DoS Slowhttptest")
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    gd, ed = Path(args.graph_dir), Path(args.emb_dir)
    meta = json.loads((gd / "meta.json").read_text(encoding="utf-8"))
    names = [k for k, _ in sorted(meta["label_id"].items(), key=lambda t: t[1])]
    k = names.index(args.class_name)
    Z = {s: np.load(gd / f"{s}.npz") for s in ("train", "val", "test")}
    Y = {s: Z[s]["y"].astype(np.int64) for s in Z}
    P = {s: np.load(ed / f"pred{args.pred_tag}_{s}.npy").astype(np.int64) for s in Z}

    L = [f"# Classe « {args.class_name} » : où elle est détectée ou manquée", "", f"- prédictions : {args.pred_tag}", "",
         "## Résumé par ensemble", "", "| Ensemble | flux | fenêtres | flux par fenêtre (médiane) | rappel exact | manqués vers BENIGN | classés ailleurs |", "| --- | --- | --- | --- | --- | --- | --- |"]
    tabs = {}
    for s in ("train", "val", "test"):
        tabs[s] = window_table(Z[s], Y[s], P[s], k)
        m = Y[s] == k
        n = int(m.sum())
        if n == 0:
            L.append(f"| {s} | 0 | 0 | - | - | - | - |")
            continue
        pm = P[s][m]
        L.append(f"| {s} | {n} | {len(tabs[s])} | {int(np.median([r[3] for r in tabs[s]]))} | {(pm == k).mean():.3f} | {(pm == 0).mean():.3f} | {((pm != k) & (pm != 0)).mean():.3f} |")
    for s in ("val", "test"):
        L += ["", f"## Fenêtres de {s} contenant la classe", "",
              "| fichier | minute (UTC) | flux de la fenêtre | flux de la classe | détectés | manqués (BENIGN) | classés ailleurs |", "| --- | --- | --- | --- | --- | --- | --- |"]
        L += [f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]} | {r[5]} | {r[6]} |" for r in tabs[s]] or ["| - | - | - | - | - | - | - |"]

    X = Z["test"]["edge_attr"]
    m_det = (Y["test"] == k) & (P["test"] == k)
    m_mis = (Y["test"] == k) & (P["test"] == 0)
    L += ["", "## Caractéristiques qui séparent les flux manqués des flux détectés (test)", ""]
    if m_det.sum() > 1 and m_mis.sum() > 1:
        cls = X[Y["test"] == k]
        sd = cls.std(0) + 1e-9
        d = (X[m_mis].mean(0) - X[m_det].mean(0)) / sd
        ben = X[Y["test"] == 0][::20].mean(0)
        order = np.argsort(-np.abs(d))[:args.top]
        L += [f"flux détectés : {int(m_det.sum())} ; flux manqués : {int(m_mis.sum())}", "",
              "| Caractéristique | moyenne détectés | moyenne manqués | moyenne BENIGN | écart standardisé |", "| --- | --- | --- | --- | --- |"]
        L += [f"| {meta['features'][j]} | {X[m_det][:, j].mean():.3f} | {X[m_mis][:, j].mean():.3f} | {ben[j]:.3f} | {d[j]:+.2f} |" for j in order]
    else:
        L.append("Pas assez de flux détectés et manqués pour comparer.")
    rp = Path(f"reports/inspect{args.tag}.md")
    rp.parent.mkdir(parents=True, exist_ok=True)
    rp.write_text("\n".join(L), encoding="utf-8")
    print(f"Rapport écrit : {rp.resolve()}")


if __name__ == "__main__":
    main()
