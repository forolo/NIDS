"""Étape 6 (analyse) - Erreurs systématiques ou dues à la graine ?

À partir des prédictions de plusieurs graines (data/embeddings/pred{motif}_{ensemble}.npy), pour chaque flux du test :
  1. par classe d'attaque : nombre de flux classés correctement par toutes les graines, par aucune, par une partie ;
     parmi ceux que toutes les graines ratent, combien sont prédits BENIGN par toutes ;
  2. fenêtres où se concentrent les flux ratés par toutes les graines (classe, fichier, minute, flux, classe prédite dominante) ;
  3. flux BENIGN signalés comme attaque : nombre de graines qui les signalent ; pour ceux signalés par au moins --min-seeds
     graines : classe prédite dominante, fenêtres et ports de destination les plus fréquents (avec la part du port dans BENIGN).
Numpy seul.

Usage (depuis la racine du projet) :
    python -m src.eval.stability --pattern _tl_f10_s{} --seeds 0 1 2 3 4 --min-seeds 3 --tag _tl_f10
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def dominant(P, idx, K, exclude_benign=False):
    """Classe prédite la plus fréquente (sur graines) pour chaque flux de idx."""
    c = np.zeros((len(idx), K), dtype=np.int64)
    for p in P:
        np.add.at(c, (np.arange(len(idx)), p[idx]), 1)
    if exclude_benign:
        c[:, 0] = -1
    return c.argmax(1)


def minute(z, w):
    return datetime.fromtimestamp(int(z["win_start"][w]), tz=timezone.utc).strftime("%Y-%m-%d %H:%M")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph-dir", default="data/graphs")
    ap.add_argument("--emb-dir", default="data/embeddings")
    ap.add_argument("--pattern", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--split", default="test")
    ap.add_argument("--min-seeds", type=int, default=3)
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    gd, ed = Path(args.graph_dir), Path(args.emb_dir)
    meta = json.loads((gd / "meta.json").read_text(encoding="utf-8"))
    names = [k for k, _ in sorted(meta["label_id"].items(), key=lambda t: t[1])]
    K, S = len(names), len(args.seeds)
    z = np.load(gd / f"{args.split}.npz")
    y = z["y"].astype(np.int64)
    P = np.stack([np.load(ed / f"pred{args.pattern.format(s)}_{args.split}.npy").astype(np.int64) for s in args.seeds])
    ptr = z["win_ptr"]
    wid = np.repeat(np.arange(len(ptr) - 1), np.diff(ptr))
    dport = z["dport"].astype(np.int64)
    n_ok = (P == y[None, :]).sum(0)
    L = [f"# Stabilité des erreurs sur {S} graines ({args.split})", "", f"- étiquettes : {args.pattern} ; graines : {args.seeds}", "",
         "## 1. Par classe d'attaque : combien de graines classent bien chaque flux", "",
         f"| Classe | support | bien classés par les {S} graines | par aucune graine | par une partie | ratés par toutes, prédits BENIGN par toutes |", "| --- | --- | --- | --- | --- | --- |"]
    wrong_all = {}
    for k in range(1, K):
        idx = np.flatnonzero(y == k)
        if not len(idx):
            continue
        a, b = int((n_ok[idx] == S).sum()), int((n_ok[idx] == 0).sum())
        wrong = idx[n_ok[idx] == 0]
        wrong_all[k] = wrong
        allb = int((P[:, wrong] == 0).all(0).sum()) if len(wrong) else 0
        L.append(f"| {names[k]} | {len(idx)} | {a} | {b} | {len(idx) - a - b} | {allb} |")

    rows = []
    for k, wrong in wrong_all.items():
        if not len(wrong):
            continue
        dom = dominant(P, wrong, K)
        for w in np.unique(wid[wrong]):
            m = wid[wrong] == w
            tot = int(((y == k) & (wid == w)).sum())
            vals, cnt = np.unique(dom[m], return_counts=True)
            rows.append((int(m.sum()), names[k], int(z["win_file"][w]), minute(z, w), tot, names[vals[cnt.argmax()]]))
    rows.sort(key=lambda r: -r[0])
    L += ["", "## 2. Fenêtres où se concentrent les flux ratés par toutes les graines", "",
          "| Classe | fichier | minute (UTC) | flux ratés par toutes | flux de la classe dans la fenêtre | classe prédite dominante |", "| --- | --- | --- | --- | --- | --- |"]
    L += [f"| {r[1]} | {r[2]} | {r[3]} | {r[0]} | {r[4]} | {r[5]} |" for r in rows[:args.top]] or ["| - | - | - | - | - | - |"]

    ben = np.flatnonzero(y == 0)
    n_flag = (P[:, ben] != 0).sum(0)
    L += ["", f"## 3. Flux BENIGN signalés comme attaque (BENIGN : {len(ben)} flux)", "",
          "| Nombre de graines qui signalent le flux | flux |", "| --- | --- |"]
    L += [f"| {j} | {int((n_flag == j).sum())} |" for j in range(1, S + 1)]
    sel = ben[n_flag >= args.min_seeds]
    L += ["", f"Flux signalés par au moins {args.min_seeds} graines : {len(sel)}"]
    if len(sel):
        dom = dominant(P, sel, K, exclude_benign=True)
        v, c = np.unique(dom, return_counts=True)
        L += ["", "Classe prédite dominante :", "", "| Classe | flux |", "| --- | --- |"] + [f"| {names[i]} | {n} |" for i, n in sorted(zip(v, c), key=lambda t: -t[1])]
        v, c = np.unique(wid[sel], return_counts=True)
        o = np.argsort(-c)[:args.top]
        L += ["", "Fenêtres les plus concernées :", "", "| fichier | minute (UTC) | flux signalés | flux de la fenêtre |", "| --- | --- | --- | --- |"]
        L += [f"| {int(z['win_file'][v[i]])} | {minute(z, v[i])} | {c[i]} | {int(ptr[v[i] + 1] - ptr[v[i]])} |" for i in o]
        v, c = np.unique(dport[sel], return_counts=True)
        o = np.argsort(-c)[:args.top]
        base = np.bincount(dport[ben], minlength=int(dport.max()) + 1)
        L += ["", "Ports de destination les plus fréquents :", "", "| port | flux signalés | part des flux signalés | part de ce port dans BENIGN |", "| --- | --- | --- | --- |"]
        L += [f"| {v[i]} | {c[i]} | {c[i] / len(sel):.3f} | {base[v[i]] / len(ben):.4f} |" for i in o]
    rp = Path(f"reports/stability{args.tag}.md")
    rp.parent.mkdir(parents=True, exist_ok=True)
    rp.write_text("\n".join(L), encoding="utf-8")
    print(f"Rapport écrit : {rp.resolve()}")


if __name__ == "__main__":
    main()
