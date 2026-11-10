"""HS1 (diagnostic) - Sensibilité de la détection d'une classe au nombre de flux de cette classe dans une fenêtre.

Pour chaque fenêtre du jeu qui contient la classe, on refait passer la fenêtre dans le détecteur en ligne (historique réel, cache
non modifié) avec un nombre n de flux de la classe imposé :
  n inférieur au nombre réel : sous-échantillon aléatoire des flux de la classe (les autres flux ne changent pas) ;
  n supérieur : ajout de flux de la même classe pris dans les autres fenêtres du jeu (extrémités et variables réelles).
Mesure sur les flux de la classe conservés : part classée exactement dans la classe, part classée attaque (non BENIGN).
La ligne « réel » est la fenêtre non modifiée. Moyenne sur --draws tirages.
Deux colonnes supplémentaires séparent le contenu de la fenêtre de l'historique des hôtes :
  « sans historique » : fenêtre réelle passée avec un cache d'historique vide ;
  « flux remplacés » : les flux de la classe sont remplacés (même nombre) par des flux de la classe pris dans d'autres fenêtres,
      le reste de la fenêtre et l'historique restent ceux de la fenêtre réelle ;
  « remplacés, sans historique » : les deux modifications ensemble.
Une dernière section échange le contexte : les flux de la classe d'une fenêtre sont placés dans le reste (flux d'autres classes)
d'une autre fenêtre contenant la classe ; historique de la fenêtre cible conservée. Elle décrit aussi la composition de chaque
fenêtre (étiquettes des autres flux, hôte le plus visé, hôtes distincts).

Usage (depuis la racine du projet) :
    python -m src.realtime.volume_test --head models/checkpoints/temporal_tl_f10_s5.pt --class-name DDoS --tag _ddos_s5
"""
import argparse
import json
from pathlib import Path

import numpy as np


def rescale(X, S, D, Y, a, b, cid, n_t, pool, rng):
    """Retourne (x, s, d, ev) : fenêtre modifiée et positions (dans la fenêtre modifiée) des flux de la classe à évaluer."""
    pos = np.flatnonzero(Y[a:b] == cid)
    x, s, d = X[a:b], S[a:b], D[a:b]
    if n_t <= len(pos):
        drop = rng.choice(pos, len(pos) - n_t, replace=False)
        keep = np.ones(b - a, bool)
        keep[drop] = False
        k = np.flatnonzero(keep)
        return x[k], s[k], d[k], np.flatnonzero(Y[a:b][k] == cid)
    cand = pool[(pool < a) | (pool >= b)]
    sel = rng.choice(cand, n_t - len(pos), replace=len(cand) < n_t - len(pos))
    return np.concatenate([x, X[sel]]), np.concatenate([s, S[sel]]), np.concatenate([d, D[sel]]), pos


def replace_class(X, S, D, Y, a, b, cid, pool, rng):
    """Remplace les flux de la classe de la fenêtre par le même nombre de flux de la classe pris hors de la fenêtre."""
    pos = np.flatnonzero(Y[a:b] == cid)
    cand = pool[(pool < a) | (pool >= b)]
    sel = rng.choice(cand, len(pos), replace=len(cand) < len(pos))
    other = np.setdiff1d(np.arange(b - a), pos)
    x = np.concatenate([X[a:b][other], X[sel]])
    s = np.concatenate([S[a:b][other], S[sel]])
    d = np.concatenate([D[a:b][other], D[sel]])
    return x, s, d, np.arange(len(other), len(other) + len(sel))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph-dir", default="data/graphs")
    ap.add_argument("--head", required=True)
    ap.add_argument("--class-name", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--counts", default="500,1348,2500,4000,6500,9000")
    ap.add_argument("--draws", type=int, default=3)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    from src.realtime.engine import OnlineDetector

    gd = Path(args.graph_dir)
    meta = json.loads((gd / "meta.json").read_text(encoding="utf-8"))
    names = [k for k, _ in sorted(meta["label_id"].items(), key=lambda t: t[1])]
    cid, benign = names.index(args.class_name), names.index("BENIGN")
    z = np.load(gd / f"{args.split}.npz")
    X, S, D, Y = z["edge_attr"], z["src"], z["dst"], z["y"].astype(np.int64)
    ptr, ws, wf = z["win_ptr"], z["win_start"].astype(np.int64), z["win_file"].astype(np.int64)
    pool = np.flatnonzero(Y == cid)
    counts = [int(c) for c in args.counts.split(",") if c]
    det = OnlineDetector(args.head, X.shape[1], args.device)
    rng = np.random.default_rng(args.seed)
    order = np.lexsort((ws, wf))
    res = []
    class_windows = [int(i) for i in order if (Y[int(ptr[i]):int(ptr[i + 1])] == cid).any()]
    ctx, compo = {}, {}
    for i in order:
        a, b = int(ptr[i]), int(ptr[i + 1])
        n_cls = int((Y[a:b] == cid).sum())
        file, minute = int(wf[i]), int(ws[i] // 60)
        if n_cls:
            row = {"file": file, "minute": minute, "flows": b - a, "n": n_cls, "cells": {}}
            p, _ = det.process(file, minute, X[a:b], S[a:b], D[a:b], update_cache=False)
            m = Y[a:b] == cid
            row["real"] = (float((p[m] == cid).mean()), float((p[m] != benign).mean()))
            for n_t in counts:
                ex, de = [], []
                for _ in range(args.draws):
                    x, s, d, ev = rescale(X, S, D, Y, a, b, cid, n_t, pool, rng)
                    p, _ = det.process(file, minute, x, s, d, update_cache=False)
                    ex.append((p[ev] == cid).mean()); de.append((p[ev] != benign).mean())
                row["cells"][n_t] = (float(np.mean(ex)), float(np.mean(de)))
            from src.realtime.history_cache import HostHistory
            saved = det.hist
            det.hist = HostHistory(det.k)
            p, _ = det.process(file, minute, X[a:b], S[a:b], D[a:b], update_cache=False)
            row["nohist"] = (float((p[m] == cid).mean()), float((p[m] != benign).mean()))
            rr, rn = [], []
            for _ in range(args.draws):
                x, s, d, ev = replace_class(X, S, D, Y, a, b, cid, pool, rng)
                det.hist = HostHistory(det.k)
                p, _ = det.process(file, minute, x, s, d, update_cache=False)
                rn.append((p[ev] == cid).mean())
                det.hist = saved
                p, _ = det.process(file, minute, x, s, d, update_cache=False)
                rr.append((p[ev] == cid).mean())
            det.hist = saved
            row["swap"], row["swap_nohist"] = float(np.mean(rr)), float(np.mean(rn))
            res.append(row)
            m_cls = Y[a:b] == cid
            lab, cnt = np.unique(Y[a:b][~m_cls], return_counts=True)
            hosts_d, dcnt = np.unique(D[a:b], return_counts=True)
            compo[(file, minute)] = (", ".join(f"{names[l]} {c}" for l, c in sorted(zip(lab, cnt), key=lambda t: -t[1])[:4]) or "aucun",
                                     int(dcnt.max()), len(np.unique(np.concatenate([S[a:b], D[a:b]]))))
            for j in class_windows:
                if j == int(i):
                    continue
                a2, b2 = int(ptr[j]), int(ptr[j + 1])
                oth = np.flatnonzero(Y[a2:b2] != cid)
                x = np.concatenate([X[a:b][m_cls], X[a2:b2][oth]])
                s = np.concatenate([S[a:b][m_cls], S[a2:b2][oth]])
                d = np.concatenate([D[a:b][m_cls], D[a2:b2][oth]])
                p, _ = det.process(file, minute, x, s, d, update_cache=False)
                ctx[(file, minute, int(ws[j] // 60))] = float((p[:int(m_cls.sum())] == cid).mean())
        det.process(file, minute, X[a:b], S[a:b], D[a:b])
    L = [f"# Sensibilité au nombre de flux de la classe « {args.class_name} » dans la fenêtre", "",
         f"- tête : {args.head} ; ensemble : {args.split} ; {args.draws} tirages ; périphérique : {det.dev}",
         "- cellule : part des flux de la classe classés exactement dans la classe / part classés attaque ; « réel » : fenêtre non modifiée", "",
         "| fichier | minute (UTC) | flux de la fenêtre | flux de la classe | réel | " + " | ".join(f"n = {c}" for c in counts) + " |",
         "| --- | --- | --- | --- | --- | " + " | ".join(["---"] * len(counts)) + " |"]
    import datetime
    for r in res:
        t = datetime.datetime.fromtimestamp(r["minute"] * 60, datetime.timezone.utc).strftime("%Y-%m-%d %H:%M")
        L.append(f"| {r['file']} | {t} | {r['flows']} | {r['n']} | {r['real'][0]:.3f} / {r['real'][1]:.3f} | " +
                 " | ".join(f"{r['cells'][c][0]:.3f} / {r['cells'][c][1]:.3f}" for c in counts) + " |")
    L += ["", "## Contenu de la fenêtre et historique (part classée exactement dans la classe)", "",
          "| minute (UTC) | réel | sans historique | flux de la classe remplacés | remplacés, sans historique |", "| --- | --- | --- | --- | --- |"]
    for r in res:
        t = datetime.datetime.fromtimestamp(r["minute"] * 60, datetime.timezone.utc).strftime("%Y-%m-%d %H:%M")
        L.append(f"| {t} | {r['real'][0]:.3f} | {r['nohist'][0]:.3f} | {r['swap']:.3f} | {r['swap_nohist']:.3f} |")
    mins = [r["minute"] for r in res]
    fmt = lambda mm: datetime.datetime.fromtimestamp(mm * 60, datetime.timezone.utc).strftime("%H:%M")
    L += ["", "## Composition des fenêtres", "",
          "| minute (UTC) | autres flux (étiquette, nombre) | flux vers l'hôte le plus visé | hôtes distincts |", "| --- | --- | --- | --- |"]
    for r in res:
        c = compo[(r["file"], r["minute"])]
        L.append(f"| {fmt(r['minute'])} | {c[0]} | {c[1]} | {c[2]} |")
    L += ["", "## Échange de contexte : flux de la classe de la ligne placés dans le reste de la fenêtre de la colonne (part classée exactement dans la classe)", "",
          "| flux de la fenêtre \\ reste de la fenêtre | " + " | ".join(fmt(mm) for mm in mins) + " |", "| --- | " + " | ".join(["---"] * len(mins)) + " |"]
    for r in res:
        L.append(f"| {fmt(r['minute'])} | " + " | ".join("réel : %.3f" % r["real"][0] if mm == r["minute"] else "%.3f" % ctx[(r["file"], r["minute"], mm)] for mm in mins) + " |")
    L += ["", "## Limites", "",
          "- Les flux ajoutés viennent d'autres fenêtres du même ensemble : leurs extrémités et variables sont réelles, mais pas celles qu'aurait produites l'attaque dans cette minute.",
          "- Seul le nombre de flux de la classe change ; les flux d'autres classes restent identiques. Une seule tête.",
          "- Un résultat sur un jeu de test ne peut pas servir à régler un correctif sans validation séparée."]
    rp = Path(f"reports/volume{args.tag}.md")
    rp.parent.mkdir(exist_ok=True)
    rp.write_text("\n".join(L), encoding="utf-8")
    print(f"Rapport écrit : {rp.resolve()}")


if __name__ == "__main__":
    main()
