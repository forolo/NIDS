"""HS1 (diagnostic) - Quel élément du contexte d'une fenêtre fait varier la détection d'une classe ?

Pour chaque fenêtre contenant la classe, les flux de la classe de la fenêtre (extrémités, variables, historique des hôtes : réels)
sont repassés dans le détecteur en ligne avec un « reste de fenêtre » (flux d'autres classes) modifié. Mesure : part des flux de la
classe classés exactement dans la classe (cache non modifié). Variantes du reste :
  A. reste réel conservé à la fraction f (sous-échantillon aléatoire de ses flux), f = 0 ; 0,25 ; 0,5 ; 1 ;
  B. reste réel + m flux d'autres classes pris dans les autres fenêtres du jeu, m = 1000 ; 3000 ; 6000 ;
  C. reste réel remplacé par un sous-échantillon, de même effectif que le reste réel, du reste d'une autre fenêtre (colonne) :
     l'effectif de flux du contexte est alors identique, seule leur nature (hôtes, structure) change.
Moyenne sur --draws tirages.

Usage (depuis la racine du projet) :
    python -m src.realtime.context_test --head models/checkpoints/temporal_tl_f10_s5.pt --class-name DDoS --tag _ddos_s5
"""
import argparse
import datetime
import json
from pathlib import Path

import numpy as np


def run_rest(det, file, minute, cls, rest, cid):
    """cls, rest : tuples (x, s, d). Retourne la part des flux de la classe classés exactement dans la classe."""
    x = np.concatenate([cls[0], rest[0]])
    s = np.concatenate([cls[1], rest[1]])
    d = np.concatenate([cls[2], rest[2]])
    p, _ = det.process(file, minute, x, s, d, update_cache=False)
    return float((p[:len(cls[0])] == cid).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph-dir", default="data/graphs")
    ap.add_argument("--head", required=True)
    ap.add_argument("--class-name", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--fracs", default="0,0.25,0.5,1")
    ap.add_argument("--extras", default="1000,3000,6000")
    ap.add_argument("--draws", type=int, default=3)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    from src.realtime.engine import OnlineDetector

    gd = Path(args.graph_dir)
    meta = json.loads((gd / "meta.json").read_text(encoding="utf-8"))
    names = [k for k, _ in sorted(meta["label_id"].items(), key=lambda t: t[1])]
    cid = names.index(args.class_name)
    z = np.load(gd / f"{args.split}.npz")
    X, S, D, Y = z["edge_attr"], z["src"], z["dst"], z["y"].astype(np.int64)
    ptr, ws, wf = z["win_ptr"], z["win_start"].astype(np.int64), z["win_file"].astype(np.int64)
    fracs = [float(f) for f in args.fracs.split(",") if f]
    extras = [int(e) for e in args.extras.split(",") if e]
    det = OnlineDetector(args.head, X.shape[1], args.device)
    rng = np.random.default_rng(args.seed)
    order = np.lexsort((ws, wf))
    cw = [int(i) for i in order if (Y[int(ptr[i]):int(ptr[i + 1])] == cid).any()]
    win_of = np.repeat(np.arange(len(ptr) - 1), np.diff(ptr))
    nonc = np.flatnonzero(Y != cid)
    rows = []
    for i in order:
        a, b = int(ptr[i]), int(ptr[i + 1])
        m = Y[a:b] == cid
        if not m.any():
            det.process(int(wf[i]), int(ws[i] // 60), X[a:b], S[a:b], D[a:b])
            continue
        file, minute = int(wf[i]), int(ws[i] // 60)
        cls = (X[a:b][m], S[a:b][m], D[a:b][m])
        oi = np.flatnonzero(~m)
        own = (X[a:b][oi], S[a:b][oi], D[a:b][oi])
        r = {"minute": minute, "n_cls": int(m.sum()), "n_rest": len(oi), "A": {}, "B": {}, "C": {}}
        for f in fracs:
            v = []
            for _ in range(args.draws if 0 < f < 1 else 1):
                k = rng.choice(len(oi), int(round(f * len(oi))), replace=False)
                v.append(run_rest(det, file, minute, cls, (own[0][k], own[1][k], own[2][k]), cid))
            r["A"][f] = float(np.mean(v))
        cand = nonc[win_of[nonc] != i]
        for e in extras:
            v = []
            for _ in range(args.draws):
                sel = rng.choice(cand, e, replace=len(cand) < e)
                v.append(run_rest(det, file, minute, cls, (np.concatenate([own[0], X[sel]]), np.concatenate([own[1], S[sel]]),
                                                            np.concatenate([own[2], D[sel]])), cid))
            r["B"][e] = float(np.mean(v))
        for j in cw:
            if j == int(i):
                continue
            a2, b2 = int(ptr[j]), int(ptr[j + 1])
            oj = np.flatnonzero(Y[a2:b2] != cid)
            v = []
            for _ in range(args.draws):
                k = oj[rng.choice(len(oj), len(oi), replace=len(oj) < len(oi))]
                v.append(run_rest(det, file, minute, cls, (X[a2:b2][k], S[a2:b2][k], D[a2:b2][k]), cid))
            r["C"][int(ws[j] // 60)] = float(np.mean(v))
        rows.append(r)
        det.process(file, minute, X[a:b], S[a:b], D[a:b])
    fmt = lambda mm: datetime.datetime.fromtimestamp(mm * 60, datetime.timezone.utc).strftime("%Y-%m-%d %H:%M")
    mins = [r["minute"] for r in rows]
    L = [f"# Contexte de la fenêtre et détection de la classe « {args.class_name} »", "",
         f"- tête : {args.head} ; ensemble : {args.split} ; {args.draws} tirages ; périphérique : {det.dev}",
         "- cellule : part des flux de la classe classés exactement dans la classe ; flux de la classe et historique des hôtes réels", "",
         "## A. Reste réel de la fenêtre conservé à la fraction f", "",
         "| minute (UTC) | flux de la classe | autres flux (réel) | " + " | ".join(f"f = {f:g}" for f in fracs) + " |",
         "| --- | --- | --- | " + " | ".join(["---"] * len(fracs)) + " |"]
    for r in rows:
        L.append(f"| {fmt(r['minute'])} | {r['n_cls']} | {r['n_rest']} | " + " | ".join(f"{r['A'][f]:.3f}" for f in fracs) + " |")
    L += ["", "## B. Reste réel + m flux d'autres classes pris dans les autres fenêtres", "",
          "| minute (UTC) | " + " | ".join(f"+ {e}" for e in extras) + " |", "| --- | " + " | ".join(["---"] * len(extras)) + " |"]
    for r in rows:
        L.append(f"| {fmt(r['minute'])} | " + " | ".join(f"{r['B'][e]:.3f}" for e in extras) + " |")
    L += ["", "## C. Reste remplacé par un sous-échantillon, de même effectif que le reste réel, du reste d'une autre fenêtre", "",
          "| flux de la fenêtre \\ reste (sous-échantillon) de la fenêtre | " + " | ".join(fmt(mm)[11:] for mm in mins) + " |",
          "| --- | " + " | ".join(["---"] * len(mins)) + " |"]
    for r in rows:
        L.append(f"| {fmt(r['minute'])} | " + " | ".join("—" if mm == r["minute"] else f"{r['C'][mm]:.3f}" for mm in mins) + " |")
    L += ["", "## Limites", "",
          "- Les flux ajoutés ou échangés viennent d'autres fenêtres du même ensemble : extrémités et variables réelles, mais hôtes et structure mélangés (B, C).",
          "- Les flux de la classe gardent l'historique de leur propre fenêtre. Une seule tête.",
          "- Un résultat sur un jeu de test ne peut pas servir à régler un correctif sans validation séparée."]
    rp = Path(f"reports/context{args.tag}.md")
    rp.parent.mkdir(exist_ok=True)
    rp.write_text("\n".join(L), encoding="utf-8")
    print(f"Rapport écrit : {rp.resolve()}")


if __name__ == "__main__":
    main()
