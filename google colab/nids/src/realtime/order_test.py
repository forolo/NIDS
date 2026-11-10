"""Diagnostic - Les prédictions du détecteur en ligne dépendent-elles de l'ordre des flux dans la fenêtre ?

Pour chaque fenêtre du jeu (historique réel, cache non modifié pendant les essais, puis fenêtre insérée dans le cache) :
  - passage de référence : ordre d'origine ;
  - répétition : même ordre, deuxième passage (mesure du non-déterminisme du périphérique) ;
  - permutations : --perms ordres aléatoires ; les sorties sont remises dans l'ordre d'origine avant comparaison.
Mesures par rapport à la référence : part des flux dont la classe prédite change, écart maximal des logits (valeur absolue),
et, pour la classe choisie, part des flux de la classe classés exactement dans la classe.
Par défaut seules les fenêtres contenant la classe sont évaluées (--all-windows pour toutes ; le cache est mis à jour pour toutes).

Usage (depuis la racine du projet), à lancer sur cuda puis sur cpu :
    python -m src.realtime.order_test --head models/checkpoints/temporal_tl_f10_s5.pt --class-name DDoS --device cuda --tag _s5_cuda
    python -m src.realtime.order_test --head models/checkpoints/temporal_tl_f10_s5.pt --class-name DDoS --device cpu --tag _s5_cpu
"""
import argparse
import datetime
import json
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph-dir", default="data/graphs")
    ap.add_argument("--head", required=True)
    ap.add_argument("--class-name", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--perms", type=int, default=5)
    ap.add_argument("--all-windows", action="store_true")
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
    det = OnlineDetector(args.head, X.shape[1], args.device)
    rng = np.random.default_rng(args.seed)
    order = np.lexsort((ws, wf))
    rows = []
    for i in order:
        a, b = int(ptr[i]), int(ptr[i + 1])
        file, minute = int(wf[i]), int(ws[i] // 60)
        m = Y[a:b] == cid
        if m.any() or args.all_windows:
            x, s, d = X[a:b], S[a:b], D[a:b]
            p0, _, l0 = det.process(file, minute, x, s, d, want_logits=True, update_cache=False)
            p1, _, l1 = det.process(file, minute, x, s, d, want_logits=True, update_cache=False)
            r = {"file": file, "minute": minute, "flows": b - a, "n": int(m.sum()),
                 "det0": float((p0[m] == cid).mean()) if m.any() else float("nan"),
                 "rep_chg": float((p1 != p0).mean()), "rep_dl": float(np.abs(l1 - l0).max()),
                 "perm_chg": [], "perm_dl": [], "perm_det": []}
            for _ in range(args.perms):
                pm = rng.permutation(b - a)
                pp, _, lp = det.process(file, minute, x[pm], s[pm], d[pm], want_logits=True, update_cache=False)
                pa, la = np.empty_like(pp), np.empty_like(lp)
                pa[pm], la[pm] = pp, lp
                r["perm_chg"].append(float((pa != p0).mean()))
                r["perm_dl"].append(float(np.abs(la - l0).max()))
                r["perm_det"].append(float((pa[m] == cid).mean()) if m.any() else float("nan"))
            rows.append(r)
        det.process(file, minute, X[a:b], S[a:b], D[a:b])
    fmt = lambda mm: datetime.datetime.fromtimestamp(mm * 60, datetime.timezone.utc).strftime("%Y-%m-%d %H:%M")
    L = [f"# Sensibilité à l'ordre des flux dans la fenêtre (classe « {args.class_name} »)", "",
         f"- tête : {args.head} ; ensemble : {args.split} ; {args.perms} permutations ; périphérique : {det.dev}",
         "- « répétition » : même ordre, second passage ; « permutations » : ordres aléatoires, sorties remises dans l'ordre d'origine",
         "- changement : part des flux dont la classe prédite diffère de la référence ; écart logits : maximum de la valeur absolue", "",
         "| fichier | minute (UTC) | flux | flux de la classe | détection de référence | répétition : changement | répétition : écart logits | "
         "permutations : changement max | permutations : écart logits max | permutations : détection min – max |",
         "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in rows:
        L.append(f"| {r['file']} | {fmt(r['minute'])} | {r['flows']} | {r['n']} | {r['det0']:.3f} | {r['rep_chg']:.4f} | {r['rep_dl']:.2e} | "
                 f"{max(r['perm_chg']):.4f} | {max(r['perm_dl']):.2e} | {np.nanmin(r['perm_det']):.3f} – {np.nanmax(r['perm_det']):.3f} |"
                 if args.perms else "")
    L += ["", f"- fenêtres évaluées : {len(rows)} ; fenêtres avec au moins un changement de classe sur permutation : "
          f"{sum(1 for r in rows if args.perms and max(r['perm_chg']) > 0)} ; sur répétition : {sum(1 for r in rows if r['rep_chg'] > 0)}"]
    rp = Path(f"reports/order{args.tag}.md")
    rp.parent.mkdir(exist_ok=True)
    rp.write_text("\n".join(L), encoding="utf-8")
    print(f"Rapport écrit : {rp.resolve()}")


if __name__ == "__main__":
    main()
