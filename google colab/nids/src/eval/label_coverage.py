"""Étape 6 (analyse) - Les flux étiquetés tirés par chaque graine expliquent-ils les résultats par classe ?

Reproduit le tirage d'étiquettes de l'entraînement (stratified_mask(y_train, fraction, graine), identique à celui de
finetune.py et train_temporal.py) et, pour chaque classe d'attaque et chaque graine, donne :
  - le nombre de flux étiquetés dans train et le nombre de fenêtres de train de la classe qui en contiennent au moins un ;
  - le rappel exact de la classe sur le test (pred{motif}_{ensemble}.npy).
Pour les classes de --detail, donne aussi, fenêtre de train par fenêtre de train, le nombre de flux étiquetés par graine.
Numpy seul. Les fenêtres couvertes ne prouvent pas une cause : le préentraînement GCL et l'initialisation dépendent
aussi de la graine.

Usage (depuis la racine du projet) :
    python -m src.eval.label_coverage --pattern '_tl_f10_s{}' --seeds 0 1 2 3 4 --label-fraction 0.1 --tag _tl_f10
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.models.baseline_xgb import stratified_mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph-dir", default="data/graphs")
    ap.add_argument("--emb-dir", default="data/embeddings")
    ap.add_argument("--pattern", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--label-fraction", type=float, default=0.1)
    ap.add_argument("--min-labels-per-class", type=int, default=0,
                    help="plancher d'étiquettes par classe (0 = protocole d'origine : seulement ceil(fraction x effectif))")
    ap.add_argument("--detail", nargs="+", default=["DoS slowloris", "Web Attack - XSS", "SSH-Patator"])
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    gd, ed = Path(args.graph_dir), Path(args.emb_dir)
    meta = json.loads((gd / "meta.json").read_text(encoding="utf-8"))
    names = [k for k, _ in sorted(meta["label_id"].items(), key=lambda t: t[1])]
    K = len(names)
    ztr, zte = np.load(gd / "train.npz"), np.load(gd / "test.npz")
    ytr, yte = ztr["y"].astype(np.int64), zte["y"].astype(np.int64)
    ptr = ztr["win_ptr"]
    wid = np.repeat(np.arange(len(ptr) - 1), np.diff(ptr))
    masks = {s: stratified_mask(ytr, args.label_fraction, s, min_per_class=args.min_labels_per_class) for s in args.seeds}
    preds = {s: np.load(ed / f"pred{args.pattern.format(s)}_test.npy").astype(np.int64) for s in args.seeds}

    L = [f"# Flux étiquetés tirés par chaque graine (fraction {args.label_fraction}) et rappel de test", "",
         f"- graines : {args.seeds} ; flux de train étiquetés par graine : " + ", ".join(f"s{s} {int(masks[s].sum())}" for s in args.seeds), "",
         "## Par classe d'attaque", "",
         "| Classe | graine | flux de train de la classe | fenêtres de train de la classe | flux étiquetés | fenêtres avec au moins un flux étiqueté | rappel exact sur le test |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for k in range(1, K):
        itr, ite = np.flatnonzero(ytr == k), np.flatnonzero(yte == k)
        if not len(itr):
            continue
        nw = len(np.unique(wid[itr]))
        for s in args.seeds:
            lab = itr[masks[s][itr]]
            rec = f"{(preds[s][ite] == k).mean():.3f}" if len(ite) else "-"
            L.append(f"| {names[k]} | {s} | {len(itr)} | {nw} | {len(lab)} | {len(np.unique(wid[lab]))} | {rec} |")

    for cname in args.detail:
        if cname not in names:
            continue
        k = names.index(cname)
        itr = np.flatnonzero(ytr == k)
        if not len(itr):
            continue
        L += ["", f"## Détail par fenêtre de train : {cname}", "",
              "| fichier | minute (UTC) | flux de la classe | " + " | ".join(f"étiquetés s{s}" for s in args.seeds) + " |",
              "| --- | --- | --- | " + " | ".join("---" for _ in args.seeds) + " |"]
        rows = []
        for w in np.unique(wid[itr]):
            m = itr[wid[itr] == w]
            ts = datetime.fromtimestamp(int(ztr["win_start"][w]), tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
            rows.append((int(ztr["win_file"][w]), ts, len(m), [int(masks[s][m].sum()) for s in args.seeds]))
        rows.sort(key=lambda r: (r[0], r[1]))
        L += [f"| {r[0]} | {r[1]} | {r[2]} | " + " | ".join(str(c) for c in r[3]) + " |" for r in rows]
    rp = Path(f"reports/label_coverage{args.tag}.md")
    rp.parent.mkdir(parents=True, exist_ok=True)
    rp.write_text("\n".join(L), encoding="utf-8")
    print(f"Rapport écrit : {rp.resolve()}")


if __name__ == "__main__":
    main()
