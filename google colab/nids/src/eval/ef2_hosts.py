"""EF2 (inférence sur nœuds inconnus) - Deux mesures complémentaires.

A. Recouvrement des hôtes par classe d'attaque (numpy) : pour chaque classe, nombre d'hôtes distincts (source ou destination) parmi ses flux
   dans l'entraînement, la validation et le test, et part des flux du test de la classe dont les deux extrémités sont des hôtes de l'entraînement.
B. Test de ré-étiquetage (détecteur en ligne) : tous les hôtes du test reçoivent un nouvel identifiant, pris hors de l'ensemble des
   identifiants de l'entraînement, de la validation et du test (permutation aléatoire, cohérente d'une minute à l'autre). Le test est rejoué
   fenêtre par fenêtre ; les prédictions sont comparées à celles de l'évaluation par lots (--pred-file) et au rejeu sans ré-étiquetage.
   Si le résultat dépend de l'identité des hôtes (et non de leur comportement), les prédictions changent.

Usage (depuis la racine du projet) :
    python -m src.eval.ef2_hosts --head models/checkpoints/temporal_tl_f10_s0.pt --pred-file data/embeddings/pred_tl_f10_s0_test.npy --tag _s0
"""
import argparse
import json
from pathlib import Path

import numpy as np

from src.realtime.replay import attack_summary


def overlap_table(gd, names):
    S = {s: np.load(gd / f"{s}.npz") for s in ("train", "val", "test")}
    y = {s: S[s]["y"].astype(np.int64) for s in S}
    b = names.index("BENIGN")
    hosts = lambda s, m: np.unique(np.concatenate([S[s]["src"][m], S[s]["dst"][m]]))
    tr = hosts("train", np.ones(len(y["train"]), bool))
    L = ["## A. Hôtes par classe d'attaque", "",
         "| classe | flux du test | hôtes (entraînement) | hôtes (validation) | hôtes (test) | hôtes du test hors entraînement | flux du test dont les 2 extrémités sont dans l'entraînement |",
         "| --- | --- | --- | --- | --- | --- | --- |"]
    for k, nm in enumerate(names):
        if k == b or not (y["test"] == k).any():
            continue
        h = {s: hosts(s, y[s] == k) for s in S}
        m = y["test"] == k
        both = np.isin(S["test"]["src"][m], tr) & np.isin(S["test"]["dst"][m], tr)
        L.append(f"| {nm} | {int(m.sum())} | {len(h['train'])} | {len(h['val'])} | {len(h['test'])} | {int((~np.isin(h['test'], tr)).sum())} | {both.mean():.4f} |")
    return L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph-dir", default="data/graphs")
    ap.add_argument("--head", required=True)
    ap.add_argument("--pred-file", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--min-support", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-windows", type=int, default=0)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    from src.realtime.engine import OnlineDetector

    gd = Path(args.graph_dir)
    meta = json.loads((gd / "meta.json").read_text(encoding="utf-8"))
    names = [k for k, _ in sorted(meta["label_id"].items(), key=lambda t: t[1])]
    L = ["# EF2 - hôtes inconnus", ""] + overlap_table(gd, names)
    z = np.load(gd / "test.npz")
    X, S, D, Y = z["edge_attr"], z["src"], z["dst"], z["y"].astype(np.int64)
    ptr, ws, wf = z["win_ptr"], z["win_start"].astype(np.int64), z["win_file"].astype(np.int64)
    allids = np.concatenate([np.concatenate([np.load(gd / f"{s}.npz")[c] for c in ("src", "dst")]) for s in ("train", "val", "test")])
    base = int(allids.max()) + 1
    th = np.unique(np.concatenate([S, D]))
    new = base + np.random.default_rng(args.seed).permutation(len(th))
    S2, D2 = new[np.searchsorted(th, S)], new[np.searchsorted(th, D)]
    order = np.lexsort((ws, wf))
    if args.max_windows:
        order = order[:args.max_windows]
    preds = {}
    for tag, (s_, d_) in (("original", (S, D)), ("relabel", (S2, D2))):
        det = OnlineDetector(args.head, X.shape[1], args.device)
        p = np.full(len(Y), -1, np.int64)
        for i in order:
            a, b = int(ptr[i]), int(ptr[i + 1])
            p[a:b], _ = det.process(int(wf[i]), int(ws[i] // 60), X[a:b], s_[a:b], d_[a:b])
        preds[tag] = p
    done = preds["original"] >= 0
    L += ["", "## B. Test de ré-étiquetage des hôtes du test", "",
          f"- tête : {args.head} ; périphérique : {det.dev} ; hôtes du test ré-étiquetés : {len(th)} (nouveaux identifiants à partir de {base}, hors entraînement, validation et test) ; flux rejoués : {int(done.sum())}",
          f"- flux dont la classe prédite diffère, rejeu avec ré-étiquetage vs rejeu original : {int((preds['relabel'][done] != preds['original'][done]).sum())}"]
    if args.pred_file and Path(args.pred_file).exists():
        ref = np.load(args.pred_file).astype(np.int64)
        L.append(f"- accord de chaque rejeu avec l'évaluation par lots : original {float((preds['original'][done] == ref[done]).mean()):.6f} ; ré-étiquetage {float((preds['relabel'][done] == ref[done]).mean()):.6f}")
    for tag in ("original", "relabel"):
        s = attack_summary(Y[done], preds[tag][done], names, args.min_support)
        L.append(f"- {tag} : F1 macro attaques {s['f1_att']:.4f} ({s['n_att']} classes) ; {s['f1_big']:.4f} ({s['n_big']} classes ≥ {args.min_support} flux) ; détection binaire {s['det']:.4f} ; fausses alertes {s['fa']:.5f}")
    L += ["", "## Limites", "",
          "- Le ré-étiquetage montre que les prédictions ne dépendent pas de la valeur des identifiants d'hôtes ; il ne montre pas qu'un attaquant au comportement nouveau serait détecté.",
          "- Les hôtes gardent leur comportement et leurs liens entre eux : seul l'identifiant change."]
    rp = Path(f"reports/ef2_hosts{args.tag}.md")
    rp.parent.mkdir(exist_ok=True)
    rp.write_text("\n".join(L), encoding="utf-8")
    print(f"Rapport écrit : {rp.resolve()}")


if __name__ == "__main__":
    main()
