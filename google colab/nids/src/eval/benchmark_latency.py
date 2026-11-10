"""Étape 6 (mesures) - Latence et mémoire de l'inférence fenêtre par fenêtre.

Chemin mesuré pour chaque fenêtre d'une minute du jeu choisi (une fenêtre = un appel) :
  1. préparation des tableaux de la fenêtre et copie vers le périphérique ;
  2. encodeur E-GraphSAGE (états d'hôtes de la fenêtre) ;
  3. historique des k-1 fenêtres précédentes (table des états déjà calculés), LSTM, tête, argmax, retour des classes au CPU.
La latence d'une fenêtre est mesurée après synchronisation du périphérique. Les premières fenêtres
(--warmup) sont exclues des statistiques. Les prédictions en ligne sont comparées à celles de
l'évaluation par lots (--pred-file) pour vérifier l'équivalence.

Optionnel : --surrogate mesure aussi le temps de prédiction de XGBoost imitateur sur les mêmes fenêtres
(variables brutes + graphe précalculées ; le temps de calcul des variables de graphe est donné à part).

Usage (depuis la racine du projet) :
    python -m src.eval.benchmark_latency --head models/checkpoints/temporal_tp_f10_s0.pt --pred-file data/embeddings/pred_tp_f10_s0_test.npy --surrogate models/checkpoints/surrogate_s0.json
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from src.models.lstm.train_temporal import SplitData, TemporalHead, load_encoder


def sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize()


def pct(a, q):
    return float(np.percentile(a, q))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph-dir", default="data/graphs")
    ap.add_argument("--head", required=True, help="checkpoint temporal*.pt de train_temporal.py")
    ap.add_argument("--split", default="test")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--pred-file", default=None)
    ap.add_argument("--surrogate", default=None)
    ap.add_argument("--surrogate-device", default="cuda")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    dev = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    gd = Path(args.graph_dir)
    meta = json.loads((gd / "meta.json").read_text(encoding="utf-8"))
    ckh = torch.load(args.head, map_location=dev, weights_only=False)
    a, K = ckh["args"], len(ckh["names"])
    path = gd / f"{args.split}.npz"
    edge_dim = np.load(path)["edge_attr"].shape[1]
    enc = load_encoder(a["checkpoint"], edge_dim, K, dev)
    hid = enc.node_init[0].out_features
    data = SplitData(path, enc, dev, 60000, a["k"])
    head = TemporalHead(hid, K, a["lstm_hidden"], a["dropout"], not a["no_temporal"]).to(dev)
    head.load_state_dict(ckh["state"])
    head.eval()
    st = data.store
    n_par = sum(p.numel() for p in enc.parameters()) + sum(p.numel() for p in head.parameters())

    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t_prep, t_enc, t_head, preds = [], [], [], []
    with torch.no_grad():
        for i in range(len(st)):
            sync(dev); t0 = time.perf_counter()
            x, s, d, y, n = st.batch([i], dev)
            sync(dev); t1 = time.perf_counter()
            ep, z = enc.encode(x, s, d, n)
            sync(dev); t2 = time.perf_counter()
            rows = torch.from_numpy(st.node_rows([i])).to(dev)
            idx = data.hist[rows]
            mask = idx >= 0
            seq = data.Z[idx.clamp(min=0)] * mask.unsqueeze(-1)
            seq[:, -1, :] = z
            seq = torch.cat([seq, mask.unsqueeze(-1).float()], dim=-1)
            h = head.node_h(seq) if head.temporal else None
            p = head(ep, z, h, s, d).argmax(1).cpu().numpy()
            sync(dev); t3 = time.perf_counter()
            t_prep.append(t1 - t0); t_enc.append(t2 - t1); t_head.append(t3 - t2); preds.append(p)
    w = args.warmup
    prep, encm, headm = (np.array(v[w:]) * 1000 for v in (t_prep, t_enc, t_head))
    total = encm + headm
    total_all = prep + total
    edges = st.n_edges[w:]
    peak_mb = torch.cuda.max_memory_allocated() / 2**20 if dev.type == "cuda" else float("nan")
    rss = float("nan")
    try:
        import psutil
        rss = psutil.Process().memory_info().rss / 2**20
    except Exception:
        pass

    L = ["# Latence et mémoire de l'inférence par fenêtre", "",
         f"- tête : {args.head} ; ensemble : {args.split} ; périphérique : {dev} ; paramètres (encodeur + tête) : {n_par}",
         f"- fenêtres mesurées : {len(total)} (après {w} fenêtres d'échauffement) ; flux par fenêtre : médiane {int(np.median(edges))}, max {int(edges.max())}",
         f"- mémoire GPU maximale allouée : {peak_mb:.0f} Mo ; mémoire du processus (RSS) : {rss:.0f} Mo", "",
         "## Temps par fenêtre (ms)", "",
         "| Étape | médiane | p95 | p99 | max |", "| --- | --- | --- | --- | --- |"]
    for name, v in (("préparation + copie vers le périphérique", prep), ("encodeur E-GraphSAGE", encm),
                    ("historique + LSTM + tête + retour CPU", headm), ("calcul du réseau (encodeur + suite)", total),
                    ("total avec préparation", total_all)):
        L.append(f"| {name} | {pct(v, 50):.2f} | {pct(v, 95):.2f} | {pct(v, 99):.2f} | {v.max():.2f} |")
    L += ["", f"- débit du calcul du réseau : {edges.sum() / (total.sum() / 1000):.0f} flux par seconde",
          f"- part des fenêtres dont le calcul du réseau dépasse 10 ms : {100 * float((total > 10).mean()):.1f} % ; dont le total avec préparation dépasse 10 ms : {100 * float((total_all > 10).mean()):.1f} %", "",
          "## Temps de calcul du réseau selon la taille de la fenêtre (ms)", "", "| Flux par fenêtre | fenêtres | médiane | p95 | max |", "| --- | --- | --- | --- | --- |"]
    for lo, hi in ((0, 1000), (1000, 5000), (5000, 20000), (20000, 10 ** 9)):
        m = (edges >= lo) & (edges < hi)
        if m.any():
            L.append(f"| {lo} à {hi if hi < 10**9 else '...'} | {int(m.sum())} | {pct(total[m], 50):.2f} | {pct(total[m], 95):.2f} | {total[m].max():.2f} |")

    pred = np.concatenate(preds)
    if args.pred_file and Path(args.pred_file).exists():
        ref = np.load(args.pred_file).astype(np.int64)
        L += ["", f"## Équivalence avec l'évaluation par lots", "", f"- accord des classes prédites : {float((pred == ref).mean()):.6f} sur {len(ref)} flux"]
    y_true = st.y
    L += ["", f"- exactitude de la prédiction en ligne contre les vraies étiquettes : {float((pred == y_true).mean()):.4f}"]

    if args.surrogate:
        from xgboost import XGBClassifier
        from src.models import graph_features
        z_ = np.load(path)
        t0 = time.perf_counter()
        G = graph_features.compute(z_["src"], z_["dst"], z_["dport"], z_["win_ptr"], z_["win_start"], z_["win_file"])
        t_g = time.perf_counter() - t0
        X = np.hstack([z_["edge_attr"], G]).astype("float32")
        clf = XGBClassifier()
        clf.load_model(args.surrogate)
        clf.set_params(device=args.surrogate_device)
        ptr = st.ptr
        ts = []
        for i in range(len(st)):
            t0 = time.perf_counter()
            clf.predict_proba(X[ptr[i]:ptr[i + 1]])
            ts.append((time.perf_counter() - t0) * 1000)
        ts = np.array(ts[w:])
        L += ["", "## XGBoost imitateur (74 variables), prédiction par fenêtre", "",
              f"- périphérique : {args.surrogate_device} ; médiane {pct(ts, 50):.2f} ms ; p95 {pct(ts, 95):.2f} ms ; p99 {pct(ts, 99):.2f} ms ; max {ts.max():.2f} ms",
              f"- calcul des variables de graphe sur l'ensemble {args.split} (numpy, tout le jeu) : {t_g:.1f} s, soit {1000 * t_g / len(st):.1f} ms par fenêtre en moyenne"]
    rp = Path(f"reports/latency{args.tag}.md")
    rp.parent.mkdir(parents=True, exist_ok=True)
    rp.write_text("\n".join(L), encoding="utf-8")
    print(f"Rapport écrit : {rp.resolve()}")


if __name__ == "__main__":
    main()
