"""Étape 7 - Rejeu en ligne : les fenêtres d'une minute d'un ensemble passent une à une, dans l'ordre du temps,
dans un détecteur sans table précalculée (l'historique des hôtes vient du cache des fenêtres déjà passées).

Mesures : temps par étape et par fenêtre, part des fenêtres > 10 ms (HS4), mémoire (GPU, processus, cache),
équivalence avec les prédictions de l'évaluation par lots (--pred-file), qualité de détection des prédictions en ligne.
Le rejeu part des flux déjà extraits (variables normalisées) : la capture, l'extraction des flux et l'attente
de la fin de la minute ne sont pas dans les temps mesurés.

Usage (depuis la racine du projet) :
    python -m src.realtime.replay --head models/checkpoints/temporal_tl_f10_s0.pt --pred-file data/embeddings/pred_tl_f10_s0_test.npy --tag _s0
"""
import argparse
import json
from pathlib import Path

import numpy as np

from src.eval.confusion_analysis import confusion, f1_from


def pct(a, q):
    return float(np.percentile(a, q))


def rss_mb():
    try:
        import psutil
        return psutil.Process().memory_info().rss / 2**20
    except Exception:
        return float("nan")


def attack_summary(y, p, names, min_support):
    K = len(names)
    cm = confusion(y, p, K)
    b = names.index("BENIGN")
    att = [k for k in range(K) if k != b and cm[k].sum() > 0]
    big = [k for k in att if cm[k].sum() >= min_support]
    f1 = lambda ks: float(np.mean([f1_from(cm, k)[2] for k in ks]))
    rc = lambda ks: float(np.mean([f1_from(cm, k)[0] for k in ks]))
    ya, pa = y != b, p != b
    return {"f1_att": f1(att), "rec_att": rc(att), "n_att": len(att), "f1_big": f1(big), "rec_big": rc(big), "n_big": len(big),
            "det": float((pa & ya).sum() / max(ya.sum(), 1)), "fa": float((pa & ~ya).sum() / max((~ya).sum(), 1)),
            "fa_n": int((pa & ~ya).sum()), "miss_n": int((~pa & ya).sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph-dir", default="data/graphs")
    ap.add_argument("--head", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--max-windows", type=int, default=0, help="limite le rejeu aux premières fenêtres (essai)")
    ap.add_argument("--pred-file", default=None)
    ap.add_argument("--min-support", type=int, default=100)
    ap.add_argument("--save-pred", action="store_true", help="écrit data/embeddings/pred_online{tag}_{split}.npy")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    import torch
    from src.realtime.engine import OnlineDetector

    gd = Path(args.graph_dir)
    meta = json.loads((gd / "meta.json").read_text(encoding="utf-8"))
    names = [k for k, _ in sorted(meta["label_id"].items(), key=lambda t: t[1])]
    z = np.load(gd / f"{args.split}.npz")
    X, SRC, DST, Y = z["edge_attr"], z["src"], z["dst"], z["y"].astype(np.int64)
    ptr, wstart, wfile = z["win_ptr"], z["win_start"].astype(np.int64), z["win_file"].astype(np.int64)
    det = OnlineDetector(args.head, X.shape[1], args.device)
    dev = det.dev
    order = np.lexsort((wstart, wfile))  # fichier puis temps
    if args.max_windows:
        order = order[: args.max_windows]
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    rss0 = rss_mb()

    pred = np.full(len(Y), -1, dtype=np.int64)
    rows, max_cache = [], 0
    for i in order:
        a, b = int(ptr[i]), int(ptr[i + 1])
        p, tm = det.process(int(wfile[i]), int(wstart[i] // 60), X[a:b], SRC[a:b], DST[a:b])
        pred[a:b] = p
        max_cache = max(max_cache, det.hist.n_cached())
        rows.append((int(i), int(wfile[i]), int(wstart[i]), b - a, tm["prep"], tm["enc"], tm["hist"], tm["head"], tm["update"], tm["net"], tm["total"]))
    rss1 = rss_mb()
    R = np.array([r[3:] for r in rows], dtype=np.float64)
    w = min(args.warmup, max(len(R) - 1, 0))
    M = R[w:]
    edges, prep, enc, hist, head, upd, net, tot = (M[:, i] for i in range(8))
    peak = torch.cuda.max_memory_allocated() / 2**20 if dev.type == "cuda" else float("nan")
    cache_mb = sum(Zt.numel() * Zt.element_size() for (_, Zt) in det.hist.win.values()) / 2**20

    Path("reports").mkdir(exist_ok=True)
    csv = Path(f"reports/replay{args.tag}_windows.csv")
    with csv.open("w", encoding="utf-8") as f:
        f.write("fenetre,fichier,debut_s,flux,prep_ms,enc_ms,hist_ms,tete_ms,maj_ms,reseau_ms,total_ms\n")
        for r in rows:
            f.write(",".join([str(r[0]), str(r[1]), str(r[2]), str(r[3])] + [f"{v:.3f}" for v in r[4:]]) + "\n")

    L = ["# Étape 7 - Rejeu en ligne", "",
         f"- tête : {args.head} ; ensemble : {args.split} ; périphérique : {dev} ; paramètres (encodeur + tête) : {det.n_par} ; k = {det.k}",
         f"- fenêtres rejouées : {len(rows)} (dans l'ordre fichier puis temps) ; mesurées : {len(M)} (après {w} d'échauffement) ; flux par fenêtre : médiane {int(np.median(edges))}, max {int(edges.max())}",
         f"- cache d'historique : au plus {max_cache} fenêtres, {cache_mb:.2f} Mo d'états en fin de rejeu",
         f"- mémoire GPU maximale allouée : {peak:.0f} Mo ; mémoire du processus : {rss0:.0f} Mo au début, {rss1:.0f} Mo à la fin", "",
         "## Temps par fenêtre (ms)", "", "| Étape | médiane | p95 | p99 | max |", "| --- | --- | --- | --- | --- |"]
    for nm, v in (("préparation (indexation locale + copie vers le périphérique)", prep), ("encodeur E-GraphSAGE", enc),
                  ("historique des hôtes (cache) + séquence", hist), ("LSTM + tête + retour CPU", head),
                  ("mise à jour du cache", upd), ("calcul du réseau (encodeur + historique + tête)", net),
                  ("total de la fenêtre", tot)):
        L.append(f"| {nm} | {pct(v, 50):.2f} | {pct(v, 95):.2f} | {pct(v, 99):.2f} | {v.max():.2f} |")
    L += ["", "## Seuil de 10 ms (HS4)", "",
          f"- fenêtres dont le calcul du réseau dépasse 10 ms : {100 * float((net > 10).mean()):.1f} % ({int((net > 10).sum())} / {len(net)})",
          f"- fenêtres dont le total dépasse 10 ms : {100 * float((tot > 10).mean()):.1f} % ({int((tot > 10).sum())} / {len(tot)})",
          f"- plus longue fenêtre (total) : {tot.max():.1f} ms pour {int(edges[tot.argmax()])} flux, soit {100 * tot.max() / 60000:.3f} % de la durée d'une fenêtre (60 s)",
          f"- débit moyen (total) : {edges.sum() / (tot.sum() / 1000):.0f} flux par seconde", "",
          "## Total par taille de fenêtre (ms)", "", "| Flux par fenêtre | fenêtres | médiane | p95 | max | part > 10 ms |", "| --- | --- | --- | --- | --- | --- |"]
    for lo, hi in ((0, 1000), (1000, 5000), (5000, 20000), (20000, 10 ** 9)):
        m = (edges >= lo) & (edges < hi)
        if m.any():
            L.append(f"| {lo} à {hi if hi < 10**9 else '...'} | {int(m.sum())} | {pct(tot[m], 50):.2f} | {pct(tot[m], 95):.2f} | {tot[m].max():.2f} | {100 * float((tot[m] > 10).mean()):.1f} % |")

    done = pred >= 0
    L += ["", "## Prédictions en ligne", ""]
    if args.pred_file and Path(args.pred_file).exists():
        ref = np.load(args.pred_file).astype(np.int64)
        agree = float((pred[done] == ref[done]).mean())
        L.append(f"- accord avec l'évaluation par lots ({args.pred_file}) : {agree:.6f} sur {int(done.sum())} flux ; flux en désaccord : {int((pred[done] != ref[done]).sum())}")
    s = attack_summary(Y[done], pred[done], names, args.min_support)
    L += [f"- F1 macro des classes d'attaque : {s['f1_att']:.4f} ({s['n_att']} classes) ; {s['f1_big']:.4f} ({s['n_big']} classes de {args.min_support} flux ou plus)",
          f"- rappel macro des classes d'attaque : {s['rec_att']:.4f} ; {s['rec_big']:.4f} (classes de {args.min_support} flux ou plus)",
          f"- détection binaire (attaque vs BENIGN) : {s['det']:.4f} ({s['miss_n']} flux d'attaque manqués) ; fausses alertes : {s['fa']:.5f} ({s['fa_n']} flux)"]
    L += ["", "## Limites", "",
          "- Le rejeu part des flux déjà extraits : capture, extraction des variables des flux et attente de la fin de la minute ne sont pas mesurées.",
          "- Mesures faites sur un seul périphérique (voir ci-dessus), un seul processus, sans autre charge."]
    rp = Path(f"reports/replay{args.tag}.md")
    rp.write_text("\n".join(L), encoding="utf-8")
    if args.save_pred:
        out = Path("data/embeddings"); out.mkdir(parents=True, exist_ok=True)
        np.save(out / f"pred_online{args.tag}_{args.split}.npy", pred)
    print(f"Rapport écrit : {rp.resolve()}")


if __name__ == "__main__":
    main()
