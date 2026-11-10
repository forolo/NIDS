"""Étape 8 (surveillance) - Suivi de la dérive des entrées et des scores au fil des fenêtres d'une minute.

Définition retenue (aucune étiquette n'est utilisée pendant la surveillance) :
  dérive des entrées : écart entre la distribution des 63 attributs des flux que la tête classe BENIGN dans les W dernières
      minutes du même fichier et la distribution de référence (flux d'entraînement classés BENIGN par la tête). Indicateur :
      PSI par attribut (10 classes issues des quantiles de la référence), moyenné sur les attributs ; on note aussi le nombre
      d'attributs dont le PSI atteint 0,25.
  dérive des scores : incertitude moyenne (1 - probabilité maximale) des flux classés BENIGN sur les W dernières minutes.
  Seuils d'alarme : quantile --calib-q de l'indicateur sur le flux de validation (même procédure, sans dérive).
Les flux d'attaque non détectés (classés BENIGN) restent dans les indicateurs : l'arrivée d'attaques est elle-même un écart de distribution.

Évaluation :
  1. taux d'alarme sur le flux de test sans modification (alarmes « naturelles ») ;
  2. dérive injectée : à partir de la fenêtre p0 de l'ordre fichier puis temps, tous les attributs sont décalés de delta x écart-type
     de la référence ; délai de détection (fenêtres), alarmes avant p0 ; effet sur la détection (F1 macro des attaques, détection
     binaire) sur les fenêtres à partir de p0, comparé aux prédictions non décalées des mêmes flux.

Usage (depuis la racine du projet) :
    python -m src.realtime.drift --head models/checkpoints/temporal_tl_f10_s0.pt --pred-tag _tl_f10_s0 --tag _s0
"""
import argparse
import json
from collections import deque
from pathlib import Path

import numpy as np

NB = 10
EPS = 1e-4


def make_edges(ref, nb=NB):
    return np.quantile(ref, np.linspace(1 / nb, 1 - 1 / nb, nb - 1), axis=0).T.astype(np.float64)


def bin_counts(X, edges, nb=NB):
    out = np.zeros((X.shape[1], nb))
    for j in range(X.shape[1]):
        out[j] = np.bincount(np.searchsorted(edges[j], X[:, j], side="right"), minlength=nb)
    return out


def psi_from(counts, ref_p):
    n = counts.sum(axis=1, keepdims=True)
    p = np.clip(counts / np.maximum(n, 1), EPS, None)
    r = np.clip(ref_p, EPS, None)
    return ((p - r) * np.log(p / r)).sum(axis=1)


class DriftMonitor:
    def __init__(self, edges, ref_p, window, min_flows):
        self.edges, self.ref_p, self.W, self.min_flows = edges, ref_p, window, min_flows
        self.buf = deque(maxlen=window)

    def reset(self):
        self.buf.clear()

    def update(self, counts, n, unc_sum):
        """Ajoute une fenêtre ; retourne (PSI moyen, nb d'attributs PSI >= 0,25, incertitude moyenne) ou None si trop peu de flux."""
        self.buf.append((counts, n, unc_sum))
        N = sum(b[1] for b in self.buf)
        if N < self.min_flows:
            return None
        c = sum(b[0] for b in self.buf)
        psi = psi_from(c, self.ref_p)
        return float(psi.mean()), int((psi >= 0.25).sum()), float(sum(b[2] for b in self.buf) / N), psi


def softmax_unc(lg):
    e = np.exp(lg - lg.max(axis=1, keepdims=True))
    return 1.0 - (e.max(axis=1) / e.sum(axis=1))


def run_stream(det, X, S, D, ptr, wstart, wfile, order, mon, benign, shift=0.0, std=None, p0=0):
    """Rejoue les fenêtres dans l'ordre ; retourne les prédictions par flux et les indicateurs par position."""
    det.reset()
    mon.reset()
    pred = np.full(len(X), -1, dtype=np.int64)
    rows, psis, last_file = [], [], None
    for pos, i in enumerate(order):
        a, b = int(ptr[i]), int(ptr[i + 1])
        f = int(wfile[i])
        if f != last_file:
            mon.reset()
            last_file = f
        x = X[a:b]
        if shift and pos >= p0:
            x = (x + shift * std).astype(np.float32)
        p, _, lg = det.process(f, int(wstart[i] // 60), x, S[a:b], D[a:b], want_logits=True)
        pred[a:b] = p
        m = p == benign
        st = mon.update(bin_counts(x[m], mon.edges), int(m.sum()), float(softmax_unc(lg[m]).sum()))
        rows.append((pos, f, int(wstart[i]), b - a, int(m.sum()), float((~m).mean()), st[:3] if st else None))
        psis.append(st[3] if st else None)
    return pred, rows, psis


def series(rows, k):
    """Valeurs de l'indicateur k (0 PSI moyen, 1 nb PSI>=0,25, 2 incertitude) ; NaN si non disponible."""
    return np.array([r[6][k] if r[6] else np.nan for r in rows], dtype=np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph-dir", default="data/graphs")
    ap.add_argument("--emb-dir", default="data/embeddings")
    ap.add_argument("--head", required=True)
    ap.add_argument("--pred-tag", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--window", type=int, default=10, help="W : minutes (fenêtres) du tampon glissant")
    ap.add_argument("--min-flows", type=int, default=1000)
    ap.add_argument("--calib-q", type=float, default=0.99)
    ap.add_argument("--shifts", default="0.1,0.25,0.5,1.0")
    ap.add_argument("--p0-frac", type=float, default=0.5)
    ap.add_argument("--ref-size", type=int, default=300000)
    ap.add_argument("--min-support", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-windows", type=int, default=0)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    from src.realtime.engine import OnlineDetector
    from src.realtime.replay import attack_summary

    gd, ed = Path(args.graph_dir), Path(args.emb_dir)
    meta = json.loads((gd / "meta.json").read_text(encoding="utf-8"))
    names = [k for k, _ in sorted(meta["label_id"].items(), key=lambda t: t[1])]
    feats = list(meta["features"])
    benign = names.index("BENIGN")
    rng = np.random.default_rng(args.seed)

    tr = np.load(gd / "train.npz")
    ptr_tr = np.load(ed / f"pred{args.pred_tag}_train.npy").astype(np.int64)
    ref_all = tr["edge_attr"][ptr_tr == benign]
    ref = ref_all[rng.choice(len(ref_all), min(args.ref_size, len(ref_all)), replace=False)]
    del tr, ref_all
    edges = make_edges(ref)
    rc = bin_counts(ref, edges)
    ref_p = rc / rc.sum(axis=1, keepdims=True)
    std = ref.std(axis=0).astype(np.float32)

    def load(split):
        z = np.load(gd / f"{split}.npz")
        order = np.lexsort((z["win_start"].astype(np.int64), z["win_file"].astype(np.int64)))
        return (z["edge_attr"], z["src"], z["dst"], z["y"].astype(np.int64), z["win_ptr"],
                z["win_start"].astype(np.int64), z["win_file"].astype(np.int64),
                order[:args.max_windows] if args.max_windows else order)
    val, test = load("val"), load("test")
    det = OnlineDetector(args.head, test[0].shape[1], args.device)
    mon = DriftMonitor(edges, ref_p, args.window, args.min_flows)

    def run(d, **kw):
        X, S, D, Y, ptr, ws, wf, order = d
        return run_stream(det, X, S, D, ptr, ws, wf, order, mon, benign, **kw)

    _, rv, _ = run(val)
    sv = [series(rv, k) for k in (0, 1, 2)]
    tau_in = float(np.nanquantile(sv[0], args.calib_q))
    tau_out = float(np.nanquantile(sv[2], args.calib_q))
    Xt, St, Dt, Yt, ptr_t, ws_t, wf_t, order_t = test
    base, rt, pt = run(test)
    st = [series(rt, k) for k in (0, 1, 2)]
    ok = ~np.isnan(st[0])
    a_in, a_out = st[0] > tau_in, st[2] > tau_out
    nwin = len(rt)
    p0 = int(args.p0_frac * nwin)

    L = ["# Étape 8 - Surveillance de la dérive (entrées et scores)", "",
         f"- tête : {args.head} ; périphérique : {det.dev} ; fenêtres de test : {nwin} ; tampon glissant : {args.window} minutes du même fichier, au moins {args.min_flows} flux classés BENIGN",
         f"- référence : {len(ref)} flux d'entraînement classés BENIGN par la tête (10 classes par quantiles de la référence, PSI avec plancher {EPS}) ; décalage de dérive : delta x écart-type de la référence",
         f"- seuils (quantile {args.calib_q} de la validation, {int((~np.isnan(sv[0])).sum())} valeurs) : PSI moyen {tau_in:.4f} ; incertitude moyenne {tau_out:.4f}",
         f"- repères usuels pour un PSI par attribut : 0,10 (modéré), 0,25 (fort) ; ce sont des conventions de la pratique, pas des seuils établis ici", "",
         "## Alarmes sur le flux de test non modifié", "",
         f"- indicateurs disponibles : {int(ok.sum())} fenêtres sur {nwin}",
         f"- alarme d'entrée (PSI moyen > seuil) : {100 * a_in[ok].mean():.1f} % des fenêtres ; alarme de scores : {100 * a_out[ok].mean():.1f} % ; l'une ou l'autre : {100 * (a_in | a_out)[ok].mean():.1f} %",
         f"- PSI moyen : médiane {np.nanmedian(st[0]):.4f}, p95 {np.nanquantile(st[0], .95):.4f}, max {np.nanmax(st[0]):.4f} ; attributs à PSI >= 0,25 : médiane {np.nanmedian(st[1]):.0f}, max {np.nanmax(st[1]):.0f}", "",
         "| Fichier | fenêtres | avec indicateur | alarmes d'entrée | alarmes de scores | PSI moyen médian | part alertes (non BENIGN) moyenne |", "| --- | --- | --- | --- | --- | --- | --- |"]
    files = np.array([r[1] for r in rt])
    for f in np.unique(files):
        m = files == f
        mo = m & ok
        L.append(f"| {f} | {int(m.sum())} | {int(mo.sum())} | {100 * a_in[mo].mean() if mo.any() else float('nan'):.1f} % | {100 * a_out[mo].mean() if mo.any() else float('nan'):.1f} % | "
                 f"{np.nanmedian(st[0][m]) if mo.any() else float('nan'):.4f} | {100 * np.mean([r[5] for r, mm in zip(rt, m) if mm]):.2f} % |")
    P = np.array([p for p in pt if p is not None])
    if len(P):
        mp = P.mean(axis=0)
        L += ["", "Attributs de plus fort PSI moyen sur le flux de test non modifié :", ""]
        L += [f"- {feats[j]} : {mp[j]:.3f}" for j in np.argsort(-mp)[:10]]
    Path("reports").mkdir(exist_ok=True)
    with open(f"reports/drift{args.tag}_windows.csv", "w", encoding="utf-8") as f:
        f.write("position,fichier,debut_s,flux,flux_benign,part_alertes,psi_moyen,n_psi25,incertitude\n")
        for r in rt:
            f.write(f"{r[0]},{r[1]},{r[2]},{r[3]},{r[4]},{r[5]:.5f}," + (",".join(f"{v:.5f}" for v in r[6]) if r[6] else ",,") + "\n")

    sel = np.zeros(len(Yt), dtype=bool)
    for i in order_t[p0:]:
        sel[int(ptr_t[i]):int(ptr_t[i + 1])] = True
    sb = attack_summary(Yt[sel], base[sel], names, args.min_support)
    L += ["", f"## Dérive injectée à partir de la position {p0} (fenêtre {p0 + 1} sur {nwin})", "",
          f"- référence non décalée sur les fenêtres à partir de p0 : F1 macro des attaques {sb['f1_big']:.4f} ({sb['n_big']} classes de {args.min_support} flux ou plus) ; détection binaire {sb['det']:.4f} ; fausses alertes {sb['fa']:.5f}",
          "- délai : nombre de fenêtres entre p0 et la première alarme (alarme d'entrée ou de scores) à partir de p0 ; alarmes avant p0 : part des fenêtres avant p0 en alarme", "",
          "| décalage (écarts-types) | alarmes avant p0 | part en alarme à partir de p0 | délai (fenêtres) | PSI moyen à partir de p0 (médiane) | F1 macro (n classes) | dégradation F1 | détection binaire | fausses alertes |",
          "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    pre_alarm = (a_in | a_out)[:p0][ok[:p0]]
    L.append(f"| 0 (référence) | {100 * pre_alarm.mean() if len(pre_alarm) else float('nan'):.1f} % | {100 * (a_in | a_out)[p0:][ok[p0:]].mean():.1f} % | - | {np.nanmedian(st[0][p0:]):.4f} | {sb['f1_big']:.4f} ({sb['n_big']}) | 0.00 % | {sb['det']:.4f} | {sb['fa']:.5f} |")
    for dl in [float(s) for s in args.shifts.split(",") if s]:
        pd_, rd, _ = run(test, shift=dl, std=std, p0=p0)
        sd = [series(rd, k) for k in (0, 1, 2)]
        al = ((sd[0] > tau_in) | (sd[2] > tau_out)) & ~np.isnan(sd[0])
        first = np.flatnonzero(al[p0:])
        delay = f"{int(first[0])}" if len(first) else "non détectée"
        okd = ~np.isnan(sd[0])
        pre = al[:p0][okd[:p0]]
        sp = attack_summary(Yt[sel], pd_[sel], names, args.min_support)
        sr = attack_summary(Yt[sel], base[sel], names, args.min_support)
        deg = (sr["f1_big"] - sp["f1_big"]) / sr["f1_big"]
        L.append(f"| {dl:g} | {100 * pre.mean() if len(pre) else float('nan'):.1f} % | {100 * al[p0:][okd[p0:]].mean():.1f} % | {delay} | {np.nanmedian(sd[0][p0:]):.4f} | "
                 f"{sp['f1_big']:.4f} ({sp['n_big']}) | {100 * deg:.2f} % | {sp['det']:.4f} | {sp['fa']:.5f} |")
        print(f"décalage {dl:g} : délai {delay}, dégradation F1 {100 * deg:.2f} %", flush=True)
    L += ["", "## Limites", "",
          "- La dérive injectée est un décalage de moyenne identique pour les 63 attributs ; ce n'est pas un changement réaliste du trafic ni un changement de la relation entre attributs et classes (dérive conceptuelle).",
          "- Les seuils sont calibrés sur la validation avec le même estimateur, par quantile ; le taux de fausses alarmes sur le test non modifié inclut les écarts réels entre fichiers (jours) et la présence d'attaques non détectées.",
          "- Les fenêtres de test sont des fenêtres propres à chaque classe (découpe par classe) : leur composition ne reflète pas un flux de trafic réel continu.",
          "- Une seule tête (graine 0) ; les délais sont en fenêtres d'une minute."]
    rp = Path(f"reports/drift{args.tag}.md")
    rp.write_text("\n".join(L), encoding="utf-8")
    print(f"Rapport écrit : {rp.resolve()}")


if __name__ == "__main__":
    main()
