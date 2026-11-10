"""Étape 8 (robustesse) - Perturbations injectées dans le flux de fenêtres du détecteur en ligne.

Quatre familles, chacune à plusieurs niveaux, rejouées sur l'ensemble choisi avec le détecteur de l'étape 7 :
  fantomes  : ajout, dans chaque fenêtre, de niveau x (nombre de flux de la fenêtre) flux supplémentaires. Variables copiées de
              flux BENIGN réels de l'ensemble, extrémités tirées au hasard parmi les hôtes de la fenêtre.
  bruit     : sur les flux d'attaque seulement, ajout d'un bruit gaussien d'écart-type niveau x (écart-type de la variable sur
              l'ensemble), sans contrainte de plausibilité physique des variables.
  suppression : retrait au hasard de la part niveau des flux de chaque fenêtre (capture incomplète).
  fenetres  : perte de la part niveau des fenêtres entières (elles ne passent pas dans le détecteur ni dans le cache),
              ce qui retire aussi l'historique des minutes suivantes. Ce n'est pas une dérive de distribution.
Évaluation : toujours sur les flux d'origine encore présents et traités ; la référence est la prédiction non perturbée du
même détecteur sur exactement les mêmes flux. Dégradation relative = (référence - perturbé) / référence.

Usage (depuis la racine du projet) :
    python -m src.realtime.adversarial --head models/checkpoints/temporal_tl_f10_s0.pt --tag _s0
"""
import argparse
import json
from pathlib import Path

import numpy as np

from src.realtime.replay import attack_summary

GRILLE = {
    "fantomes": [0.01, 0.05, 0.10, 0.20],
    "bruit": [0.05, 0.10, 0.20, 0.50],
    "suppression": [0.05, 0.10, 0.20],
    "fenetres": [0.05, 0.10, 0.20],
}


class Data:
    def __init__(self, path, benign_id):
        z = np.load(path)
        self.X, self.S, self.D = z["edge_attr"], z["src"], z["dst"]
        self.Y = z["y"].astype(np.int64)
        self.ptr = z["win_ptr"]
        self.wstart = z["win_start"].astype(np.int64)
        self.wfile = z["win_file"].astype(np.int64)
        self.order = np.lexsort((self.wstart, self.wfile))
        self.benign = benign_id
        self.benign_idx = np.flatnonzero(self.Y == benign_id)
        self.std = self.X.std(axis=0).astype(np.float32)


def perturb(kind, level, rng, data, a, b):
    """Retourne None (fenêtre perdue) ou (x, s, d, keep) ; keep = positions (dans la fenêtre) des flux d'origine conservés, placés en tête."""
    x, s, d = data.X[a:b], data.S[a:b], data.D[a:b]
    E = b - a
    allk = np.arange(E)
    if kind == "fenetres":
        return None if rng.random() < level else (x, s, d, allk)
    if kind == "suppression":
        keep = np.flatnonzero(rng.random(E) >= level)
        if len(keep) == 0:
            keep = np.array([int(rng.integers(E))])
        return x[keep], s[keep], d[keep], keep
    if kind == "bruit":
        m = data.Y[a:b] != data.benign
        x2 = x.copy()
        if m.any():
            x2[m] += (level * data.std * rng.standard_normal((int(m.sum()), x.shape[1]))).astype(x.dtype)
        return x2, s, d, allk
    if kind == "fantomes":
        hosts = np.unique(np.concatenate([s, d]))
        n_g = int(round(level * E))
        if n_g == 0 or len(hosts) < 2 or len(data.benign_idx) == 0:
            return x, s, d, allk
        gi = rng.integers(0, len(hosts), n_g)
        gj = (gi + rng.integers(1, len(hosts), n_g)) % len(hosts)
        sel = rng.choice(data.benign_idx, n_g)
        return (np.concatenate([x, data.X[sel]]), np.concatenate([s, hosts[gi]]),
                np.concatenate([d, hosts[gj]]), allk)
    raise ValueError(kind)


def run_config(det, data, kind, level, rng, max_windows=0):
    """Rejoue l'ensemble avec la perturbation. Retourne (prédictions [-1 si non évalué], temps totaux par fenêtre en ms)."""
    det.reset()
    pred = np.full(len(data.Y), -1, dtype=np.int64)
    times = []
    order = data.order[:max_windows] if max_windows else data.order
    for i in order:
        a, b = int(data.ptr[i]), int(data.ptr[i + 1])
        r = (data.X[a:b], data.S[a:b], data.D[a:b], np.arange(b - a)) if kind is None else perturb(kind, level, rng, data, a, b)
        if r is None:
            continue
        x, s, d, keep = r
        p, tm = det.process(int(data.wfile[i]), int(data.wstart[i] // 60), x, s, d)
        pred[a + keep] = p[:len(keep)]
        times.append(tm["total"])
    return pred, np.array(times)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph-dir", default="data/graphs")
    ap.add_argument("--head", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seeds", type=int, default=3, help="nombre de tirages aléatoires par niveau de perturbation")
    ap.add_argument("--min-support", type=int, default=100)
    ap.add_argument("--max-windows", type=int, default=0)
    ap.add_argument("--kinds", default="fantomes,bruit,suppression,fenetres")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    from src.realtime.engine import OnlineDetector

    gd = Path(args.graph_dir)
    meta = json.loads((gd / "meta.json").read_text(encoding="utf-8"))
    names = [k for k, _ in sorted(meta["label_id"].items(), key=lambda t: t[1])]
    data = Data(gd / f"{args.split}.npz", names.index("BENIGN"))
    det = OnlineDetector(args.head, data.X.shape[1], args.device)
    ms = args.min_support

    run_config(det, data, None, 0, None, args.max_windows)  # échauffement (non utilisé)
    base, tb = run_config(det, data, None, 0, None, args.max_windows)
    done = base >= 0
    sb = attack_summary(data.Y[done], base[done], names, ms)
    L = ["# Étape 8 - Robustesse aux perturbations (détecteur en ligne)", "",
         f"- tête : {args.head} ; ensemble : {args.split} ; périphérique : {det.dev} ; tirages par niveau : {args.seeds} ; classes comptées « 11 classes » : au moins {ms} flux",
         f"- référence non perturbée : F1 macro attaques {sb['f1_big']:.4f} ({sb['n_big']} classes) ; {sb['f1_att']:.4f} ({sb['n_att']} classes) ; détection binaire {sb['det']:.4f} ; fausses alertes {sb['fa']:.5f} ; temps total médian {np.median(tb):.2f} ms, p99 {np.percentile(tb, 99):.2f} ms",
         "- dégradation relative = (référence - perturbé) / référence, référence calculée sur les mêmes flux que la version perturbée ; moyenne (écart-type) sur les tirages ; maximum entre parenthèses",
         "- seuil HS4 : dégradation < 5 %", "",
         "| Perturbation | niveau | flux évalués | F1 macro (n classes) | dégradation F1 | F1 macro (14 classes) | détection binaire (réf.) | fausses alertes (réf.) | temps médian / p99 (ms) | dégradation F1 < 5 % |",
         "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for kind in [k for k in args.kinds.split(",") if k]:
        for li, level in enumerate(GRILLE[kind]):
            rows, tt = [], []
            for sd in range(args.seeds):
                rng = np.random.default_rng([sd, li, len(kind)])
                p, t = run_config(det, data, kind, level, rng, args.max_windows)
                m = p >= 0
                sp = attack_summary(data.Y[m], p[m], names, ms)
                sr = attack_summary(data.Y[m], base[m], names, ms)
                rows.append((int(m.sum()), sp, sr))
                tt.append(t)
            tt = np.concatenate(tt)
            f = lambda g: np.array([g(r) for r in rows])
            deg = f(lambda r: (r[2]["f1_big"] - r[1]["f1_big"]) / r[2]["f1_big"])
            L.append(f"| {kind} | {level:.2f} | {int(f(lambda r: r[0]).mean())} | {f(lambda r: r[1]['f1_big']).mean():.4f} ({rows[0][1]['n_big']}) | "
                     f"{100 * deg.mean():.2f} % ({100 * deg.std():.2f} ; max {100 * deg.max():.2f}) | {f(lambda r: r[1]['f1_att']).mean():.4f} | "
                     f"{f(lambda r: r[1]['det']).mean():.4f} ({f(lambda r: r[2]['det']).mean():.4f}) | "
                     f"{f(lambda r: r[1]['fa']).mean():.5f} ({f(lambda r: r[2]['fa']).mean():.5f}) | {np.median(tt):.2f} / {np.percentile(tt, 99):.2f} | "
                     f"{'oui' if deg.max() < 0.05 else 'non'} |")
            print(kind, level, f"dégradation F1 {100 * deg.mean():.2f} %", flush=True)
    L += ["", "## Limites", "",
          "- Les perturbations sont aléatoires et non optimisées contre le modèle : elles ne constituent pas une attaque adverse au pire cas.",
          "- Le bruit sur les variables est gaussien et sans contrainte de plausibilité physique des variables du flux.",
          "- Les flux fantômes copient des variables de flux BENIGN réels et relient des hôtes déjà présents dans la fenêtre.",
          "- La perte de fenêtres simule une panne de capture ; elle ne simule pas un changement de distribution du trafic (concept drift)."]
    rp = Path(f"reports/adversarial{args.tag}.md")
    rp.parent.mkdir(exist_ok=True)
    rp.write_text("\n".join(L), encoding="utf-8")
    print(f"Rapport écrit : {rp.resolve()}")


if __name__ == "__main__":
    main()
