"""Étape 1 (fin) - Contrôle des caractéristiques normalisées (colonnes identiques, binaires rares,
corrélations très fortes). Lit train.parquet ; ne modifie rien ; écrit reports/features_check.md.

Usage (depuis la racine du projet) :
    python scripts/audit/05_features_check.py --dir data/raw/csvnormalisees
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

SAMPLE = 300_000
CORR_THRESHOLD = 0.99


def analyse(train, feats):
    L = []
    X = train[feats]
    # 1. colonnes identiques
    sig = {c: (round(float(X[c].mean()), 9), round(float(X[c].std()), 9), float(X[c].min()), float(X[c].max())) for c in feats}
    groups = {}
    for c, s in sig.items():
        groups.setdefault(s, []).append(c)
    ident = []
    for cols in groups.values():
        for i, a in enumerate(cols):
            for b in cols[i + 1:]:
                if np.array_equal(X[a].to_numpy(), X[b].to_numpy()):
                    ident.append((a, b))
    L += ["## Colonnes strictement identiques (entraînement)", ""]
    L += [f"- {a} = {b}" for a, b in ident] or ["- aucune"]

    # 2. colonnes à deux valeurs : fréquence de la valeur minoritaire
    L += ["", "## Colonnes à deux valeurs (binaires) : part de la valeur minoritaire", "",
          "| Colonne | flux minoritaires | part (%) | classes concernées (flux) |", "| --- | --- | --- | --- |"]
    lab = train["Label"]
    for c in feats:
        vc = X[c].value_counts()
        if len(vc) != 2:
            continue
        minor = vc.index[vc.argmin()]
        m = X[c] == minor
        by = lab[m].value_counts()
        L.append(f"| {c} | {int(m.sum())} | {100 * m.mean():.4f} | " + "; ".join(f"{k}={v}" for k, v in by.head(4).items()) + " |")

    # 3. corrélations
    s = X.sample(min(SAMPLE, len(X)), random_state=0)
    corr = s.corr().abs()
    pairs = [(corr.index[i], corr.columns[j], corr.iat[i, j]) for i in range(len(corr)) for j in range(i + 1, len(corr))
             if corr.iat[i, j] >= CORR_THRESHOLD]
    L += ["", f"## Paires de colonnes avec |corrélation| >= {CORR_THRESHOLD} (échantillon de {len(s)} flux)", ""]
    L += [f"- {a} ~ {b} : {r:.4f}" for a, b, r in sorted(pairs, key=lambda t: -t[2])] or ["- aucune"]
    return L, ident, pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data/raw/csvnormalisees")
    ap.add_argument("--report", default="reports/features_check.md")
    args = ap.parse_args()
    d = Path(args.dir)
    sc = json.loads((d / "scaler.json").read_text(encoding="utf-8"))
    feats = sc["features"]
    train = pd.read_parquet(d / "train.parquet", columns=feats + ["Label"])
    L, ident, pairs = analyse(train, feats)
    out = Path(args.report)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(["# Contrôle des caractéristiques normalisées", "", f"- caractéristiques : {len(feats)} ; flux d'entraînement : {len(train)}", ""] + L), encoding="utf-8")
    print(f"Rapport écrit : {out.resolve()}")


if __name__ == "__main__":
    main()
