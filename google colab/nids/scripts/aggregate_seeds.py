"""Agrège les métriques du test de plusieurs graines (fichiers reports/metrics/*.json).

Chaque groupe est donné sous la forme "Libellé=motif" ; le motif désigne des fichiers sans l'extension
.json (ex. "ft_unfrozen_f10_s*"). Écrit reports/seeds_summary.md : moyenne et écart-type (n-1) du F1 macro,
de la détection et des faux positifs, rappel moyen par classe, et valeurs par graine.

Usage (depuis la racine du projet) :
    python scripts/aggregate_seeds.py --group "XGBoost 10 %=baseline_xgb_f10_s*" --group "GCL dégelé 10 %=ft_unfrozen_f10_s*"
"""
import argparse
import json
from pathlib import Path

import numpy as np


def ms(a):
    a = np.asarray(a, dtype=float)
    return f"{a.mean():.4f} ± {a.std(ddof=1) if len(a) > 1 else 0.0:.4f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", action="append", required=True, help='"Libellé=motif"')
    ap.add_argument("--metrics-dir", default="reports/metrics")
    ap.add_argument("--out", default="reports/seeds_summary.md")
    args = ap.parse_args()

    groups = {}
    for g in args.group:
        label, pattern = g.split("=", 1)
        files = sorted(Path(args.metrics_dir).glob(pattern + ".json"))
        if not files:
            raise SystemExit(f"Aucun fichier pour {pattern} dans {Path(args.metrics_dir).resolve()}")
        groups[label] = [json.loads(f.read_text(encoding="utf-8")) for f in files]

    L = ["# Synthèse multi-graines (test)", "", "Moyenne ± écart-type (n-1) sur les graines.", "",
         "| Groupe | Graines | F1 macro | Détection | Faux positifs |", "| --- | --- | --- | --- | --- |"]
    for lab, ms_ in groups.items():
        L.append(f"| {lab} | {len(ms_)} | {ms([m['macro_f1'] for m in ms_])} | {ms([m['detection'] for m in ms_])} | {ms([m['fpr'] for m in ms_])} |")
    classes = sorted({c for ms_ in groups.values() for m in ms_ for c in m["per_class"]})
    L += ["", "## Rappel par classe (moyenne ± écart-type)", "", "| Classe | Support test | " + " | ".join(groups) + " |",
          "| --- | --- | " + " | ".join(["---"] * len(groups)) + " |"]
    for c in classes:
        sup = next(m["per_class"][c]["support"] for ms_ in groups.values() for m in ms_ if c in m["per_class"])
        cells = [ms([m["per_class"][c]["recall"] for m in ms_ if c in m["per_class"]]) for ms_ in groups.values()]
        L.append(f"| {c} | {sup} | " + " | ".join(cells) + " |")
    L += ["", "## F1 macro par graine (ordre alphabétique des fichiers)", ""]
    for lab, ms_ in groups.items():
        L.append(f"- {lab} : " + " ; ".join(f"{m['macro_f1']:.4f}" for m in ms_))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L), encoding="utf-8")
    print(f"Rapport écrit : {out.resolve()}")


if __name__ == "__main__":
    main()
