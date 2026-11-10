"""Étape 1 (suite) - Exploration des épisodes d'attaque et de la taille des fenêtres.

Lit les fichiers Parquet nettoyés (sortie de 02_clean_dryrun.py --apply) et écrit un
rapport. Aucune donnée n'est modifiée.

Usage (depuis la racine du projet) :
    python scripts/audit/03_episodes.py --in-dir data/interim
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def minute_stats(df):
    """Arêtes et nœuds par minute (fenêtre d'une minute, instant de début du flux)."""
    m = df["Timestamp"].dt.floor("min")
    edges = m.value_counts()
    ips = pd.concat([
        pd.DataFrame({"m": m, "ip": df["Source IP"]}),
        pd.DataFrame({"m": m, "ip": df["Destination IP"]}),
    ])
    nodes = ips.groupby("m")["ip"].nunique()
    return edges, nodes


def q(s):
    return f"min={int(s.min())}; médiane={int(s.median())}; p95={int(s.quantile(0.95))}; max={int(s.max())}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", default="data/interim")
    ap.add_argument("--report", default="reports/episodes.md")
    args = ap.parse_args()

    files = sorted(Path(args.in_dir).glob("*.parquet"))
    if not files:
        raise SystemExit(f"Aucun fichier Parquet dans {Path(args.in_dir).resolve()}")

    lines = ["# Épisodes d'attaque et taille des fenêtres (fenêtre d'une minute)", ""]
    all_edges, all_nodes = [], []
    for f in files:
        df = pd.read_parquet(f, columns=["Timestamp", "Label", "Source IP", "Destination IP", "Flow Duration"])
        edges, nodes = minute_stats(df)
        all_edges.append(edges)
        all_nodes.append(nodes)
        lines += [f"## {f.name}", "",
                  f"- flux : {len(df)} ; de {df['Timestamp'].min()} à {df['Timestamp'].max()}",
                  f"- minutes avec trafic : {len(edges)}",
                  f"- arêtes par minute : {q(edges)}",
                  f"- nœuds par minute : {q(nodes)}", "",
                  "| Étiquette | Flux | Premier | Dernier | Minutes distinctes | Flux/min max | Part des flux de l'étiquette dans son intervalle (%) |",
                  "| --- | --- | --- | --- | --- | --- | --- |"]
        for lab, g in df.groupby("Label"):
            if lab == "BENIGN":
                continue
            first, last = g["Timestamp"].min(), g["Timestamp"].max()
            in_span = ((df["Timestamp"] >= first) & (df["Timestamp"] <= last)).sum()
            per_min = g["Timestamp"].dt.floor("min").value_counts()
            lines.append(f"| {lab} | {len(g)} | {first} | {last} | {len(per_min)} | {int(per_min.max())} | {100 * len(g) / in_span:.2f} |")
        lines.append("")

    e, n = pd.concat(all_edges), pd.concat(all_nodes)
    lines += ["## Toutes les fenêtres d'une minute (tous fichiers)", "",
              f"- fenêtres : {len(e)}",
              f"- arêtes par fenêtre : {q(e)}",
              f"- nœuds par fenêtre : {q(n)}"]
    out = Path(args.report)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Rapport écrit : {out.resolve()}")


if __name__ == "__main__":
    main()
