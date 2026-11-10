"""Étape 2 (1/2) - Magasin de graphes temporels : une fenêtre d'une minute = un graphe.

Entrée  : train.parquet, val.parquet, test.parquet (sortie de normalize.py) et scaler.json.
Sortie  : un fichier {split}.npz par ensemble, hosts.json, meta.json et un rapport Markdown.

Un graphe de fenêtre est reconstruit à la demande à partir des tableaux ci-dessous (pas de
copie par fenêtre) : les flux d'une fenêtre sont contigus, délimités par win_ptr.
  - nœuds = hôtes (indice global dans hosts.json) ; arêtes = flux orientés source -> destination ;
  - edge_attr : caractéristiques normalisées du flux (liste dans meta.json) ;
  - y : Label_id du flux (0 = BENIGN) ;
  - win_ptr[i]:win_ptr[i+1] : flux de la fenêtre i ; win_start : début de fenêtre (secondes
    depuis 1970) ; win_file : indice du fichier d'origine ;
  - sport, dport, proto : valeurs brutes conservées pour l'ablation (jamais des IP).
Les nœuds n'ont pas de caractéristiques propres (initialisation constante, à l'étape 3).

Usage (depuis la racine du projet) :
    python -m src.ingestion.graph_store --in-dir data/raw/csvnormalisees --out-dir data/graphs
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

SPLITS = ["train", "val", "test"]
NON_FEATURES = {"Flow ID", "Source IP", "Source Port", "Destination IP", "Destination Port",
                "Protocol", "Timestamp", "Label", "file", "win", "Label_id"}
EPOCH = pd.Timestamp("1970-01-01")


def build_split(df, feats, hosts, files):
    """Retourne le dictionnaire de tableaux d'un ensemble."""
    f = df["file"].map(files).to_numpy(dtype="int64")
    w = ((df["win"] - EPOCH) // pd.Timedelta("1s")).to_numpy(dtype="int64")
    order = np.lexsort((df["Timestamp"].to_numpy(), w, f))
    d = df.iloc[order]
    f, w = f[order], w[order]
    n = len(d)
    start = np.ones(n, dtype=bool)
    start[1:] = (f[1:] != f[:-1]) | (w[1:] != w[:-1])
    win_id = (np.cumsum(start) - 1).astype("int32")
    first = np.flatnonzero(start)
    return {
        "edge_attr": d[feats].to_numpy(dtype="float32"),
        "src": hosts.get_indexer(d["Source IP"]).astype("int32"),
        "dst": hosts.get_indexer(d["Destination IP"]).astype("int32"),
        "y": d["Label_id"].to_numpy(dtype="int16"),
        "sport": d["Source Port"].to_numpy(dtype="int32"),
        "dport": d["Destination Port"].to_numpy(dtype="int32"),
        "proto": d["Protocol"].to_numpy(dtype="int16"),
        "win_id": win_id,
        "win_ptr": np.r_[first, n].astype("int64"),
        "win_start": w[first],
        "win_file": f[first].astype("int16"),
    }


def q(a):
    a = np.asarray(a)
    return f"min={int(a.min())} ; médiane={int(np.median(a))} ; p95={int(np.quantile(a, 0.95))} ; max={int(a.max())}"


def split_stats(name, S, n_hosts):
    W = len(S["win_ptr"]) - 1
    edges = np.diff(S["win_ptr"])
    wid = S["win_id"].astype("int64")
    ks = np.concatenate([wid * n_hosts + S["src"], wid * n_hosts + S["dst"]])
    nodes = np.bincount(np.unique(ks) // n_hosts, minlength=W)
    kd, cd = np.unique(wid * n_hosts + S["dst"], return_counts=True)
    maxin = pd.Series(cd).groupby(kd // n_hosts).max()
    att = np.unique(S["win_id"][S["y"] != 0]).size
    return [f"### {name}", "",
            f"- flux : {len(S['y'])} ; fenêtres : {W} (dont {att} avec au moins un flux d'attaque)",
            f"- arêtes par fenêtre : {q(edges)}",
            f"- nœuds par fenêtre : {q(nodes)}",
            f"- degré entrant maximal par fenêtre : {q(maxin.to_numpy())}", ""]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", default="data/raw/csvnormalisees")
    ap.add_argument("--out-dir", default="data/graphs")
    ap.add_argument("--report", default="reports/graph_store.md")
    args = ap.parse_args()
    in_dir, out_dir = Path(args.in_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sc = json.loads((in_dir / "scaler.json").read_text(encoding="utf-8"))
    feats = sc["features"]
    dfs = {s: pd.read_parquet(in_dir / f"{s}.parquet") for s in SPLITS}
    missing = [c for c in feats if any(c not in d.columns for d in dfs.values())]
    if missing:
        raise SystemExit(f"Colonnes absentes : {missing}")

    ips = pd.concat([d[c] for d in dfs.values() for c in ("Source IP", "Destination IP")])
    hosts = pd.Index(sorted(ips.unique()))
    files = {x: i for i, x in enumerate(sorted(set().union(*[set(d["file"].unique()) for d in dfs.values()])))}

    lines = ["# Magasin de graphes temporels", "",
             f"- caractéristiques d'arête : {len(feats)}", f"- hôtes distincts : {len(hosts)}", "",
             "## Statistiques par ensemble", ""]
    host_sets, win_keys = {}, {}
    for s in SPLITS:
        S = build_split(dfs[s], feats, hosts, files)
        np.savez(out_dir / f"{s}.npz", **S)
        lines += split_stats(s, S, len(hosts))
        host_sets[s] = set(np.unique(np.r_[S["src"], S["dst"]]).tolist())
        win_keys[s] = set(zip(S["win_file"].tolist(), S["win_start"].tolist()))
        print(f"{s} : {len(S['y'])} flux, {len(S['win_ptr']) - 1} fenêtres")

    lines += ["## Contrôles", "",
              f"- fenêtres communes train/val : {len(win_keys['train'] & win_keys['val'])} ; train/test : {len(win_keys['train'] & win_keys['test'])} ; val/test : {len(win_keys['val'] & win_keys['test'])}",
              f"- hôtes du test absents de l'entraînement : {len(host_sets['test'] - host_sets['train'])} sur {len(host_sets['test'])}",
              f"- hôtes de la validation absents de l'entraînement : {len(host_sets['val'] - host_sets['train'])} sur {len(host_sets['val'])}"]

    (out_dir / "hosts.json").write_text(json.dumps(list(hosts)), encoding="utf-8")
    meta = {"features": feats, "label_id": sc["label_id"], "files": files}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    rp = Path(args.report)
    rp.parent.mkdir(parents=True, exist_ok=True)
    rp.write_text("\n".join(lines), encoding="utf-8")
    print(f"Rapport écrit : {rp.resolve()}")


if __name__ == "__main__":
    main()
