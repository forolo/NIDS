"""Étape 1 (fin) - Découpage temporel puis normalisation ajustée sur l'entraînement seul.

Entrée  : Parquet nettoyés (data/interim).
Sortie  : train.parquet, val.parquet, test.parquet, scaler.json (dossier --out-dir)
          et un rapport Markdown.

Colonnes de sortie :
  - métadonnées conservées telles quelles : Flow ID, IP et ports source/destination, Protocol,
    Timestamp, Label, plus file (fichier d'origine), win (fenêtre d'une minute), Label_id ;
  - caractéristiques de flux normalisées (float32).
Les ports et Protocol restent bruts (non normalisés) : leur usage comme entrées du modèle sera
décidé plus tard par ablation. Les adresses IP ne sont jamais des entrées du modèle.

Normalisation (statistiques calculées sur les flux d'entraînement uniquement) :
  1. colonnes constantes sur l'entraînement : retirées ;
  2. colonnes strictement identiques à une colonne précédente : retirées (la première est gardée) ;
  3. NaN (issus des valeurs infinies) : remplacés par la médiane de l'entraînement ;
  4. colonnes binaires (valeurs 0/1 seulement) : laissées en 0/1, sans transformation ;
  5. autres colonnes dont le minimum d'entraînement est >= 0 : log1p ; les autres (valeur -1 de
     Init_Win_bytes_*) : inchangées ;
  6. centrage-réduction (moyenne et écart-type de l'entraînement) des colonnes non binaires.

Usage (depuis la racine du projet) :
    python -m src.ingestion.normalize --in-dir data/interim --out-dir data/raw/csvnormalisees
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.ingestion.splitting import assign_file

META = ["Flow ID", "Source IP", "Source Port", "Destination IP", "Destination Port",
        "Protocol", "Timestamp", "Label"]
RAW_KEPT = ["Source Port", "Destination Port", "Protocol"]
SPLITS = ["train", "val", "test"]


def find_duplicates(tr, feats):
    """Colonnes strictement identiques à une colonne précédente. Retourne {supprimée: gardée}."""
    sig = {}
    for c in feats:
        sig.setdefault((round(float(tr[c].mean()), 9), round(float(tr[c].std()), 9),
                        float(tr[c].min()), float(tr[c].max())), []).append(c)
    dup = {}
    for cols in sig.values():
        for i, b in enumerate(cols):
            for a in cols[:i]:
                if a in dup:
                    continue
                if np.array_equal(tr[a].to_numpy(), tr[b].to_numpy(), equal_nan=True):
                    dup[b] = a
                    break
    return dup


def fit_scaler(train, feats):
    tr = train[feats].astype("float64")
    const = [c for c in feats if tr[c].nunique(dropna=True) <= 1]
    feats = [c for c in feats if c not in const]
    dup = find_duplicates(tr[feats], feats)
    feats = [c for c in feats if c not in dup]
    tr = tr[feats]
    median = tr.median()
    binary = [c for c in feats if set(tr[c].dropna().unique()) <= {0.0, 1.0}]
    nonneg = tr.min() >= 0
    log_cols = [c for c in feats if nonneg[c] and c not in binary]
    z = tr.fillna(median)
    z[log_cols] = np.log1p(z[log_cols])
    mean, std = z.mean(), z.std(ddof=0)
    std = std.where(std > 0, 1.0)
    mean[binary], std[binary] = 0.0, 1.0
    return {"features": feats, "constant_dropped": const, "duplicates_dropped": dup,
            "binary_cols": binary, "median": median.to_dict(), "log_cols": log_cols,
            "mean": mean.to_dict(), "std": std.to_dict()}


def transform(df, sc):
    feats = sc["features"]
    z = df[feats].astype("float64")
    z = z.fillna(pd.Series(sc["median"]))
    lc = sc["log_cols"]
    z[lc] = np.log1p(z[lc].clip(lower=0))
    z = (z - pd.Series(sc["mean"])) / pd.Series(sc["std"])
    return z.astype("float32")


def prepare(df):
    """df : flux nettoyés avec colonnes 'split' et 'file'. Retourne (sorties par ensemble, scaler, rapport)."""
    meta = [c for c in META if c in df.columns]
    skip = set(meta) | {"win", "split", "file"}
    feats = [c for c in df.select_dtypes(include=[np.number]).columns if c not in skip]
    train = df[df["split"] == "train"]
    sc = fit_scaler(train, feats)

    labels = ["BENIGN"] + sorted(l for l in df["Label"].unique() if l != "BENIGN")
    lab_id = {l: i for i, l in enumerate(labels)}
    sc["label_id"] = lab_id

    outs, rep = {}, []
    for s in SPLITS:
        part = df[df["split"] == s].sort_values(["file", "Timestamp"], kind="stable")
        z = transform(part, sc)
        o = part[meta + ["file", "win"]].copy()
        o["Label_id"] = o["Label"].map(lab_id).astype("int16")
        outs[s] = pd.concat([o.reset_index(drop=True), z.reset_index(drop=True)], axis=1)
        rep.append((s, part, z))
    return outs, sc, rep


def build_report(df, sc, rep):
    L = ["# Normalisation (statistiques ajustées sur l'entraînement)", ""]
    L += [f"- caractéristiques conservées : {len(sc['features'])}",
          f"- colonnes constantes retirées ({len(sc['constant_dropped'])}) : " + ", ".join(sc["constant_dropped"]),
          f"- colonnes identiques retirées ({len(sc['duplicates_dropped'])}) : " + (", ".join(f"{k} (= {v})" for k, v in sc["duplicates_dropped"].items()) or "aucune"),
          f"- colonnes binaires laissées en 0/1 ({len(sc['binary_cols'])}) : " + ", ".join(sc["binary_cols"]),
          f"- colonnes avec log1p : {len(sc['log_cols'])}",
          "- colonnes ni binaires ni log1p : " + (", ".join(c for c in sc["features"] if c not in sc["log_cols"] and c not in sc["binary_cols"]) or "aucune"),
          "- ports et Protocol : conservés bruts, hors des caractéristiques normalisées", ""]
    labs = list(sc["label_id"])
    L += ["## Flux par étiquette et par ensemble", "", "| Étiquette | train | val | test |", "| --- | --- | --- | --- |"]
    cnt = {s: p["Label"].value_counts() for s, p, _ in rep}
    for lab in labs:
        L.append(f"| {lab} | " + " | ".join(str(int(cnt[s].get(lab, 0))) for s in SPLITS) + " |")
    L += ["", "## Valeurs manquantes imputées (médiane de l'entraînement)", ""]
    for s, p, _ in rep:
        na = p[sc["features"]].isna().sum()
        na = na[na > 0]
        L.append(f"- {s} : " + ("; ".join(f"{k}={int(v)}" for k, v in na.items()) if len(na) else "aucune"))
    L += ["", "## Contrôle après normalisation", ""]
    nb = [c for c in sc["features"] if c not in sc["binary_cols"]]
    z_tr = rep[0][2]
    L.append(f"- entraînement (colonnes non binaires) : moyenne des moyennes = {z_tr[nb].mean().mean():.4f} ; moyenne des écarts-types = {z_tr[nb].std(ddof=0).mean():.4f}")
    for s, _, z in rep[1:]:
        mx = z.abs().max().sort_values(ascending=False).head(5)
        L.append(f"- {s} : plus grandes valeurs absolues : " + "; ".join(f"{k}={v:.1f}" for k, v in mx.items()))
    mx = z_tr.abs().max().sort_values(ascending=False).head(5)
    L.append("- train : plus grandes valeurs absolues : " + "; ".join(f"{k}={v:.1f}" for k, v in mx.items()))
    L += ["", "## Part des flux de test dont l'IP source apparaît dans l'entraînement (même étiquette)", "",
          "| Étiquette | flux test | part (%) |", "| --- | --- | --- |"]
    tr_df, te_df = rep[0][1], rep[2][1]
    for lab in labs:
        t = te_df[te_df["Label"] == lab]
        if len(t) == 0:
            continue
        seen = set(tr_df.loc[tr_df["Label"] == lab, "Source IP"])
        L.append(f"| {lab} | {len(t)} | {100 * t['Source IP'].isin(seen).mean():.1f} |")
    return L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", default="data/interim")
    ap.add_argument("--out-dir", default="data/raw/csvnormalisees")
    ap.add_argument("--report", default="reports/normalisation.md")
    args = ap.parse_args()

    files = sorted(Path(args.in_dir).glob("*.parquet"))
    if not files:
        raise SystemExit(f"Aucun fichier Parquet dans {Path(args.in_dir).resolve()}")
    parts = []
    for f in files:
        print(f"Lecture de {f.name} ...")
        d, _, _ = assign_file(pd.read_parquet(f))
        d["file"] = f.stem
        parts.append(d)
    df = pd.concat(parts, ignore_index=True)
    del parts

    outs, sc, rep = prepare(df)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for s, o in outs.items():
        o.to_parquet(out_dir / f"{s}.parquet", index=False)
        print(f"{s} : {len(o)} flux")
    (out_dir / "scaler.json").write_text(json.dumps(sc, indent=1), encoding="utf-8")

    rp = Path(args.report)
    rp.parent.mkdir(parents=True, exist_ok=True)
    rp.write_text("\n".join(build_report(df, sc, rep)), encoding="utf-8")
    print(f"Rapport écrit : {rp.resolve()}")


if __name__ == "__main__":
    main()
