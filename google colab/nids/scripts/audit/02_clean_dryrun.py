"""Étape 1 - Nettoyage structurel de CIC-IDS2017.

Par défaut le script fonctionne en SIMULATION : il ne modifie ni n'écrit aucune donnée,
il écrit seulement un rapport qui indique, règle par règle et étiquette par étiquette,
ce que le nettoyage retirerait ou corrigerait.

Usage (depuis la racine du projet) :
    python scripts/audit/02_clean_dryrun.py
    python scripts/audit/02_clean_dryrun.py --apply --out-dir <dossier des données nettoyées>

Règles appliquées, dans l'ordre :
  1. Noms de colonnes : espaces retirés.
  2. Lignes entièrement vides : supprimées.
  3. Étiquettes : normalisées (ex. "Web Attack  Brute Force" -> "Web Attack - Brute Force").
  4. Colonne "Fwd Header Length.1" : supprimée si elle est identique à "Fwd Header Length".
  5. Horodatage : analysé, puis heures 1 à 7 ramenées à l'après-midi (+12 h), car l'horloge
     est sur 12 h sans AM/PM ; lignes triées chronologiquement.
  6. Doublons exacts (toutes colonnes) : supprimés.
  7. Lignes à durée de flux, longueurs d'en-tête ou min_seg_size_forward négatifs : supprimées.
  8. "Flow IAT Min" et "Fwd IAT Min" négatifs : ramenés à 0.
  9. Valeurs infinies : remplacées par NaN (l'imputation se fait plus tard, sur l'entraînement seul).
 10. Init_Win_bytes_forward / backward = -1 : conservées (valeur sentinelle).
"""
import argparse
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

TS_FORMATS = ["%d/%m/%Y %H:%M:%S", "%d/%m/%Y %H:%M"]
DUP_COL, DUP_REF = "Fwd Header Length.1", "Fwd Header Length"
SIZE_COLS = ["Flow Duration", "Fwd Header Length", "Bwd Header Length", "min_seg_size_forward"]
IAT_CLIP_COLS = ["Flow IAT Min", "Fwd IAT Min"]
FIRST_AFTERNOON_HOUR_LIMIT = 8  # heures < 8 => après-midi


def read_csv_robust(path):
    try:
        return pd.read_csv(path, low_memory=False)
    except UnicodeDecodeError:
        return pd.read_csv(path, low_memory=False, encoding="latin-1")


def normalize_label(s):
    s = re.sub(r"[^\x20-\x7E]", " ", str(s))
    s = re.sub(r"\s+", " ", s).strip()
    m = re.match(r"^Web Attack\s*-?\s*(.+)$", s)
    return "Web Attack - " + m.group(1).strip() if m else s


def parse_timestamps(series):
    best_fmt, best = None, None
    for fmt in TS_FORMATS:
        ts = pd.to_datetime(series, format=fmt, errors="coerce")
        if best is None or ts.isna().sum() < best.isna().sum():
            best_fmt, best = fmt, ts
    return best_fmt, best


def clean_file(path):
    st, removed = {}, {}
    df = read_csv_robust(path)
    df.columns = [str(c).strip() for c in df.columns]
    st["lignes_brutes"] = len(df)

    empty = df.isna().all(axis=1)
    st["lignes_entierement_vides"] = int(empty.sum())
    df = df.loc[~empty].reset_index(drop=True)
    df["Label"] = df["Label"].map(normalize_label)
    before = df["Label"].value_counts()

    if DUP_COL in df.columns and DUP_REF in df.columns:
        same = bool(df[DUP_COL].equals(df[DUP_REF]))
        st["colonne_dupliquee_identique"] = same
        if same:
            df = df.drop(columns=[DUP_COL])
    else:
        st["colonne_dupliquee_identique"] = "colonne absente"

    fmt, ts = parse_timestamps(df["Timestamp"].astype(str))
    st["format_horodatage"] = fmt
    st["horodatages_non_analysables"] = int(ts.isna().sum())
    hours_before = sorted(int(h) for h in ts.dropna().dt.hour.unique())
    st["heures_avant_correction"] = hours_before
    df["Timestamp"] = ts

    dup = df.duplicated(keep="first")
    st["doublons_supprimes"] = int(dup.sum())
    removed["doublons exacts"] = df.loc[dup, "Label"].value_counts()
    df = df.loc[~dup]

    cols = [c for c in SIZE_COLS if c in df.columns]
    bad = (df[cols] < 0).any(axis=1)
    st["lignes_valeurs_negatives_supprimees"] = int(bad.sum())
    removed["durée / en-têtes négatifs"] = df.loc[bad, "Label"].value_counts()
    df = df.loc[~bad]

    clip = {}
    for c in IAT_CLIP_COLS:
        if c in df.columns:
            n = int((df[c] < 0).sum())
            clip[c] = n
            df[c] = df[c].clip(lower=0)
    st["valeurs_IAT_ramenees_a_0"] = clip

    num_cols = df.select_dtypes(include=[np.number]).columns
    inf_counts = {}
    for c in num_cols:
        n = int(np.isinf(df[c]).sum())
        if n:
            inf_counts[c] = n
    st["valeurs_inf_remplacees_par_NaN"] = inf_counts
    df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan)
    inf_rows = df[num_cols].isna().any(axis=1)
    st["lignes_avec_NaN_apres_remplacement"] = int(inf_rows.sum())
    removed["(information) lignes avec NaN conservées"] = df.loc[inf_rows, "Label"].value_counts()

    # Correction AM/PM puis tri chronologique
    early = df["Timestamp"].dt.hour < FIRST_AFTERNOON_HOUR_LIMIT
    st["horodatages_corriges_plus_12h"] = int(early.sum())
    df.loc[early, "Timestamp"] = df.loc[early, "Timestamp"] + pd.Timedelta(hours=12)
    df = df.sort_values("Timestamp", kind="stable").reset_index(drop=True)
    st["heures_apres_correction"] = sorted(int(h) for h in df["Timestamp"].dt.hour.unique())
    st["timestamp_min"] = str(df["Timestamp"].min())
    st["timestamp_max"] = str(df["Timestamp"].max())
    if "Flow Duration" in df.columns:
        st["duree_flux_max_secondes"] = float(df["Flow Duration"].max() / 1e6)
    st["lignes_finales"] = len(df)
    return df, st, removed, before


def fmt(v):
    if isinstance(v, dict):
        return "; ".join(f"{k}={n}" for k, n in v.items()) if v else "aucun"
    return str(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="data/raw/csv")
    ap.add_argument("--report", default="reports/clean_dryrun.md")
    ap.add_argument("--apply", action="store_true", help="écrit les fichiers Parquet nettoyés")
    ap.add_argument("--out-dir", default=None, help="obligatoire avec --apply")
    args = ap.parse_args()
    if args.apply and not args.out_dir:
        raise SystemExit("--apply nécessite --out-dir")

    files = sorted(Path(args.raw_dir).glob("*.csv"))
    if not files:
        raise SystemExit(f"Aucun CSV dans {Path(args.raw_dir).resolve()}")
    out_dir = Path(args.out_dir) if args.out_dir else None
    if args.apply:
        out_dir.mkdir(parents=True, exist_ok=True)

    lines = ["# Nettoyage CIC-IDS2017 - " + ("APPLIQUÉ" if args.apply else "simulation (aucune donnée écrite)"), ""]
    before_tot, after_tot = Counter(), Counter()
    removed_tot = defaultdict(Counter)
    gmin, gmax = None, None
    for f in files:
        print(f"Traitement de {f.name} ...")
        df, st, removed, before = clean_file(f)
        lines += [f"## {f.name}", ""]
        lines += [f"- {k} : {fmt(v)}" for k, v in st.items()]
        lines.append("")
        before_tot.update(before.to_dict())
        after_tot.update(df["Label"].value_counts().to_dict())
        for rule, cnt in removed.items():
            removed_tot[rule].update(cnt.to_dict())
        num = df.select_dtypes(include=[np.number])
        mn, mx = num.min(), num.max()
        gmin = mn if gmin is None else np.minimum(gmin, mn)
        gmax = mx if gmax is None else np.maximum(gmax, mx)
        if args.apply:
            df.to_parquet(out_dir / (f.stem + ".parquet"), index=False)

    lines += ["## Effet par étiquette (tous fichiers)", "", "| Étiquette | Avant | Après | Retirées | % retirées |", "| --- | --- | --- | --- | --- |"]
    for lab, n in sorted(before_tot.items(), key=lambda x: -x[1]):
        a = after_tot.get(lab, 0)
        lines.append(f"| {lab} | {n} | {a} | {n - a} | {100 * (n - a) / n:.3f} |")
    lines += ["", f"Total avant : {sum(before_tot.values())} ; total après : {sum(after_tot.values())}", ""]

    lines += ["## Détail par règle et par étiquette", ""]
    for rule, cnt in removed_tot.items():
        lines.append(f"- {rule} : " + ("; ".join(f"{k}={v}" for k, v in cnt.most_common()) if cnt else "aucune"))

    const = [c for c in gmin.index if gmin[c] == gmax[c]]
    lines += ["", "## Colonnes numériques constantes (min = max sur tous les fichiers)", "", ", ".join(const) if const else "aucune"]

    out = Path(args.report)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Rapport écrit : {out.resolve()}")


if __name__ == "__main__":
    main()
