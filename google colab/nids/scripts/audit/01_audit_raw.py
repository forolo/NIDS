"""Étape 1 - Audit des fichiers CSV bruts de CIC-IDS2017 (version 3 du script).

Usage (depuis la racine du projet) :
    python scripts/audit/01_audit_raw.py
    python scripts/audit/01_audit_raw.py --raw-dir data/raw/csv --out reports/audit_raw.md

Le script ne modifie aucun fichier de données. Il écrit un rapport Markdown.
Les lignes entièrement vides sont comptées à part, puis exclues des autres statistiques.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# Noms possibles des colonnes de métadonnées selon la version du jeu
META_ALIASES = {
    "Flow ID": ["Flow ID"],
    "IP source": ["Source IP", "Src IP"],
    "Port source": ["Source Port", "Src Port"],
    "IP destination": ["Destination IP", "Dst IP"],
    "Port destination": ["Destination Port", "Dst Port"],
    "Protocole": ["Protocol"],
    "Horodatage": ["Timestamp"],
}
# Formats d'horodatage rencontrés selon les fichiers (jour/mois/année)
TS_FORMATS = [
    "%d/%m/%Y %H:%M:%S",
    "%d/%m/%Y %H:%M",
    "%d/%m/%Y %I:%M:%S %p",
    "%d/%m/%Y %I:%M %p",
]


def read_csv_robust(path: Path) -> pd.DataFrame:
    # Les fichiers CIC-IDS2017 ne sont pas toujours en UTF-8 : repli sur latin-1.
    try:
        return pd.read_csv(path, low_memory=False)
    except UnicodeDecodeError:
        return pd.read_csv(path, low_memory=False, encoding="latin-1")


def find_col(df, names):
    lower = {c.lower(): c for c in df.columns}
    for n in names:
        if n.lower() in lower:
            return lower[n.lower()]
    return None


def parse_timestamps(series):
    """Essaie chaque format et garde celui qui échoue le moins."""
    best_fmt, best_ts = None, None
    for fmt in TS_FORMATS:
        ts = pd.to_datetime(series, format=fmt, errors="coerce")
        if best_ts is None or ts.isna().sum() < best_ts.isna().sum():
            best_fmt, best_ts = fmt, ts
    return best_fmt, best_ts


def audit_file(path: Path):
    df = read_csv_robust(path)
    raw_cols = list(df.columns)
    df.columns = [str(c).strip() for c in df.columns]
    info = {"fichier": path.name, "lignes_brutes": len(df), "colonnes": df.shape[1]}

    empty = df.isna().all(axis=1)
    info["lignes_entierement_vides"] = int(empty.sum())
    df = df.loc[~empty].reset_index(drop=True)
    info["lignes_non_vides"] = len(df)

    info["colonnes_dupliquees"] = [c for c in raw_cols if str(c).strip().endswith(".1")]
    info["colonnes_avec_espaces"] = sum(1 for c in raw_cols if str(c) != str(c).strip())

    meta = {k: find_col(df, v) for k, v in META_ALIASES.items()}
    info["metadonnees_trouvees"] = {k: v for k, v in meta.items() if v}
    info["metadonnees_absentes"] = [k for k, v in meta.items() if not v]

    label_col = find_col(df, ["Label"])
    info["colonne_label"] = label_col
    lab = df[label_col].astype(str).str.strip() if label_col else pd.Series([""] * len(df))
    labels = lab.value_counts()

    num = df.select_dtypes(include=[np.number])
    info["colonnes_numeriques"] = num.shape[1]
    info["colonnes_non_numeriques"] = [c for c in df.columns if c not in num.columns]

    arr = num.to_numpy(dtype=float, na_value=np.nan)
    inf_mask = np.isinf(arr)
    nan_mask = np.isnan(arr)
    info["valeurs_inf"] = int(inf_mask.sum())
    info["valeurs_nan"] = int(df.isna().sum().sum())
    info["colonnes_avec_inf"] = {c: int(n) for c, n in zip(num.columns, inf_mask.sum(axis=0)) if n}
    info["colonnes_avec_nan"] = {c: int(n) for c, n in df.isna().sum().items() if n}
    bad_rows = (inf_mask | nan_mask).any(axis=1)
    info["lignes_avec_inf_ou_nan"] = int(bad_rows.sum())
    info["lignes_inf_ou_nan_par_etiquette"] = lab[bad_rows].value_counts().to_dict()

    neg = {}
    for c in num.columns:
        col = num[c]
        n_neg = int((col < 0).sum())
        if n_neg:
            neg[c] = f"{n_neg} lignes (min={col.min():g}, valeur -1 : {int((col == -1).sum())})"
    info["lignes_negatives_par_colonne"] = neg

    dup = df.duplicated()
    info["lignes_dupliquees"] = int(dup.sum())
    info["doublons_par_etiquette"] = lab[dup].value_counts().to_dict()

    ts_col = meta["Horodatage"]
    if ts_col:
        info["exemples_timestamp"] = df[ts_col].astype(str).head(3).tolist()
        fmt, ts = parse_timestamps(df[ts_col].astype(str))
        info["format_horodatage_detecte"] = fmt
        info["timestamps_non_analysables"] = int(ts.isna().sum())
        ok = ts.dropna()
        if len(ok):
            info["timestamp_min"] = str(ok.min())
            info["timestamp_max"] = str(ok.max())
            info["heures_presentes"] = sorted(int(h) for h in ok.dt.hour.unique())
            info["fichier_trie_chronologiquement"] = bool(ok.is_monotonic_increasing)

    hosts, label_src = set(), {}
    if meta["IP source"] and meta["IP destination"]:
        hosts = set(df[meta["IP source"]].astype(str)) | set(df[meta["IP destination"]].astype(str))
        info["hotes_distincts"] = len(hosts)
        for name, g in df.groupby(lab):
            label_src[name] = set(g[meta["IP source"]].astype(str))
    return info, labels, list(df.columns), hosts, label_src


def fmt_value(v):
    if isinstance(v, dict):
        return "; ".join(f"{k}={n}" for k, n in v.items()) if v else "aucun"
    return str(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="data/raw/csv")
    ap.add_argument("--out", default="reports/audit_raw.md")
    args = ap.parse_args()

    raw_dir = Path(args.raw_dir)
    files = sorted(raw_dir.glob("*.csv"))
    if not files:
        raise SystemExit(f"Aucun fichier CSV trouvé dans {raw_dir.resolve()}")

    lines = ["# Audit des données brutes CIC-IDS2017 (script v3)", ""]
    total_labels = pd.Series(dtype=int)
    schemas, all_hosts, all_label_src = {}, set(), {}
    for f in files:
        print(f"Lecture de {f.name} ...")
        info, labels, cols, hosts, label_src = audit_file(f)
        schemas[f.name] = cols
        all_hosts |= hosts
        for k, v in label_src.items():
            all_label_src.setdefault(k, set()).update(v)
        total_labels = total_labels.add(labels, fill_value=0)
        lines += [f"## {info['fichier']}", ""]
        for k, v in info.items():
            if k != "fichier":
                lines.append(f"- {k} : {fmt_value(v)}")
        lines += ["", "Étiquettes (lignes non vides) :", ""]
        lines += [f"- {k} : {int(v)}" for k, v in labels.items()]
        lines.append("")

    lines += ["## Total des étiquettes (tous fichiers, lignes non vides)", ""]
    total = int(total_labels.sum())
    for k, v in total_labels.sort_values(ascending=False).items():
        extra = f", IP sources distinctes : {len(all_label_src[k])}" if k in all_label_src else ""
        lines.append(f"- {k} : {int(v)} ({100 * v / total:.4f} %){extra}")
    lines += ["", f"Total : {total}"]
    if all_hosts:
        lines.append(f"Hôtes distincts (IP source ou destination, tous fichiers) : {len(all_hosts)}")

    lines += ["", "## Cohérence des colonnes entre fichiers", ""]
    ref_name, ref_cols = next(iter(schemas.items()))
    same = all(c == ref_cols for c in schemas.values())
    lines.append(f"Colonnes identiques dans tous les fichiers : {same} (référence : {ref_name}, {len(ref_cols)} colonnes)")
    lines += ["", "## Liste des colonnes (fichier de référence)", ""]
    lines += [f"{i + 1}. {c}" for i, c in enumerate(ref_cols)]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Rapport écrit : {out.resolve()}")


if __name__ == "__main__":
    main()
