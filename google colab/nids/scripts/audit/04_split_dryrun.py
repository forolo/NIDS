"""Étape 1 (suite) - Simulation du découpage temporel train / validation / test.

Aucune donnée n'est écrite : seul un rapport est produit.

Règle simulée (par fichier, c'est-à-dire par jour ou demi-journée) :
  1. Fenêtre = minute de fin estimée du flux (Timestamp + Flow Duration), pas de recouvrement.
  2. Chaque fenêtre contenant au moins un flux d'attaque reçoit l'étiquette d'attaque dominante
     (celle qui compte le plus de flux dans la fenêtre).
  3. Pour chaque classe d'attaque d'un fichier, ses fenêtres sont triées chronologiquement ;
     les 70 % premières -> train, les 10 % suivantes -> validation, les 20 % dernières -> test.
  4. Une classe de moins de 6 fenêtres n'a pas de validation ; sinon la validation et le test
     contiennent au moins une fenêtre. Zone tampon optionnelle (--buffer N fenêtres retirées à la
     fin de chaque ensemble qui précède une coupure ; 0 par défaut : les séquences temporelles
     seront construites à l'intérieur d'un même ensemble).
  5. Fenêtres sans attaque : coupures 70/10/20 sur l'ordre chronologique des fenêtres du fichier,
     indépendamment des attaques.

Usage (depuis la racine du projet) :
    python scripts/audit/04_split_dryrun.py --in-dir data/interim
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

CUT_TRAIN, CUT_VAL = 0.70, 0.80
BUFFER = 0
MIN_WINDOWS_FOR_VAL = 6
SPLITS = ["train", "val", "test", "buffer"]


def cut_ranks(n, buffer=BUFFER):
    """Étiquette d'ensemble pour n éléments ordonnés dans le temps."""
    c1, c2 = int(round(CUT_TRAIN * n)), int(round(CUT_VAL * n))
    if n >= MIN_WINDOWS_FOR_VAL:  # au moins une fenêtre de validation et une de test
        c2 = min(max(c2, c1 + 1), n - 1)
    arr = np.array(["train"] * n, dtype=object)
    arr[c1:] = "val"
    arr[c2:] = "test"
    for c in (c1, c2):
        arr[max(0, c - buffer):c] = "buffer"
    return arr


def assign_file(df, buffer=BUFFER):
    """Retourne df avec les colonnes 'win' (fenêtre) et 'split' (ensemble du flux)."""
    end = df["Timestamp"] + pd.to_timedelta(df["Flow Duration"], unit="us")
    df = df.assign(win=end.dt.floor("min"))
    wins = np.sort(df["win"].unique())
    win_split = pd.Series("", index=wins, dtype=object)
    win_dom = pd.Series("BENIGN", index=wins, dtype=object)

    att = df[df["Label"] != "BENIGN"]
    if len(att) == 0:
        win_split[:] = cut_ranks(len(wins), buffer)
    else:
        cnt = att.groupby(["win", "Label"]).size().reset_index(name="n")
        dom = (cnt.sort_values(["win", "n"], ascending=[True, False])
                  .drop_duplicates("win").set_index("win")["Label"])
        win_dom.loc[dom.index] = dom.values
        for lab in dom.unique():
            ws = np.sort(dom.index[dom.values == lab].values)
            win_split.loc[ws] = cut_ranks(len(ws), buffer)
        benign_w = np.array([w for w in wins if w not in set(dom.index.values)])
        if len(benign_w):
            win_split.loc[benign_w] = cut_ranks(len(benign_w), buffer)
    df["split"] = df["win"].map(win_split)
    return df, win_split, win_dom


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", default="data/interim")
    ap.add_argument("--report", default="reports/split_dryrun.md")
    ap.add_argument("--buffer", type=int, default=BUFFER)
    args = ap.parse_args()

    files = sorted(Path(args.in_dir).glob("*.parquet"))
    if not files:
        raise SystemExit(f"Aucun fichier Parquet dans {Path(args.in_dir).resolve()}")

    flow_rows, win_rows = [], []
    for f in files:
        df = pd.read_parquet(f, columns=["Timestamp", "Label", "Flow Duration"])
        df, win_split, win_dom = assign_file(df, args.buffer)
        flow_rows.append(df.groupby(["Label", "split"]).size().rename("n").reset_index())
        win_rows.append(pd.DataFrame({"Label": win_dom.values, "split": win_split.values}))
        print(f"{f.name} : {len(win_split)} fenêtres")

    def table(rows, unit):
        t = pd.concat(rows)
        t = t.groupby(["Label", "split"]).agg(n=("n", "sum")) if "n" in t else \
            t.groupby(["Label", "split"]).size().rename("n").to_frame()
        t = t["n"].unstack(fill_value=0).reindex(columns=SPLITS, fill_value=0)
        t["total"] = t.sum(axis=1)
        t = t.sort_values("total", ascending=False)
        out = [f"| Étiquette | {' | '.join(SPLITS)} | total |", "| --- | " + " | ".join(["---"] * (len(SPLITS) + 1)) + " |"]
        for lab, r in t.iterrows():
            out.append(f"| {lab} | " + " | ".join(str(int(r[s])) for s in SPLITS) + f" | {int(r['total'])} |")
        return out

    lines = ["# Simulation du découpage temporel (aucune donnée écrite)", "",
             f"Zone tampon : {args.buffer} fenêtres ; coupures : {round(CUT_TRAIN*100)} / {round((CUT_VAL-CUT_TRAIN)*100)} / {round((1-CUT_VAL)*100)} %", "",
             "## Flux par étiquette et par ensemble", ""]
    lines += table(flow_rows, "flux")
    lines += ["", "## Fenêtres d'une minute par étiquette dominante et par ensemble", ""]
    lines += table(win_rows, "fenêtres")
    out = Path(args.report)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Rapport écrit : {out.resolve()}")


if __name__ == "__main__":
    main()
