"""Découpage temporel train / validation / test (règle validée par 04_split_dryrun.py).

  1. Fenêtre = minute de fin estimée du flux (Timestamp + Flow Duration), sans recouvrement.
  2. Une fenêtre contenant des flux d'attaque porte l'étiquette d'attaque dominante.
  3. Pour chaque classe d'attaque d'un fichier : fenêtres triées dans le temps, 70 % -> train,
     10 % -> validation, 20 % -> test (au moins une fenêtre de validation et une de test
     si la classe compte 6 fenêtres ou plus).
  4. Fenêtres sans attaque : mêmes coupures 70/10/20 sur l'ordre chronologique du fichier.
"""
import numpy as np
import pandas as pd

CUT_TRAIN, CUT_VAL = 0.70, 0.80
BUFFER = 0
MIN_WINDOWS_FOR_VAL = 6


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
