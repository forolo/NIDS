"""Étape 6 (analyse) - Matrice de confusion centrée sur les classes d'attaque.

À partir des prédictions d'une tête (data/embeddings/pred{tag}_{ensemble}.npy, écrites par
train_temporal.py --save-preds) et des vraies étiquettes (data/graphs/{ensemble}.npz), écrit :
  - la matrice de confusion complète (effectifs et pourcentages par ligne) ;
  - pour chaque classe d'attaque : rappel exact, part détectée (prédite autre que BENIGN), part manquée
    (prédite BENIGN), part confondue avec une autre attaque, principales classes de confusion, précision, F1 ;
  - des résumés limités aux classes d'attaque (F1 macro et rappel macro sans BENIGN) ;
  - la même lecture par famille d'attaque (regroupement par nom, voir FAMILLES) ;
  - l'origine des fausses alertes (flux BENIGN prédits comme attaque).
Numpy seul : aucun GPU nécessaire.

Usage (depuis la racine du projet) :
    python -m src.eval.confusion_analysis --pred-tag _tp_f10_s0 --split test
"""
import argparse
import json
from pathlib import Path

import numpy as np

FAMILLES = {
    "BENIGN": "BENIGN", "Bot": "Bot", "DDoS": "DDoS",
    "DoS GoldenEye": "DoS", "DoS Hulk": "DoS", "DoS Slowhttptest": "DoS", "DoS slowloris": "DoS",
    "FTP-Patator": "Force brute (Patator)", "SSH-Patator": "Force brute (Patator)",
    "Heartbleed": "Heartbleed", "Infiltration": "Infiltration", "PortScan": "PortScan",
    "Web Attack - Brute Force": "Attaque web", "Web Attack - Sql Injection": "Attaque web", "Web Attack - XSS": "Attaque web",
}


def confusion(y, p, K):
    cm = np.zeros((K, K), dtype=np.int64)
    np.add.at(cm, (y, p), 1)
    return cm


def f1_from(cm, k):
    tp, sup, pk = cm[k, k], cm[k].sum(), cm[:, k].sum()
    rec = tp / sup if sup else float("nan")
    prec = tp / pk if pk else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if pk and sup and (prec + rec) > 0 else 0.0
    return rec, prec, f1


def md_matrix(cm, labels, pct):
    head = "| vrai \\ prédit | " + " | ".join(labels) + " | total |"
    out = [head, "| --- | " + " | ".join(["---"] * (len(labels) + 1)) + " |"]
    for i, lab in enumerate(labels):
        tot = cm[i].sum()
        cells = [("" if cm[i, j] == 0 else (f"{100 * cm[i, j] / tot:.1f}" if pct else str(cm[i, j]))) for j in range(len(labels))]
        out.append(f"| {lab} | " + " | ".join(cells) + f" | {tot} |")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph-dir", default="data/graphs")
    ap.add_argument("--emb-dir", default="data/embeddings")
    ap.add_argument("--pred-tag", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    gd = Path(args.graph_dir)
    meta = json.loads((gd / "meta.json").read_text(encoding="utf-8"))
    names = [k for k, _ in sorted(meta["label_id"].items(), key=lambda t: t[1])]
    K = len(names)
    y = np.load(gd / f"{args.split}.npz")["y"].astype(np.int64)
    p = np.load(Path(args.emb_dir) / f"pred{args.pred_tag}_{args.split}.npy").astype(np.int64)
    cm = confusion(y, p, K)
    present = [k for k in range(K) if cm[k].sum() > 0]
    att = [k for k in present if k != 0]

    L = [f"# Matrice de confusion centrée sur les attaques ({args.split})", "",
         f"- prédictions : {args.pred_tag} ; flux : {len(y)} ; classes d'attaque présentes : {len(att)}", ""]
    rec_f = [f1_from(cm, k) for k in att]
    recs = [r[0] for r in rec_f]
    f1s = [r[2] for r in rec_f]
    det = [1 - cm[k, 0] / cm[k].sum() for k in att]
    atk_tot, atk_det = cm[att].sum(), sum(cm[k].sum() - cm[k, 0] for k in att)
    L += ["## Résumés limités aux classes d'attaque", "",
          f"- F1 macro des classes d'attaque (sans BENIGN) : {np.mean(f1s):.4f}",
          f"- rappel macro des classes d'attaque (sans BENIGN) : {np.mean(recs):.4f}",
          f"- part détectée macro (attaque prédite autre que BENIGN) : {np.mean(det):.4f}",
          f"- détection binaire globale (flux d'attaque non prédits BENIGN) : {atk_det / atk_tot:.4f} ({atk_tot - atk_det} flux d'attaque manqués sur {atk_tot})",
          f"- fausses alertes : {cm[0, 1:].sum()} flux BENIGN prédits comme attaque, soit {100 * cm[0, 1:].sum() / cm[0].sum():.3f} % de BENIGN", "",
          "## Par classe d'attaque", "",
          "| Classe | Support | Rappel exact | Détectée (non BENIGN) | Manquée (BENIGN) | Confondue avec une autre attaque | Précision | F1 | Principales confusions |",
          "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for k in att:
        sup = cm[k].sum()
        rec, prec, f1 = f1_from(cm, k)
        other = sup - cm[k, k] - cm[k, 0]
        dest = sorted(((cm[k, j], j) for j in range(K) if j != k and cm[k, j] > 0), reverse=True)[:3]
        ds = "; ".join(f"{names[j]} {100 * c / sup:.1f} %" for c, j in dest) or "-"
        L.append(f"| {names[k]} | {sup} | {rec:.3f} | {1 - cm[k, 0] / sup:.3f} | {cm[k, 0] / sup:.3f} | {other / sup:.3f} | "
                 f"{'-' if np.isnan(prec) else f'{prec:.3f}'} | {f1:.3f} | {ds} |")

    fam = {}
    for k, n in enumerate(names):
        fam.setdefault(FAMILLES.get(n, n), []).append(k)
    fn = list(fam)
    fm = np.zeros((len(fn), len(fn)), dtype=np.int64)
    for a, ka in enumerate(fn):
        for b, kb in enumerate(fn):
            fm[a, b] = cm[np.ix_(fam[ka], fam[kb])].sum()
    L += ["", "## Lecture par famille d'attaque", "",
          "Regroupement par nom (modifiable dans FAMILLES) : DoS = GoldenEye, Hulk, Slowhttptest, slowloris ; Patator = FTP, SSH ; Attaque web = Brute Force, SQL Injection, XSS.", "",
          "| Famille | Support | Rappel de famille (prédit dans la même famille) | Manquée (BENIGN) |", "| --- | --- | --- | --- |"]
    for a, ka in enumerate(fn):
        if ka == "BENIGN" or fm[a].sum() == 0:
            continue
        L.append(f"| {ka} | {fm[a].sum()} | {fm[a, a] / fm[a].sum():.3f} | {fm[a, fn.index('BENIGN')] / fm[a].sum():.3f} |")

    L += ["", "## Origine des fausses alertes (BENIGN prédit comme attaque)", ""]
    fa = sorted(((cm[0, j], j) for j in range(1, K) if cm[0, j] > 0), reverse=True)
    L += [f"- {names[j]} : {c} flux" for c, j in fa[:10]] or ["- aucune"]

    short = [n if len(n) <= 14 else n[:13] + "." for n in names]
    L += ["", "## Matrice de confusion (effectifs)", ""] + md_matrix(cm, short, False)
    L += ["", "## Matrice de confusion (% de chaque ligne)", ""] + md_matrix(cm, short, True)
    rp = Path(f"reports/confusion{args.tag}.md")
    rp.parent.mkdir(parents=True, exist_ok=True)
    rp.write_text("\n".join(L), encoding="utf-8")
    print(f"Rapport écrit : {rp.resolve()}")


if __name__ == "__main__":
    main()
