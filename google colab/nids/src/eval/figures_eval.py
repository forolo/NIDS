"""Figures d'évaluation (test) : matrice de confusion, courbes ROC (AUC), courbes par époque.

Entrées (numpy, matplotlib, scikit-learn ; aucun GPU) :
  - data/graphs/{split}.npz (étiquettes) et data/graphs/meta.json (noms de classes) ;
  - data/embeddings/logit{tag}_{split}.npy : logits de la tête (train_temporal.py --save-logits) ;
  - optionnel : reports/curves{curves_tag}.csv (train_temporal.py --curves).
Sorties dans reports/figures/ :
  - confusion{tag}.png : matrice de confusion (pourcentage par ligne, effectifs dans les cases non vides) ;
  - roc{tag}.png : courbe ROC binaire (attaque contre BENIGN, score = 1 - probabilité de BENIGN) et courbes un-contre-reste par classe d'attaque ;
  - courbes{curves_tag}.png : perte d'entraînement, perte de validation, perte de test, exactitude (si le fichier existe) ;
  - roc{tag}.md : AUC binaire, AUC par classe (classes d'effectif >= 1 et < total), moyenne sur les classes d'attaque.
La perte et l'exactitude de test sont tracées pour affichage seulement : elles n'ont servi à aucun choix.

Usage (depuis la racine du projet) :
    python -m src.eval.figures_eval --logit-tag _tl_f10_s0 --tag _s0
    python -m src.eval.figures_eval --logit-tag _cv_s0 --curves-tag _cv_s0 --tag _cv_s0
"""
import argparse
import json
from pathlib import Path

import numpy as np


def softmax(lg):
    lg = lg.astype("float64")
    lg -= lg.max(axis=1, keepdims=True)
    e = np.exp(lg)
    return e / e.sum(axis=1, keepdims=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph-dir", default="data/graphs")
    ap.add_argument("--emb-dir", default="data/embeddings")
    ap.add_argument("--logit-tag", required=True)
    ap.add_argument("--curves-tag", default="")
    ap.add_argument("--split", default="test")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_auc_score, roc_curve

    out = Path("reports/figures"); out.mkdir(parents=True, exist_ok=True)
    gd = Path(args.graph_dir)
    meta = json.loads((gd / "meta.json").read_text(encoding="utf-8"))
    names = [k for k, _ in sorted(meta["label_id"].items(), key=lambda t: t[1])]
    K = len(names)
    y = np.load(gd / f"{args.split}.npz")["y"].astype(np.int64)
    lg = np.load(Path(args.emb_dir) / f"logit{args.logit_tag}_{args.split}.npy")
    assert len(lg) == len(y), f"{len(lg)} logits pour {len(y)} flux"
    p = softmax(lg)
    pred = lg.argmax(axis=1)

    # --- matrice de confusion
    cm = np.zeros((K, K), dtype=np.int64)
    np.add.at(cm, (y, pred), 1)
    present = [k for k in range(K) if cm[k].sum() > 0]
    cmp_ = cm[np.ix_(present, present)]
    pct = 100 * cmp_ / cmp_.sum(axis=1, keepdims=True)
    lab = [names[k] for k in present]
    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(pct, cmap="Blues", vmin=0, vmax=100)
    ax.set_xticks(range(len(lab))); ax.set_xticklabels(lab, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(lab))); ax.set_yticklabels(lab, fontsize=8)
    ax.set_xlabel("Classe prédite"); ax.set_ylabel("Classe réelle")
    for i in range(len(lab)):
        for j in range(len(lab)):
            if cmp_[i, j] > 0:
                ax.text(j, i, str(cmp_[i, j]), ha="center", va="center", fontsize=6,
                        color="white" if pct[i, j] > 50 else "black")
    fig.colorbar(im, ax=ax, label="% de la classe réelle")
    ax.set_title(f"Matrice de confusion ({args.split}, {len(y)} flux)")
    fig.tight_layout(); fig.savefig(out / f"confusion{args.tag}.png", dpi=150); plt.close(fig)

    # --- ROC
    ben = names.index("BENIGN")
    is_att = (y != ben).astype(int)
    s_att = 1.0 - p[:, ben]
    auc_bin = roc_auc_score(is_att, s_att)
    fpr, tpr, _ = roc_curve(is_att, s_att)
    fig, axs = plt.subplots(1, 2, figsize=(13, 5.5))
    axs[0].plot(fpr, tpr, color="#1f5fbf", lw=2, label=f"attaque contre BENIGN (AUC = {auc_bin:.4f})")
    axs[0].plot([0, 1], [0, 1], "--", color="grey", lw=1)
    axs[0].set_xlabel("Taux de faux positifs"); axs[0].set_ylabel("Taux de vrais positifs")
    axs[0].set_title("ROC binaire"); axs[0].legend(loc="lower right"); axs[0].grid(alpha=0.3)
    rows, aucs = [], []
    for k in present:
        if k == ben:
            continue
        yk = (y == k).astype(int)
        if yk.sum() == 0 or yk.sum() == len(yk):
            continue
        a = roc_auc_score(yk, p[:, k]); aucs.append(a); rows.append((names[k], int(yk.sum()), a))
        f_, t_, _ = roc_curve(yk, p[:, k])
        axs[1].plot(f_, t_, lw=1.2, label=f"{names[k]} ({a:.3f})")
    axs[1].plot([0, 1], [0, 1], "--", color="grey", lw=1)
    axs[1].set_xlabel("Taux de faux positifs"); axs[1].set_ylabel("Taux de vrais positifs")
    axs[1].set_title("ROC un-contre-reste par classe d'attaque (AUC)"); axs[1].legend(fontsize=6, loc="lower right"); axs[1].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out / f"roc{args.tag}.png", dpi=150); plt.close(fig)
    L = [f"# Courbes ROC ({args.split})", "",
         f"- logits : {args.logit_tag} ; flux : {len(y)} ; classes d'attaque présentes : {len(rows)}",
         f"- AUC binaire (attaque contre BENIGN, score = 1 - P(BENIGN)) : {auc_bin:.4f}",
         f"- AUC moyenne un-contre-reste sur les classes d'attaque : {np.mean(aucs):.4f}", "",
         "| Classe | Support | AUC un-contre-reste |", "| --- | --- | --- |"]
    L += [f"| {n} | {s} | {a:.4f} |" for n, s, a in rows]
    L += ["", "L'AUC un-contre-reste utilise la probabilité softmax de la classe ; elle ne dépend pas du seuil de décision."]
    Path(f"reports/roc{args.tag}.md").write_text("\n".join(L), encoding="utf-8")

    # --- courbes par époque
    cp = Path(f"reports/curves{args.curves_tag}.csv")
    if args.curves_tag and cp.exists():
        d = np.genfromtxt(cp, delimiter=",", names=True)
        fig, axs = plt.subplots(1, 2, figsize=(13, 5))
        axs[0].plot(d["epoch"], d["train_loss"], label="entraînement (pondérée)")
        axs[0].plot(d["epoch"], d["val_loss"], label="validation (étiquetés)")
        axs[0].plot(d["epoch"], d["test_loss"], label="test (affichage seulement)")
        axs[0].set_yscale("log"); axs[0].set_xlabel("Époque"); axs[0].set_ylabel("Perte (échelle log)"); axs[0].legend(); axs[0].grid(alpha=0.3)
        axs[1].plot(d["epoch"], d["val_acc"], label="validation (étiquetés)")
        axs[1].plot(d["epoch"], d["test_acc"], label="test (affichage seulement)")
        axs[1].set_xlabel("Époque"); axs[1].set_ylabel("Exactitude"); axs[1].legend(); axs[1].grid(alpha=0.3)
        fig.suptitle("Tête temporelle : perte et exactitude par époque")
        fig.tight_layout(); fig.savefig(out / f"courbes{args.curves_tag}.png", dpi=150); plt.close(fig)
    print(f"figures écrites dans {out.resolve()} ; rapport reports/roc{args.tag}.md")


if __name__ == "__main__":
    main()
