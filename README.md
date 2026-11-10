# NIDS hybride auto-supervisé : E-GraphSAGE attentionnel + GCL + LSTM + XGBoost + XAI

Système de détection d'intrusions réseau (NIDS) fondé sur les graphes, développé dans le cadre d'un mémoire de master. Il transforme le trafic en graphes temporels (hôtes = nœuds, flux = arêtes), apprend des représentations de façon auto-supervisée, ajoute une mémoire temporelle par hôte, classe les flux avec XGBoost, puis explique les décisions (TreeSHAP, GNNExplainer). Un moteur d'inférence en ligne permet le rejeu des flux en temps réel.

Jeu de données : [CIC-IDS2017](https://www.unb.ca/cic/datasets/ids-2017.html) (non redistribué dans ce dépôt).

## Architecture

```
CIC-IDS2017 (CSV)
   │  nettoyage, découpage temporel 70/10/20, normalisation (sur l'entraînement seul)
   ▼
Graphes par fenêtre d'1 minute (nœuds = IP, arêtes = flux, 63 attributs par arête)
   │
   ▼
E-GraphSAGE attentionnel  ◄── pré-entraînement GCL (contrastif, sans étiquettes)
   │                           puis affinage avec 10 % d'étiquettes
   ▼
Tête temporelle LSTM (historique de k minutes par hôte)
   │
   ├──► XGBoost de fusion (63 attributs + variables de graphe et de temps)
   │
   ├──► Imitateur XGBoost de la tête ──► TreeSHAP (explications globales et par classe)
   │
   ├──► GNNExplainer (sous-graphes et arêtes explicatifs)
   │
   └──► Moteur temps réel : rejeu, latence, dérive, robustesse
```

| Brique | Rôle | Code |
| --- | --- | --- |
| E-GraphSAGE attentionnel | Représentation inductive des flux et des hôtes, agrégation pondérée par attention (option `--no-attention` pour la moyenne) | `src/models/egnn/` |
| GCL | Pré-entraînement contrastif sans étiquettes (deux vues d'une même fenêtre, perte NT-Xent) | `src/models/gcl/` |
| LSTM | Agrégation temporelle de l'historique d'un hôte | `src/models/lstm/` |
| XGBoost | Classifieur de fusion et imitateur de la tête | `src/models/xgb_fusion.py`, `src/models/surrogate_xgb.py` |
| XAI | TreeSHAP sur l'imitateur, GNNExplainer sur le graphe | `src/models/xai/` |
| Temps réel | Moteur d'inférence en ligne, rejeu, dérive, attaques adverses | `src/realtime/` |
| Évaluation | Métriques, intervalles de Wilson, multi-graines, audit de fuite | `src/eval/` |

## Structure du dépôt

```
.
├── dashboard/             Tableau de bord local : lancement des étapes dans l'ordre
├── scripts/
│   ├── aggregate_seeds.py
│   └── audit/             Audit des données (01 à 05)
├── src/
│   ├── ingestion/         Découpage, normalisation, graphes par fenêtre
│   ├── models/            E-GraphSAGE, GCL, LSTM, XGBoost, XAI
│   ├── eval/              Évaluation et analyses
│   └── realtime/          Inférence en ligne et tests de robustesse
├── data/                  Non versionné (CSV bruts, Parquet, graphes, embeddings)
├── models/checkpoints/    Non versionné (points de contrôle)
├── reports/               Rapports Markdown, CSV et figures générés
└── requirements.txt
```

## Installation

Prérequis : Python 3.11 ou plus, GPU NVIDIA recommandé (CUDA). Tous les scripts acceptent `--device cpu`.

```bash
git clone <URL_DU_DEPOT>
cd <NOM_DU_DEPOT>
python -m venv .venv
source .venv/bin/activate        # Windows : .venv\Scripts\activate
pip install -r requirements.txt
```

Les versions de `requirements.txt` incluent PyTorch (build CUDA), PyTorch Geometric et XGBoost. Adaptez la ligne `torch` à votre version de CUDA, ou installez la version CPU. Selon l'encodage du fichier sur votre système, `pip` peut exiger une conversion en UTF-8.

## Données

1. Télécharger les CSV *TrafficLabelling* de CIC-IDS2017 depuis le site de l'Université du Nouveau-Brunswick.
2. Les placer dans un dossier, par exemple `data/raw/csv`.

Le jeu contient 2 830 390 flux après nettoyage, 63 attributs physiques par flux retenus après normalisation et 14 classes d'attaque plus le trafic bénin. Le découpage est chronologique (70 % / 10 % / 20 %), en fenêtres d'une minute.

## Exécution du pipeline

Toutes les commandes se lancent depuis la racine du projet. `S` désigne une graine et `<csv>` le dossier des CSV bruts.

### 1. Préparation des données

```bash
python scripts/audit/01_audit_raw.py --raw-dir <csv>
python scripts/audit/02_clean_dryrun.py --raw-dir <csv>
python scripts/audit/02_clean_dryrun.py --raw-dir <csv> --apply --out-dir data/interim
python scripts/audit/03_episodes.py --in-dir data/interim
python scripts/audit/04_split_dryrun.py --in-dir data/interim
python -m src.ingestion.normalize --in-dir data/interim --out-dir data/raw/csvnormalisees
python scripts/audit/05_features_check.py --dir data/raw/csvnormalisees
python -m src.ingestion.graph_store --in-dir data/raw/csvnormalisees --out-dir data/graphs
```

### 2. Entraînement

```bash
# Référence XGBoost (optionnel)
python -m src.models.baseline_xgb --label-fraction 0.1 --tag _f10

# E-GraphSAGE supervisé
python -m src.models.egnn.train_egnn --tag _att

# Pré-entraînement GCL (sans étiquettes)
python -m src.models.gcl.pretrain_gcl --tag _att

# Affinage avec 10 % d'étiquettes
python -m src.models.gcl.finetune --encoder models/checkpoints/gcl_att.pt --unfreeze \
    --label-fraction 0.1 --seed S --tag _unfrozen_f10_sS

# Tête temporelle LSTM
python -m src.models.lstm.train_temporal --checkpoint models/checkpoints/ft_unfrozen_f10_sS.pt \
    --label-fraction 0.1 --seed S --tag _tl_f10_sS --save-preds --save-logits
```

### 3. Classification finale, imitateur et évaluation

```bash
python -m src.models.xgb_fusion --emb-tag _tl_f10_s0 --features raw+emb
python -m src.models.surrogate_xgb --pred-tag _tl_f10_s0 --tag _s0
python -m src.models.surrogate_tune --pred-tag _tl_f10_s0 --tag _s0
python -m src.eval.confusion_analysis --pred-tag _tl_f10_s0 --tag _s0
python -m src.eval.figures_eval --logit-tag _tl_f10_s0 --tag _s0
```

Pour plusieurs graines (au moins deux), les agrégations se font avec `aggregate_attack`, `stability`, `vote`, `hs1_classes` et `label_coverage` dans `src/eval/`.

### 4. Temps réel

```bash
python -m src.realtime.replay --head models/checkpoints/temporal_tl_f10_s0.pt \
    --pred-file data/embeddings/pred_tl_f10_s0_test.npy
python -m src.eval.benchmark_latency --head <tête> --surrogate <imitateur>
python -m src.realtime.drift --head <tête> --pred-tag _tl_f10_s0
python -m src.realtime.adversarial --head <tête>
```

### 5. Explicabilité

```bash
python -m src.models.xai.shap_fidelity --head <tête> --surrogate <imitateur> --pred-tag _tl_f10_s0
python -m src.models.xai.gnn_tune --head <tête> --pred-tag _tl_f10_s0
python -m src.models.xai.gnn_explainer --head <tête> --pred-tag _tl_f10_s0
python -m src.models.xai.xai_figures --shap-tag _s0 --gnn-tag _s0_gnn --logit-tag _tl_f10_s0 --tag _s0
```

### 6. Audit de fuite (recommandé)

```bash
python -m src.eval.audit_fuite
```

L'ordre complet des 31 étapes, avec leurs sorties, est donné dans `ORDRE_EXECUTION.md`.

## Tableau de bord

Le dossier `dashboard/` fournit une interface locale qui lance les étapes du pipeline dans l'ordre : chaque bouton ne s'active que lorsque les étapes obligatoires précédentes sont terminées. Elle affiche les rapports, les figures et les journaux. Aucune dépendance supplémentaire n'est nécessaire.

```bash
python -m dashboard.server
```

Ouvrir ensuite http://127.0.0.1:8765. Le serveur n'écoute que sur 127.0.0.1.

## Résultats

Les résultats mesurés sont produits dans `reports/` à chaque exécution (rapports Markdown, CSV et figures) : matrices de confusion, courbes ROC, stabilité multi-graines, latence, explicabilité, dérive et robustesse. Ce fichier ne reprend aucun chiffre afin de rester cohérent avec les rapports. Pour citer un résultat, se référer au rapport correspondant et à la graine utilisée.

## Limites connues

L'audit de fuite (`src/eval/audit_fuite.py`, rapport `reports/audit_fuite.md`) a mis en évidence les points suivants, à garder en tête pour interpréter les performances :

- une part importante des flux d'attaque du test (28,05 %) a un doublon exact dans l'entraînement, sur les 63 variables normalisées ;
- toutes les IP sources d'attaque du test apparaissent déjà dans l'entraînement, ce qui limite la portée du test d'inductivité sur de nouveaux hôtes ;
- pour plusieurs classes (Bot, DoS Hulk, FTP-Patator, PortScan), les ports seuls permettent déjà un rappel élevé ;
- CIC-IDS2017 est un jeu de laboratoire : la généralisation à un autre réseau n'est pas démontrée ;
- l'inférence en ligne est un rejeu de flux déjà extraits en CSV, pas une capture de trafic réel.

## Reproductibilité

- Les graines sont explicites (`--seed`), et le protocole utilise plusieurs graines avec intervalles de Wilson à 95 %.
- Les statistiques de normalisation sont calculées sur l'entraînement seul et enregistrées dans `scaler.json`.
- Le découpage est chronologique, jamais aléatoire.
- Les étapes qui lisent le test (évaluation, XAI, robustesse) ne doivent pas servir à régler le modèle. Le réglage se fait sur la validation.

## Citation

```
<Auteur>. <Titre du mémoire>. Mémoire de master, <Établissement>, <Année>.
```

Ce travail s'appuie notamment sur STG-NIDS (apprentissage contrastif spatio-temporel de graphes pour la détection d'intrusions), E-GraphSAGE, GNNExplainer et TreeSHAP.

## Licence

À définir par l'auteur (par exemple MIT). CIC-IDS2017 reste soumis à la licence de ses auteurs.
