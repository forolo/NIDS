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

Toutes les valeurs ci-dessous sont mesurées sur l'ensemble de **test** (448 559 flux, 490 fenêtres d'une minute) avec **10 % d'étiquettes** pour l'entraînement. Elles proviennent des rapports de `reports/`, cités sous chaque tableau. Moyenne ± écart-type (n-1) sur les graines.

### Performances du système complet (E-GraphSAGE + GCL + LSTM), 10 graines

| Métrique | Valeur |
| --- | --- |
| F1 macro, 14 classes d'attaque | 0,7706 ± 0,0478 |
| Rappel macro, 14 classes d'attaque | 0,7959 ± 0,0493 |
| F1 macro, 11 classes de 100 flux ou plus | 0,8980 ± 0,0250 |
| Rappel macro, 11 classes de 100 flux ou plus | 0,9199 ± 0,0219 |
| Détection binaire (attaque contre BENIGN) | 0,9908 ± 0,0083 |
| Taux de fausses alertes | 0,0005 ± 0,0002 |
| Exactitude globale (accuracy), graine 0 | 0,9977 |
| AUC binaire (attaque contre BENIGN), graine 0 | 0,9999 |
| AUC un-contre-reste moyenne, classes d'attaque, graine 0 | 0,9981 |

Sources : `reports/attack_seeds_tl_f10_10.md`, `reports/roc_s0.md`, `reports/latency.md` (exactitude de la prédiction contre les vraies étiquettes). Les classes rares (Heartbleed : 2 flux, Infiltration : 6, Web Attack - Sql Injection : 5) sont exclues du F1 macro à 11 classes. Leurs rappels sont très variables d'une graine à l'autre.

### Rappel par classe d'attaque (10 graines)

| Classe | Flux de test | Rappel moyen ± écart-type |
| --- | --- | --- |
| Bot | 161 | 0,9180 ± 0,0554 |
| DDoS | 21 678 | 0,9928 ± 0,0196 |
| DoS GoldenEye | 1 059 | 1,0000 ± 0,0000 |
| DoS Hulk | 12 101 | 0,9713 ± 0,0000 |
| DoS Slowhttptest | 1 188 | 0,8895 ± 0,0066 |
| DoS slowloris | 627 | 0,8043 ± 0,1552 |
| FTP-Patator | 1 494 | 0,9993 ± 0,0000 |
| Heartbleed | 2 | 0,7000 ± 0,4216 |
| Infiltration | 6 | 0,0833 ± 0,1620 |
| PortScan | 11 036 | 0,9998 ± 0,0000 |
| SSH-Patator | 1 097 | 0,9511 ± 0,0642 |
| Web Attack - Brute Force | 321 | 0,9009 ± 0,0593 |
| Web Attack - Sql Injection | 5 | 0,2400 ± 0,4195 |
| Web Attack - XSS | 120 | 0,6917 ± 0,1583 |

### Comparaison des approches (5 graines, 10 % d'étiquettes)

| Approche | F1 macro | Détection | Faux positifs |
| --- | --- | --- | --- |
| XGBoost seul | 0,6889 ± 0,0369 | 0,9777 ± 0,0025 | 0,0008 ± 0,0001 |
| E-GraphSAGE entraîné de zéro | 0,7586 ± 0,0171 | 0,9949 ± 0,0017 | 0,0010 ± 0,0009 |
| GCL, encodeur gelé | 0,7318 ± 0,0368 | 0,9286 ± 0,0284 | 0,0023 ± 0,0007 |
| GCL, encodeur dégelé | 0,7766 ± 0,0304 | 0,9939 ± 0,0032 | 0,0007 ± 0,0002 |
| E-GraphSAGE supervisé, 100 % d'étiquettes | 0,7584 ± 0,0363 | 0,9924 ± 0,0045 | 0,0010 ± 0,0001 |

Source : `reports/seeds_summary.md`. Le F1 macro porte sur l'ensemble des classes présentes, bénin compris, et diffère donc du F1 macro des classes d'attaque du premier tableau. Avec 5 graines, le test de Wilcoxon ne peut pas descendre sous p = 0,0625.

### Temps réel (rejeu des flux de test, k = 5, 470 fenêtres mesurées)

| Mesure | GPU (cuda) | CPU |
| --- | --- | --- |
| Calcul du réseau, médiane | 3,30 ms | 2,79 ms |
| Calcul du réseau, p95 | 4,31 ms | 4,78 ms |
| Total par fenêtre, médiane | 3,54 ms | 2,92 ms |
| Total par fenêtre, p99 | 5,84 ms | 7,52 ms |
| Fenêtres au-dessus de 10 ms (total) | 0,2 % (1 sur 470) | 0,4 % (2 sur 470) |
| Débit moyen | 231 879 flux/s | 272 777 flux/s |

Sources : `reports/replay_s0_gpu.md`, `reports/replay_s0_cpu.md` (graine 0 ; les valeurs CPU varient d'une exécution à l'autre : médiane du total de 2,92 à 3,25 ms sur deux exécutions). Les prédictions en ligne sont identiques à l'évaluation par lots sur les 448 559 flux. Le cache d'historique occupe 1,28 Mo en fin de rejeu. La mesure part de flux déjà extraits : capture, extraction des variables et attente de la fin de la minute ne sont pas incluses.

### Explicabilité

| Mesure | Valeur |
| --- | --- |
| Accord global de l'imitateur XGBoost avec la tête | 0,9945 |
| Accord binaire (attaque ou non) | 0,9980 |
| Accord sur les flux classés attaque par la tête | 0,9586 |
| Écart maximal du contrôle d'additivité TreeSHAP | 2,1e-05 |
| Variables expliquées par l'imitateur | 74 (63 physiques + 11 de graphe et de temps) |
| Flux expliqués par TreeSHAP | 2 209 (au plus 200 par classe d'attaque prédite) |

Sources : `reports/surrogate_raw_s0.md`, `reports/xai_shap_s0.md`. TreeSHAP explique l'imitateur et non la tête elle-même : l'accord de 0,9945 mesure la fidélité de cette substitution. Sur les 2 209 flux expliqués (échantillon par classe d'attaque), l'accord est de 0,7755.

## Limites connues

L'audit de fuite (`src/eval/audit_fuite.py`, rapport `reports/audit_fuite.md`) a mis en évidence les points suivants, à garder en tête pour interpréter les performances :

- une part importante des flux d'attaque du test (28,05 %) a un doublon exact dans l'entraînement, sur les 63 variables normalisées ;
- toutes les IP sources d'attaque du test apparaissent déjà dans l'entraînement, ce qui limite la portée du test d'inductivité sur de nouveaux hôtes ;
- pour plusieurs classes (Bot, DoS Hulk, FTP-Patator, PortScan), les ports seuls permettent déjà un rappel élevé ;
- CIC-IDS2017 est un jeu de laboratoire : la généralisation à un autre réseau n'est pas démontrée ;
- les mesures de latence portent sur un rejeu de flux déjà extraits en CSV, pas sur une capture de trafic réel.

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
