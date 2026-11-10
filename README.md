# NIDS hybride auto-supervisé (E-GraphSAGE attentionnel + GCL + LSTM + XGBoost + XAI)

Projet : D:/ids

## Arborescence

| Chemin | Contenu |
| --- | --- |
| data/raw/csv | Données brutes CIC-IDS2017 |
| data/raw/csvnormalisees | Données normalisées |
| scripts/audit | Scripts d'audit |
| src/ingestion | Ingestion des flux |
| src/models/egnn | E-GraphSAGE attentionnel |
| src/models/gcl | Apprentissage contrastif de graphes |
| src/models/lstm | Agrégation temporelle |
| src/models/xai | Explicabilité (TreeSHAP, GNNExplainer) |
| reports | Rapports |
| models/checkpoints | Points de contrôle |

## Pipeline

1. CIC-IDS2017 prétraité
2. Graphes temporels avec attributs d'arêtes
3. E-GraphSAGE attentionnel + GCL
4. Agrégation temporelle
5. XGBoost
6. Évaluation et optimisation
7. Inférence en temps réel
8. Explicabilité et robustesse

Les emplacements des étapes 2, 5, 6, 7 et du module de robustesse restent à définir.
