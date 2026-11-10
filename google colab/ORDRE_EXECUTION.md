# Ordre d'exécution des scripts (du début à la fin)

Commandes lancées depuis la racine `nids/` ; le carnet `NIDS_Colab.ipynb` les exécute dans cet ordre. `S` = graine ; `<csv>` = dossier des CSV sur Drive.

| # | Étape | Commande | Sorties principales |
|---|-------|----------|---------------------|
| 1 | Audit des CSV | `python scripts/audit/01_audit_raw.py --raw-dir <csv>` | `reports/audit_raw.md` |
| 2 | Nettoyage (simulation) | `python scripts/audit/02_clean_dryrun.py --raw-dir <csv>` | `reports/clean_dryrun.md` |
| 3 | Nettoyage (appliqué) | `python scripts/audit/02_clean_dryrun.py --raw-dir <csv> --apply --out-dir data/interim` | `data/interim/*.parquet` |
| 4 | Épisodes d'attaque | `python scripts/audit/03_episodes.py --in-dir data/interim` | `reports/episodes.md` |
| 5 | Découpage (contrôle) | `python scripts/audit/04_split_dryrun.py --in-dir data/interim` | `reports/split_dryrun.md` |
| 6 | Normalisation et découpage 70/10/20 | `python -m src.ingestion.normalize --in-dir data/interim --out-dir data/raw/csvnormalisees` | `train/val/test.parquet`, `scaler.json`, `reports/normalisation.md` |
| 7 | Contrôle des caractéristiques | `python scripts/audit/05_features_check.py --dir data/raw/csvnormalisees` | `reports/features_check.md` |
| 8 | Graphes par fenêtre | `python -m src.ingestion.graph_store --in-dir data/raw/csvnormalisees --out-dir data/graphs` | `data/graphs/{train,val,test}.npz`, `meta.json`, `reports/graph_store.md` |
| 9 | Référence XGBoost (optionnel) | `python -m src.models.baseline_xgb --label-fraction 0.1 --tag _f10` | `reports/baseline_xgb_f10.md` |
| 10 | E-GraphSAGE supervisé | `python -m src.models.egnn.train_egnn --tag _att` | `models/checkpoints/egnn_att.pt`, `reports/egnn_att.md` |
| 11 | Pré-entraînement GCL | `python -m src.models.gcl.pretrain_gcl --tag _att` | `models/checkpoints/gcl_att.pt`, `reports/gcl_att.md` |
| 12 | Affinage 10 % d'étiquettes | `python -m src.models.gcl.finetune --encoder models/checkpoints/gcl_att.pt --unfreeze --label-fraction 0.1 --seed S --tag _unfrozen_f10_sS` | `ft_unfrozen_f10_sS.pt`, `reports/ft_unfrozen_f10_sS.md` |
| 13 | Tête temporelle LSTM | `python -m src.models.lstm.train_temporal --checkpoint models/checkpoints/ft_unfrozen_f10_sS.pt --label-fraction 0.1 --seed S --tag _tl_f10_sS --save-preds --save-logits [--save-embeddings]` | `temporal_tl_f10_sS.pt`, `data/embeddings/{pred,logit,emb,fe}_tl_f10_sS_*.npy` |
| 14 | Courbes par époque | même commande avec `--tag _cv_s0 --save-logits --curves` | `reports/curves_cv_s0.csv` |
| 15 | XGBoost de fusion | `python -m src.models.xgb_fusion --emb-tag _tl_f10_s0 --features raw+emb` | `reports/xgbfusion_f10_s0.md` |
| 16 | Imitateur de la tête | `python -m src.models.surrogate_xgb --pred-tag _tl_f10_s0 --tag _s0` | `surrogate_s0.json` |
| 17 | Réglage de l'imitateur | `python -m src.models.surrogate_tune --pred-tag _tl_f10_s0 --tag _s0` | `surrogate_tuned_s0.json` |
| 18 | Matrice de confusion | `python -m src.eval.confusion_analysis --pred-tag _tl_f10_s0 --tag _s0` | `reports/confusion_s0.md` |
| 19 | Figures (confusion, ROC, courbes) | `python -m src.eval.figures_eval --logit-tag _tl_f10_s0 --tag _s0` | `reports/figures/*.png`, `reports/roc_s0.md` |
| 20 | Biais de seuil (optionnel) | `python -m src.eval.tune_bias --logit-tag _tl_f10_s0 --max-fpr 0.001` | `reports/tune_bias.md` |
| 21 | Multi-graines (≥ 2 graines) | `aggregate_attack`, `stability`, `vote`, `hs1_classes`, `label_coverage` | `reports/attack_seeds*.md`, `stability*.md`, `vote*.md`, `hs1_classes*.md`, `label_coverage*.md` |
| 22 | Rejeu en ligne | `python -m src.realtime.replay --head models/checkpoints/temporal_tl_f10_s0.pt --pred-file data/embeddings/pred_tl_f10_s0_test.npy` | `reports/replay_s0.md` |
| 23 | Latence | `python -m src.eval.benchmark_latency --head ... --surrogate ...` | `reports/latency.md` |
| 24 | Inductivité (EF2) | `python -m src.eval.ef2_hosts --head ...` (+ `ef2_novelty` si ≥ 2 graines) | `reports/ef2_hosts_s0.md` |
| 25 | TreeSHAP | `python -m src.models.xai.shap_fidelity --head ... --surrogate ... --pred-tag ...` | `reports/xai_shap_s0.md`, `xai_shap_s0cls_tuned.md` |
| 26 | Réglage GNNExplainer | `python -m src.models.xai.gnn_tune --head ... --pred-tag ...` | `reports/xai_gnn_tune_s0.md` |
| 27 | GNNExplainer | `python -m src.models.xai.gnn_explainer --head ... --pred-tag ... [coefficients]` | `reports/xai_gnn_s0_gnn.md`, `xai_gnn_s0_gnn_edges.csv` |
| 28 | Figures XAI | `python -m src.models.xai.xai_figures --shap-tag _s0 --gnn-tag _s0_gnn --logit-tag _tl_f10_s0 --tag _s0` | `shap_summary_s0.png`, `gnn_explication_s0.png` |
| 29 | Robustesse | `python -m src.realtime.adversarial --head ...` | `reports/adversarial_s0.md` |
| 30 | Dérive | `python -m src.realtime.drift --head ... --pred-tag ...` | `reports/drift_s0.md` |
| 31 | Diagnostics HS1 (optionnel) | `volume_test`, `context_test`, `order_test` avec `--class-name DDoS` | `reports/volume_*.md`, `context_*.md`, `order_*.md` |

Dépendances : les étapes 10 à 14 sont les entraînements ; 15 à 31 lisent leurs sorties. Les étapes 25 à 28 utilisent la graine `HEAD_SEED` (par défaut la première de `SEEDS`).
