# NIDS sous Google Colab

## Contenu
- `NIDS_Colab.ipynb` : carnet à ouvrir dans Colab (exécution dans l'ordre, affichage des métriques à la fin).
- `nids/` : le code source du projet (celui de votre dépôt, sans données ni points de contrôle).
- `ORDRE_EXECUTION.md` : tableau de l'ordre d'exécution des scripts.
- `requirements_colab.txt` : paquets installés en plus de ceux de Colab.

## Mise en place
1. Sur Drive : créer `MyDrive/NIDS/`, y déposer `NIDS_Colab.zip` et un dossier `csv/` contenant les 8 CSV de CIC-IDS2017 (version avec `Flow ID`, `Source IP`, `Destination IP`, `Timestamp`).
2. Extraire le ZIP sur votre ordinateur, téléverser `NIDS_Colab.ipynb` dans Colab (ou l'ouvrir depuis Drive) et choisir un GPU.
3. Exécuter les cellules de haut en bas. Paramètres dans la section 0 : `SEEDS`, `DEVICE`, `SMOKE`, `FORCE`.

## Remarques
- Le calcul se fait sur le disque local de Colab (`/content/NIDS_Colab/nids/{data,models,reports}`) ; après chaque étape, les fichiers nouveaux ou modifiés sont copiés (copie incrémentale) vers `MyDrive/NIDS/work/{data,models,reports}`. Au démarrage d'une nouvelle session, Drive est recopié vers le disque local : une session interrompue reprend là où elle s'était arrêtée (étape sautée si ses sorties existent). Les CSV sont copiés une fois sur le disque local (`/content/nids_csv`).
- Si une session précédente (ancienne version du carnet) a laissé des raccourcis dans `/content/NIDS_Colab/nids`, le carnet les supprime au démarrage (le contenu sur Drive n'est pas touché). Pour repartir proprement : *Exécution > Déconnecter et supprimer l'environnement d'exécution*, puis réexécuter.
- `REFRESH_CODE = True` réextrait `src/` et `scripts/` depuis l'archive sans toucher aux données, modèles et rapports.
- Les `torch.load` utilisent `weights_only=False` (points de contrôle produits par le projet), compatible avec PyTorch ≥ 2.6.
- Le carnet n'installe que les paquets absents (xgboost, pyarrow, networkx, psutil) et ne met à jour aucun paquet de Colab.
- `SMOKE = True` vérifie la chaîne avec des entraînements réduits. Après un essai, repasser `SMOKE = False` et `FORCE = True`.
- Le carnet n'a pas été exécuté de bout en bout dans l'environnement de rédaction (les bibliothèques torch et xgboost n'y étaient pas installables) : les commandes ont été vérifiées contre les arguments réels des scripts, pas par exécution. Les versions de pandas/numpy de Colab peuvent différer de celles du projet (`requirements.txt` d'origine : pandas 3, numpy 2).
- Les durées et la mémoire nécessaires sur Colab n'ont pas été mesurées.
