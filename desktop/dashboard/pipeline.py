"""Registre des étapes du pipeline NIDS, états par fichiers et exécuteur en arrière-plan.

Les commandes reprennent ORDRE_EXECUTION.md et le carnet NIDS_Colab.ipynb du projet. Aucune commande libre : seules
les commandes construites ici peuvent être lancées, avec des paramètres validés (graines, fraction, dispositif).

Un utilitaire de la bibliothèque standard uniquement (aucune dépendance à installer).
"""
from __future__ import annotations

import glob
import json
import os
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

DEVICES = ("cuda", "cpu")
GROUPS = ["Données", "Entraînement", "Évaluation", "Temps réel", "Explicabilité", "Robustesse et diagnostics"]


# ---------------------------------------------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------------------------------------------
@dataclass
class Config:
    root: Path
    csv_dir: str = "data/raw/csv"
    device: str = "cuda"
    seeds: list = field(default_factory=lambda: [0])
    head_seed: int = 0
    fraction: float = 0.1
    gnn_args: str = ""          # arguments de GNNExplainer relevés dans reports/xai_gnn_tune_s{graine}.md
    free_mode: bool = False     # ignorer l'ordre chronologique
    python: str = sys.executable

    @property
    def ftag(self) -> str:
        return f"f{int(round(self.fraction * 100))}"

    def to_json(self) -> dict:
        return {"csv_dir": self.csv_dir, "device": self.device, "seeds": self.seeds, "head_seed": self.head_seed,
                "fraction": self.fraction, "gnn_args": self.gnn_args, "free_mode": self.free_mode}


def validate_config(d: dict, root: Path) -> Config:
    """Construit une configuration à partir d'un dictionnaire non fiable ; lève ValueError si une valeur est refusée."""
    device = str(d.get("device", "cuda"))
    if device not in DEVICES:
        raise ValueError("dispositif : cuda ou cpu")
    seeds = d.get("seeds", [0])
    if isinstance(seeds, str):
        seeds = [s for s in re.split(r"[\s,;]+", seeds.strip()) if s]
    try:
        seeds = sorted({int(s) for s in seeds})
    except (TypeError, ValueError):
        raise ValueError("graines : entiers séparés par des virgules")
    if not seeds or len(seeds) > 20 or min(seeds) < 0 or max(seeds) > 9999:
        raise ValueError("graines : 1 à 20 entiers entre 0 et 9999")
    head = int(d.get("head_seed", seeds[0]))
    if head not in seeds:
        head = seeds[0]
    try:
        fraction = float(d.get("fraction", 0.1))
    except (TypeError, ValueError):
        raise ValueError("fraction d'étiquettes : nombre")
    if not 0.001 <= fraction <= 1.0:
        raise ValueError("fraction d'étiquettes entre 0,001 et 1")
    csv_dir = str(d.get("csv_dir", "data/raw/csv")).strip()
    if not csv_dir or "\n" in csv_dir or len(csv_dir) > 400:
        raise ValueError("dossier des CSV invalide")
    gnn = str(d.get("gnn_args", "")).strip()
    if gnn and not re.fullmatch(r"(--[a-z][a-z0-9-]*\s+-?[0-9.eE+-]+\s*)+", gnn + " "):
        raise ValueError("arguments GNNExplainer : forme « --option valeur » avec des valeurs numériques")
    return Config(root=root, csv_dir=csv_dir, device=device, seeds=seeds, head_seed=head, fraction=fraction,
                  gnn_args=gnn, free_mode=bool(d.get("free_mode", False)))


# ---------------------------------------------------------------------------------------------------------------
# Registre des étapes
# ---------------------------------------------------------------------------------------------------------------
@dataclass
class Step:
    id: int
    title: str
    group: str
    what: str
    units: list                 # une liste de contrôles par unité (par exemple par graine) ; chaque contrôle = chemin ou (glob, n)
    commands: list              # liste de commandes (liste d'arguments, hors interpréteur)
    optional: bool = False
    depends: list = field(default_factory=list)   # étapes dont les sorties périmées rendent celle-ci « à refaire »
    reports: list = field(default_factory=list)   # fichiers affichés après exécution (relatifs à la racine)
    reads_test: bool = False
    long: bool = False
    needs: str = ""             # condition particulière (texte) ; vide si aucune
    blocked_reason: str = ""    # non vide : le bouton reste verrouillé


def build_steps(c: Config) -> list:
    D, F, FT = c.device, c.fraction, c.ftag
    S = c.seeds
    H = c.head_seed
    PRED = f"_tl_{FT}_s{H}"
    HEAD = f"models/checkpoints/temporal_tl_{FT}_s{H}.pt"
    PREDFILE = f"data/embeddings/pred{PRED}_test.npy"
    sd = [str(s) for s in S]
    multi = len(S) >= 2
    gnn = c.gnn_args.split() if c.gnn_args else []
    steps: list[Step] = []

    def add(**k):
        steps.append(Step(id=len(steps) + 1, **k))

    # ----- Données -----
    add(title="Audit des CSV bruts", group="Données", what="Vérifie formats, valeurs aberrantes et horodatages.",
        units=[["reports/audit_raw.md"]], commands=[["scripts/audit/01_audit_raw.py", "--raw-dir", c.csv_dir]],
        reports=["reports/audit_raw.md"])
    add(title="Nettoyage (simulation)", group="Données", what="Mesure ce que le nettoyage supprimerait, sans rien écrire.",
        units=[["reports/clean_dryrun.md"]], commands=[["scripts/audit/02_clean_dryrun.py", "--raw-dir", c.csv_dir]],
        reports=["reports/clean_dryrun.md"])
    add(title="Nettoyage (appliqué)", group="Données", what="Écrit les données nettoyées (Parquet) dans data/interim.",
        units=[[("data/interim/*.parquet", 8)]],
        commands=[["scripts/audit/02_clean_dryrun.py", "--raw-dir", c.csv_dir, "--apply", "--out-dir", "data/interim"]],
        depends=[2], reports=["reports/clean_dryrun.md"])
    add(title="Épisodes d'attaque", group="Données", what="Repère les épisodes d'attaque par fichier.",
        units=[["reports/episodes.md"]], commands=[["scripts/audit/03_episodes.py", "--in-dir", "data/interim"]],
        depends=[3], reports=["reports/episodes.md"])
    add(title="Découpage (contrôle)", group="Données", what="Contrôle la règle de découpage temporel 70/10/20.",
        units=[["reports/split_dryrun.md"]], commands=[["scripts/audit/04_split_dryrun.py", "--in-dir", "data/interim"]],
        depends=[3], reports=["reports/split_dryrun.md"])
    add(title="Normalisation et découpage 70/10/20", group="Données",
        what="Ajuste la normalisation sur l'entraînement seul et écrit train, val, test.",
        units=[["data/raw/csvnormalisees/train.parquet", "data/raw/csvnormalisees/val.parquet",
                "data/raw/csvnormalisees/test.parquet", "data/raw/csvnormalisees/scaler.json", "reports/normalisation.md"]],
        commands=[["-m", "src.ingestion.normalize", "--in-dir", "data/interim", "--out-dir", "data/raw/csvnormalisees"]],
        depends=[3], reports=["reports/normalisation.md"])
    add(title="Contrôle des caractéristiques", group="Données", what="Vérifie les 63 variables après normalisation.",
        units=[["reports/features_check.md"]], commands=[["scripts/audit/05_features_check.py", "--dir", "data/raw/csvnormalisees"]],
        depends=[6], reports=["reports/features_check.md"])
    add(title="Graphes par fenêtre d'une minute", group="Données", what="Construit le magasin de graphes (train, val, test).",
        units=[["data/graphs/train.npz", "data/graphs/val.npz", "data/graphs/test.npz", "data/graphs/meta.json", "reports/graph_store.md"]],
        commands=[["-m", "src.ingestion.graph_store", "--in-dir", "data/raw/csvnormalisees", "--out-dir", "data/graphs"]],
        depends=[6], reports=["reports/graph_store.md"])

    # ----- Entraînement -----
    add(title="Référence XGBoost sans graphe", group="Entraînement", optional=True,
        what="Référence tabulaire pour mesurer l'apport du graphe.",
        units=[[f"reports/baseline_xgb_{FT}.md"]],
        commands=[["-m", "src.models.baseline_xgb", "--device", D, "--label-fraction", str(F), "--tag", f"_{FT}"]],
        depends=[8], reports=[f"reports/baseline_xgb_{FT}.md"])
    add(title="E-GraphSAGE supervisé", group="Entraînement", what="Encodeur supervisé à attention (point de comparaison).",
        units=[["models/checkpoints/egnn_att.pt"]],
        commands=[["-m", "src.models.egnn.train_egnn", "--device", D, "--tag", "_att"]],
        depends=[8], reports=["reports/egnn_att.md"], long=True)
    add(title="Pré-entraînement contrastif (GCL)", group="Entraînement", what="Apprentissage auto-supervisé de l'encodeur.",
        units=[["models/checkpoints/gcl_att.pt"]],
        commands=[["-m", "src.models.gcl.pretrain_gcl", "--device", D, "--tag", "_att"]],
        depends=[8], reports=["reports/gcl_att.md"], long=True)
    add(title="Affinage avec étiquettes partielles", group="Entraînement",
        what="Ajuste l'encodeur dégelé avec la fraction d'étiquettes choisie, une exécution par graine.",
        units=[[f"models/checkpoints/ft_unfrozen_{FT}_s{s}.pt"] for s in S],
        commands=[["-m", "src.models.gcl.finetune", "--device", D, "--encoder", "models/checkpoints/gcl_att.pt", "--unfreeze",
                   "--label-fraction", str(F), "--seed", str(s), "--tag", f"_unfrozen_{FT}_s{s}"] for s in S],
        depends=[11], reports=[f"reports/ft_unfrozen_{FT}_s{H}.md"], long=True)
    add(title="Tête temporelle (LSTM)", group="Entraînement",
        what="Entraîne la tête LSTM et écrit prédictions et logits, une exécution par graine.",
        units=[[f"models/checkpoints/temporal_tl_{FT}_s{s}.pt", f"data/embeddings/pred_tl_{FT}_s{s}_test.npy"] for s in S],
        commands=[["-m", "src.models.lstm.train_temporal", "--device", D, "--checkpoint", f"models/checkpoints/ft_unfrozen_{FT}_s{s}.pt",
                   "--label-fraction", str(F), "--seed", str(s), "--tag", f"_tl_{FT}_s{s}", "--save-preds", "--save-logits"]
                  + (["--save-embeddings"] if s == S[0] else []) for s in S],
        depends=[12], reports=[f"reports/temporal_tl_{FT}_s{H}.md"], long=True, reads_test=True)
    add(title="Courbes par époque", group="Entraînement", optional=True,
        what="Nouvel entraînement pour tracer les pertes (graine de tête).",
        units=[["reports/curves_cv_s0.csv"]],
        commands=[["-m", "src.models.lstm.train_temporal", "--device", D, "--checkpoint", f"models/checkpoints/ft_unfrozen_{FT}_s{H}.pt",
                   "--label-fraction", str(F), "--seed", str(H), "--tag", "_cv_s0", "--save-logits", "--curves"]],
        depends=[12], long=True, reads_test=True)

    # ----- Évaluation -----
    add(title="XGBoost de fusion", group="Évaluation", what="XGBoost sur attributs et représentations du graphe.",
        units=[[f"reports/xgbfusion_{FT}_s{H}.md"]],
        commands=[["-m", "src.models.xgb_fusion", "--device", D, "--emb-tag", PRED, "--features", "raw+emb",
                   "--label-fraction", str(F), "--seed", str(H), "--tag", f"_{FT}_s{H}"]],
        depends=[13], reports=[f"reports/xgbfusion_{FT}_s{H}.md"], reads_test=True)
    add(title="Imitateur de la tête", group="Évaluation", what="XGBoost entraîné sur les prédictions de la tête (pour TreeSHAP).",
        units=[[f"models/checkpoints/surrogate_s{H}.json"]],
        commands=[["-m", "src.models.surrogate_xgb", "--device", D, "--pred-tag", PRED, "--tag", f"_s{H}"]],
        depends=[13], reports=[f"reports/surrogate_s{H}.md"])
    add(title="Réglage de l'imitateur", group="Évaluation", what="Grille poids de classe et profondeur, choisie sur la validation.",
        units=[[f"models/checkpoints/surrogate_tuned_s{H}.json"]],
        commands=[["-m", "src.models.surrogate_tune", "--device", D, "--pred-tag", PRED, "--tag", f"_s{H}"]],
        depends=[16], reports=[f"reports/surrogate_tune_s{H}.md"])
    add(title="Matrice de confusion", group="Évaluation", what="Confusions par classe d'attaque sur le test.",
        units=[[f"reports/confusion_s{H}.md"]],
        commands=[["-m", "src.eval.confusion_analysis", "--pred-tag", PRED, "--split", "test", "--tag", f"_s{H}"]],
        depends=[13], reports=[f"reports/confusion_s{H}.md"], reads_test=True)
    fig_cmds = [["-m", "src.eval.figures_eval", "--logit-tag", PRED, "--tag", f"_s{H}"]]
    if (c.root / "reports/curves_cv_s0.csv").exists():
        fig_cmds.append(["-m", "src.eval.figures_eval", "--logit-tag", "_cv_s0", "--curves-tag", "_cv_s0", "--tag", "_cv_s0"])
    add(title="Figures (confusion, ROC, courbes)", group="Évaluation", what="Écrit les figures dans reports/figures.",
        units=[[f"reports/figures/confusion_s{H}.png"]], commands=fig_cmds, depends=[13],
        reports=[f"reports/roc_s{H}.md"], reads_test=True)
    add(title="Décalage du seuil BENIGN", group="Évaluation", optional=True,
        what="Seuil choisi sur la validation ; la courbe du test sert de contrôle.",
        units=[["reports/tune_bias.md"]],
        commands=[["-m", "src.eval.tune_bias", "--logit-tag", PRED, "--label-fraction", str(F), "--seed", str(H), "--max-fpr", "0.001"]],
        depends=[13], reports=["reports/tune_bias.md"], reads_test=True)
    multi_cmds = []
    if multi:
        pat = f"_tl_{FT}_s{{}}"
        multi_cmds = [
            ["-m", "src.eval.aggregate_attack", "--pattern", pat, "--seeds", *sd, "--min-support", "100", "--tag", f"_tl_{FT}"],
            ["-m", "src.eval.stability", "--pattern", pat, "--seeds", *sd, "--min-seeds", str(min(3, len(S))), "--tag", f"_tl_{FT}"],
            ["-m", "src.eval.vote", "--pattern", pat, "--seeds", *sd, "--min-support", "100", "--tag", f"_tl_{FT}",
             "--save-tag", f"_vote{len(S)}"],
            ["-m", "src.eval.hs1_classes", "--pattern", pat, "--seeds", ",".join(sd), "--tag", f"_tl_{FT}"],
            ["-m", "src.eval.label_coverage", "--pattern", pat, "--seeds", *sd, "--label-fraction", str(F), "--tag", f"_tl_{FT}"]]
    add(title="Statistiques multi-graines", group="Évaluation",
        what="Agrégation, stabilité des erreurs, vote, HS1 par classe, couverture des étiquettes.",
        units=[[f"reports/attack_seeds_tl_{FT}.md", f"reports/stability_tl_{FT}.md", f"reports/vote_tl_{FT}.md",
                f"reports/hs1_classes_tl_{FT}.md", f"reports/label_coverage_tl_{FT}.md"]],
        commands=multi_cmds, depends=[13], reports=[f"reports/vote_tl_{FT}.md", f"reports/attack_seeds_tl_{FT}.md"],
        reads_test=True, blocked_reason="" if multi else "Demande au moins 2 graines (réglage dans la configuration).")

    # ----- Temps réel -----
    add(title="Rejeu en ligne", group="Temps réel", what="Rejoue le test fenêtre par fenêtre avec le moteur en ligne.",
        units=[[f"reports/replay_s{H}.md"]],
        commands=[["-m", "src.realtime.replay", "--device", D, "--head", HEAD, "--pred-file", PREDFILE, "--tag", f"_s{H}"]],
        depends=[13], reports=[f"reports/replay_s{H}.md"], reads_test=True)
    add(title="Latence", group="Temps réel", what="Mesure du calcul par fenêtre.",
        units=[["reports/latency.md"]],
        commands=[["-m", "src.eval.benchmark_latency", "--device", D, "--surrogate-device", D, "--head", HEAD,
                   "--pred-file", PREDFILE, "--surrogate", f"models/checkpoints/surrogate_s{H}.json"]],
        depends=[13, 16], reports=["reports/latency.md"], reads_test=True)
    ef2 = [["-m", "src.eval.ef2_hosts", "--device", D, "--head", HEAD, "--pred-file", PREDFILE, "--tag", f"_s{H}"]]
    if multi:
        ef2.append(["-m", "src.eval.ef2_novelty", "--pattern", f"_tl_{FT}_s{{}}", "--seeds", ",".join(sd), "--tag", f"_tl_{FT}"])
    add(title="Inductivité (EF2)", group="Temps réel", what="Test d'indépendance à l'identité des hôtes.",
        units=[[f"reports/ef2_hosts_s{H}.md"]] if not multi else [[f"reports/ef2_hosts_s{H}.md", f"reports/ef2_novelty_tl_{FT}.md"]],
        commands=ef2, depends=[13], reports=[f"reports/ef2_hosts_s{H}.md"], reads_test=True)

    # ----- Explicabilité -----
    add(title="TreeSHAP et fidélité", group="Explicabilité", what="Contributions TreeSHAP et fidélité par perturbation (imitateur de base puis réglé).",
        units=[[f"reports/xai_shap_s{H}.md", f"reports/xai_shap_s{H}cls_tuned.md"]],
        commands=[["-m", "src.models.xai.shap_fidelity", "--device", D, "--head", HEAD, "--surrogate", f"models/checkpoints/surrogate_s{H}.json",
                   "--pred-tag", PRED, "--tag", f"_s{H}"],
                  ["-m", "src.models.xai.shap_fidelity", "--device", D, "--head", HEAD, "--surrogate", f"models/checkpoints/surrogate_tuned_s{H}.json",
                   "--pred-tag", PRED, "--tag", f"_s{H}cls_tuned"]],
        depends=[16, 17], reports=[f"reports/xai_shap_s{H}.md"], reads_test=True)
    add(title="Réglage de GNNExplainer", group="Explicabilité", what="Choisit les coefficients sur la validation.",
        units=[[f"reports/xai_gnn_tune_s{H}.md"]],
        commands=[["-m", "src.models.xai.gnn_tune", "--device", D, "--head", HEAD, "--pred-tag", PRED, "--tag", f"_s{H}"]],
        depends=[13], reports=[f"reports/xai_gnn_tune_s{H}.md"], long=True)
    add(title="GNNExplainer", group="Explicabilité", what="Explication par sous-graphe. Les coefficients se saisissent dans la configuration.",
        units=[[f"reports/xai_gnn_s{H}_gnn.md"]],
        commands=[["-m", "src.models.xai.gnn_explainer", "--device", D, "--head", HEAD, "--pred-tag", PRED, *gnn, "--tag", f"_s{H}_gnn"]],
        depends=[13, 26], reports=[f"reports/xai_gnn_s{H}_gnn.md"], reads_test=True,
        needs="" if gnn else "Aucun coefficient saisi : les valeurs par défaut du script seront utilisées.")
    add(title="Figures d'explicabilité", group="Explicabilité", what="Résumé TreeSHAP et explication de sous-graphe.",
        units=[[f"reports/figures/shap_summary_s{H}.png"]],
        commands=[["-m", "src.models.xai.xai_figures", "--shap-tag", f"_s{H}", "--gnn-tag", f"_s{H}_gnn", "--logit-tag", PRED, "--tag", f"_s{H}"]],
        depends=[25, 27], reports=[], reads_test=True)

    # ----- Robustesse -----
    add(title="Robustesse aux perturbations", group="Robustesse et diagnostics",
        what="Flux fantômes, bruit, suppression de flux, perte de fenêtres.",
        units=[[f"reports/adversarial_s{H}.md"]],
        commands=[["-m", "src.realtime.adversarial", "--device", D, "--head", HEAD, "--tag", f"_s{H}"]],
        depends=[13], reports=[f"reports/adversarial_s{H}.md"], reads_test=True, long=True)
    add(title="Surveillance de la dérive", group="Robustesse et diagnostics", what="PSI et incertitude sur dérive injectée.",
        units=[[f"reports/drift_s{H}.md"]],
        commands=[["-m", "src.realtime.drift", "--device", D, "--head", HEAD, "--pred-tag", PRED, "--tag", f"_s{H}"]],
        depends=[13], reports=[f"reports/drift_s{H}.md"], reads_test=True)
    add(title="Diagnostics DDoS (HS1)", group="Robustesse et diagnostics", optional=True,
        what="Volume, contexte et ordre des fenêtres pour la classe DDoS.",
        units=[[f"reports/volume_ddos_s{H}.md", f"reports/context_ddos_s{H}.md", f"reports/order_s{H}_{D}.md"]],
        commands=[["-m", "src.realtime.volume_test", "--device", D, "--head", HEAD, "--class-name", "DDoS", "--tag", f"_ddos_s{H}"],
                  ["-m", "src.realtime.context_test", "--device", D, "--head", HEAD, "--class-name", "DDoS", "--tag", f"_ddos_s{H}"],
                  ["-m", "src.realtime.order_test", "--device", D, "--head", HEAD, "--class-name", "DDoS", "--tag", f"_s{H}_{D}"]],
        depends=[13], reports=[f"reports/volume_ddos_s{H}.md"], reads_test=True)
    add(title="Audit de fuite", group="Robustesse et diagnostics", optional=True,
        what="Contrôles A à F de l'audit de fuite (la liste G reste manuelle).",
        units=[["reports/audit_fuite.md"]],
        commands=[["-m", "src.eval.audit_fuite", "--dir", "data/raw/csvnormalisees", "--out", "reports/audit_fuite.md"]],
        depends=[6, 8], reports=["reports/audit_fuite.md"], reads_test=True)
    return steps


# ---------------------------------------------------------------------------------------------------------------
# États déduits des fichiers
# ---------------------------------------------------------------------------------------------------------------
def _match(root: Path, item):
    """Retourne la liste des fichiers correspondant à un contrôle, et si le contrôle est satisfait."""
    if isinstance(item, tuple):
        pat, n = item
        files = [Path(p) for p in glob.glob(str(root / pat))]
        return files, len(files) >= n
    p = root / item
    return ([p] if p.exists() else []), p.exists()


def unit_status(root: Path, unit) -> tuple:
    ok, mt = True, []
    for it in unit:
        files, good = _match(root, it)
        ok = ok and good
        mt += [f.stat().st_mtime for f in files if f.exists()]
    return ok, (min(mt) if (ok and mt) else None)


def step_state(root: Path, step: Step) -> dict:
    done, times = 0, []
    for u in step.units:
        ok, mt = unit_status(root, u)
        if ok:
            done += 1
            if mt:
                times.append(mt)
    return {"done": done, "total": len(step.units), "complete": done == len(step.units) and bool(step.units),
            "mtime": min(times) if (times and done == len(step.units)) else None}


def compute_states(cfg: Config, steps: list, jobs: "JobStore") -> list:
    base = {s.id: step_state(cfg.root, s) for s in steps}
    out = []
    mandatory = [s for s in steps if not s.optional]
    for s in steps:
        b = base[s.id]
        prev = [p for p in mandatory if p.id < s.id]
        missing = [p for p in prev if not base[p.id]["complete"]]
        stale = bool(b["complete"] and any(base[d]["mtime"] and b["mtime"] and base[d]["mtime"] > b["mtime"] for d in s.depends))
        job = jobs.last(s.id)
        running = jobs.is_running(s.id)
        if running:
            status = "en_cours"
        elif s.blocked_reason and not cfg.free_mode:
            status = "verrouille"
        elif b["complete"] and stale:
            status = "a_refaire"
        elif b["complete"]:
            status = "termine"
        elif missing and not cfg.free_mode:
            status = "verrouille"
        elif job and job.get("status") in ("failed", "stopped"):
            status = "echec" if job["status"] == "failed" else "interrompu"
        else:
            status = "disponible"
        out.append({"id": s.id, "title": s.title, "group": s.group, "what": s.what, "optional": s.optional,
                    "reads_test": s.reads_test, "long": s.long, "status": status, "done": b["done"], "total": b["total"],
                    "needs": s.needs, "blocked_reason": s.blocked_reason,
                    "waiting_for": [f"{p.id}. {p.title}" for p in missing][:3] if status == "verrouille" else [],
                    "last": job, "commands": [" ".join(["python"] + cmd) for cmd in s.commands]})
    return out


# ---------------------------------------------------------------------------------------------------------------
# Exécution en arrière-plan
# ---------------------------------------------------------------------------------------------------------------
class JobStore:
    """Historique des exécutions, persistant dans dashboard/state.json."""

    def __init__(self, path: Path):
        self.path = path
        self.lock = threading.RLock()
        self.data = {}
        if path.exists():
            try:
                self.data = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                self.data = {}
        for k, v in self.data.items():           # tâches « en cours » d'une session précédente : interrompues
            if v.get("status") == "running":
                v["status"] = "stopped"
                v["note"] = "interrompue (l'application a été fermée)"

    def save(self):
        with self.lock:
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.data, ensure_ascii=False, indent=1), encoding="utf-8")
            os.replace(tmp, self.path)

    def set(self, step_id: int, **kw):
        with self.lock:
            self.data.setdefault(str(step_id), {}).update(kw)
            self.save()

    def last(self, step_id: int):
        with self.lock:
            return dict(self.data.get(str(step_id), {})) or None

    def is_running(self, step_id: int) -> bool:
        j = self.last(step_id)
        return bool(j and j.get("status") == "running")


class Runner:
    def __init__(self, root: Path, store: JobStore, logs: Path):
        self.root, self.store, self.logs = root, store, logs
        self.logs.mkdir(parents=True, exist_ok=True)
        self.thread = None
        self.proc = None
        self.step_id = None
        self.stop_flag = threading.Event()
        self.lock = threading.Lock()

    @property
    def busy(self) -> bool:
        return self.thread is not None and self.thread.is_alive()

    def start(self, step: Step, python: str) -> str:
        with self.lock:
            if self.busy:
                raise RuntimeError("Une étape est déjà en cours.")
            if not step.commands:
                raise RuntimeError(step.blocked_reason or "Aucune commande pour cette étape.")
            self.stop_flag.clear()
            ts = time.strftime("%Y%m%d_%H%M%S")
            log = self.logs / f"etape{step.id:02d}_{ts}.log"
            self.step_id = step.id
            self.store.set(step.id, status="running", started=time.strftime("%Y-%m-%d %H:%M:%S"), ended=None, rc=None,
                           log=log.name, note="")
            self.thread = threading.Thread(target=self._run, args=(step, python, log), daemon=True)
            self.thread.start()
            return log.name

    def _run(self, step: Step, python: str, log: Path):
        env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
        rc = 0
        t0 = time.time()
        try:
            with open(log, "w", encoding="utf-8", errors="replace") as fh:
                for cmd in step.commands:
                    if self.stop_flag.is_set():
                        rc = -1
                        break
                    full = [python] + cmd
                    fh.write("$ " + " ".join(full) + "\n")
                    fh.flush()
                    self.proc = subprocess.Popen(full, cwd=self.root, stdout=fh, stderr=subprocess.STDOUT, env=env)
                    rc = self.proc.wait()
                    if self.stop_flag.is_set():
                        rc = -1
                    if rc != 0:
                        break
                fh.write(f"\n[fin] code de retour {rc} ; durée {time.time() - t0:.0f} s\n")
        except Exception as e:   # échec de lancement
            rc = 1
            with open(log, "a", encoding="utf-8") as fh:
                fh.write(f"\n[erreur] {e}\n")
        finally:
            self.proc = None
            status = "ok" if rc == 0 else ("stopped" if self.stop_flag.is_set() else "failed")
            self.store.set(step.id, status=status, ended=time.strftime("%Y-%m-%d %H:%M:%S"), rc=rc,
                           seconds=round(time.time() - t0))

    def stop(self):
        self.stop_flag.set()
        p = self.proc
        if p and p.poll() is None:
            try:
                import psutil
                parent = psutil.Process(p.pid)
                for ch in parent.children(recursive=True):
                    ch.kill()
                parent.kill()
            except Exception:
                p.kill()

    def tail(self, log_name: str, n: int = 80) -> str:
        p = self.logs / Path(log_name).name
        if not p.exists():
            return ""
        try:
            data = p.read_bytes()[-20000:].decode("utf-8", errors="replace")
        except Exception:
            return ""
        return "\n".join(data.splitlines()[-n:])
