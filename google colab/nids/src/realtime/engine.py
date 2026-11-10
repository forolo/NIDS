"""Détecteur en ligne : une fenêtre d'une minute en entrée (flux bruts + identifiants d'hôtes), classes prédites en sortie.

Chemin d'une fenêtre (chaque étape est chronométrée avec synchronisation du périphérique) :
  prep   : indexation locale des hôtes de la fenêtre (np.unique), copie des tableaux vers le périphérique ;
  enc    : encodeur E-GraphSAGE (états z des hôtes de la fenêtre) ;
  hist   : recherche des états des mêmes hôtes dans les k-1 minutes précédentes (cache), assemblage de la séquence ;
  head   : LSTM, tête, argmax, retour des classes au CPU ;
  update : insertion de la fenêtre dans le cache et éviction des minutes devenues inutiles.
Aucune table précalculée : l'historique ne provient que des fenêtres déjà passées dans le détecteur.
"""
import time

import numpy as np
import torch

from src.models.lstm.train_temporal import TemporalHead, load_encoder
from src.realtime.history_cache import HostHistory


def _sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize()


class OnlineDetector:
    def __init__(self, head_ckpt, edge_dim, device):
        self.dev = torch.device(device if (torch.cuda.is_available() or device == "cpu") else "cpu")
        ckh = torch.load(head_ckpt, map_location=self.dev, weights_only=False)
        a = ckh["args"]
        self.names = ckh["names"]
        self.k = a["k"]
        self.enc = load_encoder(a["checkpoint"], edge_dim, len(self.names), self.dev)
        self.hid = self.enc.node_init[0].out_features
        self.head = TemporalHead(self.hid, len(self.names), a["lstm_hidden"], a["dropout"], not a["no_temporal"]).to(self.dev)
        self.head.load_state_dict(ckh["state"])
        self.head.eval()
        self.hist = HostHistory(self.k)
        self.n_par = sum(p.numel() for p in self.enc.parameters()) + sum(p.numel() for p in self.head.parameters())

    def reset(self):
        self.hist = HostHistory(self.k)

    @torch.no_grad()
    def process(self, file, minute, x_np, src_np, dst_np, want_logits=False, update_cache=True):
        """x_np : [E, d] float32 (variables normalisées) ; src_np, dst_np : identifiants globaux d'hôtes [E].
        Retourne (classes prédites [E], dict des temps en ms[, logits])."""
        dev, k, hid = self.dev, self.k, self.hid
        _sync(dev); t0 = time.perf_counter()
        E = len(src_np)
        gids, inv = np.unique(np.concatenate([src_np, dst_np]), return_inverse=True)
        inv = inv.reshape(-1)
        n = len(gids)
        x = torch.from_numpy(np.ascontiguousarray(x_np)).to(dev)
        s = torch.from_numpy(np.ascontiguousarray(inv[:E])).to(dev)
        d = torch.from_numpy(np.ascontiguousarray(inv[E:])).to(dev)
        _sync(dev); t1 = time.perf_counter()
        ep, z = self.enc.encode(x, s, d, n)
        _sync(dev); t2 = time.perf_counter()
        seq = torch.zeros((n, k, hid + 1), device=dev)
        if self.head.temporal:
            for j, e in enumerate(self.hist.lookup(file, minute, gids), start=1):
                if e is None:
                    continue
                Z, pc, ok = e
                idx = torch.from_numpy(np.where(ok, pc, -1)).to(dev)
                m = idx >= 0
                seq[:, k - 1 - j, :hid] = Z[idx.clamp(min=0)] * m.unsqueeze(-1)
                seq[:, k - 1 - j, hid] = m.float()
            seq[:, k - 1, :hid] = z
            seq[:, k - 1, hid] = 1.0
        _sync(dev); t3 = time.perf_counter()
        h = self.head.node_h(seq) if self.head.temporal else None
        logits = self.head(ep, z, h, s, d)
        pred = logits.argmax(1).cpu().numpy()
        _sync(dev); t4 = time.perf_counter()
        if update_cache:
            self.hist.update(file, minute, gids, z)
        t5 = time.perf_counter()
        tm = {"prep": (t1 - t0) * 1e3, "enc": (t2 - t1) * 1e3, "hist": (t3 - t2) * 1e3,
              "head": (t4 - t3) * 1e3, "update": (t5 - t4) * 1e3}
        tm["net"] = tm["enc"] + tm["hist"] + tm["head"]
        tm["total"] = tm["prep"] + tm["net"] + tm["update"]
        if want_logits:
            return pred, tm, logits.float().cpu().numpy()
        return pred, tm
