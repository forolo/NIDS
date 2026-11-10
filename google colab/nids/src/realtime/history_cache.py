"""Cache d'historique des hôtes pour l'inférence en ligne (sans dépendance à PyTorch).

Pour chaque fenêtre déjà traitée (fichier, minute), on garde les identifiants d'hôtes (triés) et les états z
associés. Pour une nouvelle fenêtre, on retrouve, pour chaque hôte, sa ligne dans les k-1 minutes précédentes
du même fichier. Même définition que src/models/lstm/history.py (évaluation par lots), mais calculée
au fil de l'eau : seules les k-1 dernières minutes de chaque fichier sont conservées.
"""
import numpy as np


class HostHistory:
    def __init__(self, k):
        self.k = k
        self.win = {}  # (fichier, minute) -> (gids triés, états)

    def lookup(self, file, minute, gids):
        """Pour j = 1..k-1 : None si la fenêtre (fichier, minute-j) n'est pas en cache, sinon (états, position, présent)."""
        out = []
        for j in range(1, self.k):
            e = self.win.get((file, minute - j))
            if e is None:
                out.append(None)
                continue
            g, Z = e
            pos = np.searchsorted(g, gids)
            pc = np.minimum(pos, len(g) - 1)
            ok = (pos < len(g)) & (g[pc] == gids)
            out.append((Z, pc, ok))
        return out

    def update(self, file, minute, gids, Z):
        self.win[(file, minute)] = (gids, Z)
        # une fenêtre à la minute m' > minute n'a besoin que des minutes >= m' - (k-1) >= minute - k + 2
        for key in [q for q in self.win if q[0] == file and q[1] < minute - self.k + 2]:
            del self.win[key]

    def n_cached(self):
        return len(self.win)
