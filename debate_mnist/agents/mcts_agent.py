# debate_mnist/agents/mcts_agent.py
"""
MCTSAgent (versión optimizada)
------------------------------
•   Cada rollout se evalúa en lote sobre GPU.
•   Sólo se expanden los Top-N píxeles más intensos (> thr) sin revelar.
•   El agente busca maximizar el logit de 'target_label' del juez.
"""

from __future__ import annotations
import torch, math, random
from typing import Tuple, List


# ============ Nodo del árbol ============ #
class TreeNode:
    __slots__ = ("mask", "shown", "parent",
                 "prior", "children",
                 "visits", "value_sum")

    def __init__(self,
                 mask: torch.Tensor,        # bool (1,1,H,W)
                 shown: int,               # píxeles revelados por ESTE agente
                 prior: float = 0.0,
                 parent: "TreeNode | None" = None):
        self.mask       = mask
        self.shown      = shown
        self.parent     = parent
        self.prior      = prior

        self.children: dict[Tuple[int, int], TreeNode] = {}
        self.visits     = 0
        self.value_sum  = 0.0

    @property
    def mean_value(self) -> float:
        return 0.0 if self.visits == 0 else self.value_sum / self.visits


# ============ Agente ============ #
class MCTSAgent:
    def __init__(self,
                 judge,                  # CNN del juez en el device correcto
                 target_label: int,      # clase 0-9 que queremos maximizar
                 sims: int      = 8_000, # rollouts por turno
                 k: int         = 3,     # píxeles totales que juega este agente
                 thr: float      = 0.02, # umbral de intensidad
                 topN: int       = 50,   # Nº máximo de hijos por nodo
                 device: str     = "cpu",
                 c_puct: float   = 2.0):
        self.judge   = judge
        self.label   = target_label
        self.sims    = sims
        self.k       = k
        self.thr     = thr
        self.topN    = topN
        self.device  = device
        self.c_puct  = c_puct
        #
        self.img  : torch.Tensor | None = None
        self.root : TreeNode            = None   # se crea en reset()

    # ---------- API ---------- #
    def reset(self, img: torch.Tensor):
        """
        img : tensor (1,1,H,W) en [0,1]  (ya float32)
        """
        self.img  = img.to(self.device)
        mask0     = torch.zeros_like(self.img, dtype=torch.bool)
        self.root = TreeNode(mask0, shown=0)

    def play_turn(self) -> Tuple[int, int] | None:
        if self.root.shown >= self.k:
            return None

        # ---------- Simulaciones ----------
        for _ in range(self.sims):
            self._simulate(self.root)

        # ---------- Mejor hijo ----------
        if not self.root.children:
            return None
        move, best_child = max(self.root.children.items(),
                               key=lambda kv: kv[1].visits)

        self.root = best_child      # avanzar raíz
        return move

    # ---------- Internas ---------- #
    # ··· MCTS pipeline ··· #
    def _simulate(self, node: TreeNode):
        path: List[TreeNode] = []
        cur = node

        # Selección
        while cur.children:
            _, cur = self._select_child(cur)
            path.append(cur)

        # Expansión (si no completó cuota)
        if cur.shown < self.k:
            self._expand(cur)

        # Roll-out
        value = self._rollout(cur)

        # Back-prop
        for n in [node] + path:
            n.visits    += 1
            n.value_sum += value

    # ··· helpers ··· #
    def _score(self, logits: torch.Tensor) -> torch.Tensor:
        """Devuelve logit de la clase objetivo (tensor shape (N,1))."""
        return logits[:, self.label:self.label+1]   # mantener dim (N,1)

    def _ucb(self, parent: TreeNode, child: TreeNode) -> float:
        prior = child.prior
        q     = child.mean_value
        u     = self.c_puct * prior * math.sqrt(parent.visits + 1) / (child.visits + 1)
        return q + u

    def _select_child(self, node: TreeNode):
        return max(node.children.items(),
                   key=lambda kv: self._ucb(node, kv[1]))

    # --------- Expansión con Top-N --------- #
    def _expand(self, node: TreeNode):
        # máscara de candidatos no revelados e intensos
        full_mask = (~node.mask) & (self.img.abs() > self.thr)  # 1×1×H×W
        cand2d    = full_mask[0, 0]                             # H×W

        coords = cand2d.nonzero(as_tuple=False)                 # N×2
        if coords.numel() == 0:
            return

        vals  = self.img[0, 0][cand2d]                          # (N,)
        top   = torch.argsort(vals, descending=True)[: self.topN]
        coords = coords[top]

        prior = 1.0 / len(coords)
        for r, c in coords:
            r, c = int(r), int(c)
            new_mask = node.mask.clone()
            new_mask[0, 0, r, c] = True
            node.children[(r, c)] = TreeNode(
                new_mask,
                shown=node.shown + 1,
                prior=prior,
                parent=node,
            )


    # --------- Roll-out rápido en lote --------- #
    def _rollout(self, node: TreeNode) -> float:
        mask  = node.mask.clone()
        shown = node.shown

        cand_mask = (~mask) & (self.img.abs() > self.thr)
        coords = cand_mask[0,0].nonzero(as_tuple=False)      # tensor N×2

        # barajamos con torch, sin convertir a lista
        perm   = torch.randperm(coords.size(0), device=coords.device)
        coords = coords[perm]

        for r, c in coords:
            if shown >= self.k:
                break
            mask[0,0,r,c] = True
            shown += 1

        with torch.no_grad():
            logits = self.judge((self.img * mask.float()))
            value  = self._score(logits).item()

        return value


        # ---------- Batch de 1 (rápido) ----------
        with torch.no_grad():
            logits = self.judge((self.img * mask.float()))
            value  = self._score(logits).item()  # escalar float

        return value
