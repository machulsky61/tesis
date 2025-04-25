import torch
from debate_mnist.utils import even_logit, odd_logit

class GreedyAgent:
    """
    Elige en cada turno el píxel (>0.5) que más cambia
    (o menos, si invertimos signo) la diferencia entre
    logit_par e impar.
    """
    def __init__(self, judge, k, maximize=True, thr=0.5, device='cpu'):
        self.judge = judge
        self.k     = k
        self.max   = maximize        # True → par   False → impar
        self.thr   = thr
        self.device= device

    def reset(self, img):
        """
        img: tensor (1,1,H,W) en [0,1]
        """
        self.img      = img.clone().to(self.device)
        self.current  = torch.zeros_like(self.img)   # máscara mostrada
        self.shown_px = 0

    def _score(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Devuelve una única escalar (tensor shape (1,))
        que mide cuán 'par' es la imagen.  Si self.max==True
        el agente intentará maximizarla; si es False la minimiza
        (equivale a favorecer 'impar').
        """
        score = even_logit(logits) - odd_logit(logits)   # (1,1)
        return score if self.max else -score

    def select_pixel(self):
        """
        Devuelve (r,c) del mejor píxel aún no mostrado.
        """
        if self.shown_px >= self.k:
            return None

        # píxeles candidatos: > thr y aún tapados
        # cand = (self.img[0,0] > self.thr) & (self.current[0,0] == 0)
        cand = (self.current[0,0] == 0)
        coords = cand.nonzero(as_tuple=False)
        if len(coords) == 0:
            return None  # nada útil

        # baseline
        base_logits = self.judge(self.current).detach()
        base_score  = self._score(base_logits)

        best_gain, best_rc = -1e9, None
        for r,c in coords:
            tmp = self.current.clone()
            tmp[0,0,r,c] = self.img[0,0,r,c]          # revela
            logits = self.judge(tmp).detach()
            gain   = self._score(logits) - base_score
            if gain > best_gain:
                best_gain = gain
                best_rc   = (int(r), int(c))
                
        # print(f"base={base_score.item():.3f} best_gain={best_gain.item():.3f}")
        return best_rc

    def play_turn(self):
        rc = self.select_pixel()
        if rc is None:
            return None
        r,c = rc
        self.current[0,0,r,c] = self.img[0,0,r,c]
        self.shown_px += 1
        return rc
