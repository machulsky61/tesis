import torch
from debate_mnist.utils import even_logit, odd_logit
class GreedyAgent:
    """
    Revela, turno a turno, el píxel que más aumenta el logit de su clase-objetivo.
    """
    def __init__(self, judge, target_label, max_pixels, thr=0.2, device='cpu'):
        """
        judge         : red convolucional que da logits (1,10)
        target_label  : entero 0-9 que este agente quiere maximizar
        max_pixels    : nº de píxeles que está permitido mostrar (k//2)
        thr           : opcional, umbral para filtrar píxeles poco informativos
        """
        self.judge   = judge
        self.label   = target_label
        self.max_px  = max_pixels
        self.thr     = thr
        self.device  = device

    # -------- ciclo por imagen ----------
    def reset(self, img):
        self.img      = img.clone().to(self.device)           # (1,1,H,W)
        self.current  = torch.zeros_like(self.img)            # máscara visible
        self.shown    = 0

    # -------- lógica interna -------------
    def _score(self, logits: torch.Tensor) -> torch.Tensor:
        # Queremos SUBIR el logit de nuestra clase-diana
        return logits[:, self.label]

    def select_pixel(self):
        if self.shown >= self.max_px:
            return None

        # Candidatos = píxeles ocultos (puedes filtrar por umbral si quieres)
        cand_mask = (self.current == 0) & (self.img.abs() > self.thr)
        coords = cand_mask[0,0].nonzero(as_tuple=False)      # (N,2)
        if coords.numel() == 0:
            return None

        base = self._score(self.judge(self.current)).item()
        best_gain, best_rc = -1e9, None

        for r, c, *_ in coords:                      # r,c en CPU
            tmp = self.current.clone()
            tmp[0,0,r,c] = self.img[0,0,r,c]
            gain = self._score(self.judge(tmp)).item() - base
            if gain > best_gain:
                best_gain, best_rc = gain, (int(r), int(c))

        return best_rc

    def play_turn(self):
        rc = self.select_pixel()
        if rc is None:
            return None
        r, c = rc
        self.current[0,0,r,c] = self.img[0,0,r,c]
        self.shown += 1
        return rc
