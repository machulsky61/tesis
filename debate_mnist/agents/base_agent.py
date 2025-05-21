import torch

class DebateAgent:
    """
    Clase base para agentes del debate. Almacena información común como modelo juez, clases objetivo y la imagen original.
    """
    def __init__(self, judge_model, my_class, opponent_class, original_image, thr=0.1):
        """
        judge_model: modelo del juez (clasificador) para evaluar estados.
        my_class: clase (dígito) que este agente defiende que es la imagen.
        opponent_class: clase que defiende el agente oponente.
        original_image: imagen original completa (tensor 1xHxW o 2D HxW) con intensidades.
        thr: umbral de relevancia de píxeles para considerar.
        """
        self.judge = judge_model
        self.my_class = my_class
        self.opp_class = opponent_class
        # Asegurarse de que la imagen original es 2D HxW para facilitar cálculos
        self.image = original_image.squeeze()
        # Mover la imagen al mismo dispositivo que el modelo del juez
        self.image = self.image.to(next(judge_model.parameters()).device)
        # Guardar umbral y precomputar píxeles relevantes (coordenadas) según thr
        self.thr = thr
        # Obtener coords relevantes: tuplas (y,x) donde la intensidad > thr
        mask_relevant = (self.image > self.thr)
        coords = mask_relevant.nonzero(as_tuple=False)
        if coords.numel() == 0:
            # si ningún pixel supera thr, considerar todos los píxeles
            H, W = self.image.shape[-2], self.image.shape[-1]
            coords = torch.cartesian_prod(torch.arange(H), torch.arange(W))
        # Convertir coords tensor a lista de tuplas de int (en CPU para facilidad)
        coords = coords.cpu().tolist()
        self.relevant_coords = [(int(y), int(x)) for (y, x) in coords]
    def choose_pixel(self, mask, reveal_count=None):
        """Método abstracto que el agente debe implementar para elegir el próximo píxel a revelar."""
        raise NotImplementedError("Este método debe ser implementado por subclases de DebateAgent")
