import torch
import random
from agents.base_agent import DebateAgent

class MCTSAgent(DebateAgent):
    """
    Agente que utiliza búsqueda Monte Carlo (MCTS) simple para decidir el próximo movimiento.
    Simula aleatoriamente múltiples partidas para cada posible movimiento y elige el pixel que conduce a la mayor probabilidad de ganar.
    """
    def __init__(self, judge_model, my_class, opponent_class, original_image, thr=0.1, simulations=100, total_moves=0, is_truth_agent=True):
        """
        simulations: número de simulaciones aleatorias a realizar por movimiento candidato (número de rollouts).
        total_moves: número total de revelaciones permitidas en el debate (k).
        is_truth_agent: booleano que indica si este agente representa la clase verdadera (True) o la clase engañosa (False).
        """
        super(MCTSAgent, self).__init__(judge_model, my_class, opponent_class, original_image, thr)
        self.simulations = simulations
        self.total_moves = total_moves
        self.is_truth_agent = is_truth_agent
    def choose_pixel(self, mask, reveal_count=0):
        # Lista de píxeles no revelados relevantes
        candidates = [(y, x) for (y, x) in self.relevant_coords if mask[y, x].item() == 0]
        if not candidates:
            return None
        best_score = -1.0
        best_pixel = None
        # Obtener cuántos movimientos quedan en total (incluyendo ambos agentes) después de esta jugada
        moves_done = reveal_count  # revelaciones ya hechas antes de esta jugada
        # Por cada pixel candidato, simular 'self.simulations' partidas aleatorias
        for (y, x) in candidates:
            wins = 0
            # Realizar movimiento hipotético: revelar pixel (y,x)
            mask_sim = mask.clone().detach()
            mask_sim[y, x] = 1.0
            # Contar esta revelación como realizada
            moves_done_sim = moves_done + 1
            # Calcular movimientos restantes después de este
            moves_left = self.total_moves - moves_done_sim
            # Simular múltiples rollouts aleatorios
            for _ in range(self.simulations):
                # Clonar estado actual para la simulación
                mask_rollout = mask_sim.clone().detach()
                # Turno siguiente: tras este agente, jugará el oponente si moves_left > 0
                next_agent_truth = not self.is_truth_agent  # si yo soy verdad, próximo es mentira; si soy mentira, próximo es verdad
                moves_left_sim = moves_left
                # Realizar movimientos aleatorios alternando agentes
                while moves_left_sim > 0:
                    # Elegir aleatoriamente uno de los píxeles no revelados restantes
                    remaining = [(yy, xx) for (yy, xx) in self.relevant_coords if mask_rollout[yy, xx].item() == 0]
                    if not remaining:
                        break
                    choice = random.choice(remaining)
                    mask_rollout[choice[0], choice[1]] = 1.0
                    moves_left_sim -= 1
                    # Cambiar turno (alternar agente)
                    next_agent_truth = not next_agent_truth
                # Al terminar simulación, evaluar ganador con el modelo juez
                # Preparar input del juez para estado final mask_rollout
                values_final = self.image * mask_rollout
                input_final = torch.stack([mask_rollout, values_final], dim=0).unsqueeze(0)
                with torch.no_grad():
                    out = self.judge(input_final)
                # Obtener logits de las dos clases en debate
                logit_my = out[0, self.my_class].item()
                logit_opp = out[0, self.opp_class].item()
                # Decidir ganador: el juez elegirá la clase con mayor logit
                if logit_my >= logit_opp:
                    # El juez favorece la clase defendida por este agente (mi_class)
                    wins += 1
            win_rate = wins / float(self.simulations)
            if win_rate > best_score:
                best_score = win_rate
                best_pixel = (y, x)
        return best_pixel
