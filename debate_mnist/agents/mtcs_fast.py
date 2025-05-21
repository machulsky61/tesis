import torch, random
from agents.base_agent import DebateAgent

class FastMCTSAgent(DebateAgent):
    def __init__(self, judge_model, my_class, opp_class,
                 original_image, thr=0.1, simulations=512,
                 total_moves=0, is_truth_agent=True):
        super().__init__(judge_model, my_class, opp_class, original_image, thr)
        self.sims = simulations
        self.total_moves = total_moves
        self.is_truth = is_truth_agent
        self.device = next(judge_model.parameters()).device
        
    def _forward_batched(self, inp, batch=8192):
        outs = []
        for i in range(0, inp.size(0), batch):
            outs.append(self.judge(inp[i:i+batch]))
        return torch.cat(outs, 0)

    @torch.no_grad()
    def choose_pixel(self, mask, reveal_count=0):
        # 1) genera lista de candidatos relevantes no revelados
        remaining = torch.tensor(
            [(y,x) for (y,x) in self.relevant_coords if mask[y,x]==0],
            device=self.device, dtype=torch.long
        )
        if remaining.numel()==0:
            return None
        n_cand = remaining.size(0)

        # 2) construye de golpe sims máscaras iniciales para TODOS los candidatos
        # base_masks shape: [n_cand, H, W]
        base_masks = mask.expand(n_cand, *mask.shape).clone()
        base_masks[torch.arange(n_cand), remaining[:,0], remaining[:,1]] = 1.

        moves_left = self.total_moves - (reveal_count + 1)
        if moves_left < 0:
            moves_left = 0

        # 3) pre-genera tensor con secuencias aleatorias de píxeles para los roll-outs
        #    idxs shape [n_cand, sims, moves_left, 2]
        if moves_left:
            rand_idx = torch.randint(0, len(self.relevant_coords),
                                     (n_cand, self.sims, moves_left),
                                     device=self.device)
            coords = torch.as_tensor(self.relevant_coords, device=self.device)
            rand_coords = coords[rand_idx]                       # [..., 2]
        else:
            rand_coords = None

        # 4) construye máscaras completas de todos los roll-outs en batch:
        #    rollout_masks shape [n_cand*sims, H, W]
        rollout_masks = base_masks.unsqueeze(1).repeat(1, self.sims, 1, 1)
        rollout_masks = rollout_masks.view(-1, *mask.shape)      # flatten

        if moves_left:
            # vectoriza: ponemos 1 en cada paso aleatorio
            # expand dims: [n_cand, sims, moves_left, H, W] → flatten
            flat_idx = torch.arange(n_cand*self.sims, device=self.device) \
                         .view(n_cand, self.sims, 1)
            y = rand_coords[...,0];  x = rand_coords[...,1]
            rollout_masks[flat_idx, y, x] = 1.

        # 5) monta batch de entrada al juez
        values = self.image * rollout_masks                      # [N, H, W]
        inp = torch.stack([rollout_masks, values], dim=1)        # [N, 2, H, W]

        # 6) pasa todo el batch por el juez de una sola vez
        # out = self.judge(inp)                                    # [N, 10]
        out = self._forward_batched(inp, batch=8192)
        out = out.view(n_cand, self.sims, -1)                    # [cand, sims, 10]

        log_my  = out[:,:,self.my_class]
        log_opp = out[:,:,self.opp_class]

        wins = (log_my >= log_opp).float().mean(dim=1)           # win-rate por cand
        best = torch.argmax(wins).item()
        return tuple(int(v) for v in remaining[best].tolist())
