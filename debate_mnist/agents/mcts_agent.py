
import torch, math, random

class Node:
    def __init__(self, mask, parent=None):
        self.mask = mask.clone()
        self.parent = parent
        self.children = {}
        self.N = 0
        self.W = 0.0
    def Q(self):
        return 0 if self.N==0 else self.W/self.N

class MCTSAgent:
    def __init__(self, model, goal_label, sims=3000, k_remaining=3, cpuct=1.0, device='cpu'):
        self.model = model
        self.goal = goal_label
        self.sims = sims
        self.k = k_remaining
        self.cpuct = cpuct
        self.device = device

    def select(self, orig_img, mask):
        with torch.no_grad():
            img = orig_img.unsqueeze(0).to(self.device)
        root = Node(mask)
        for _ in range(self.sims):
            node = root
            # selection
            while len(node.children)==len(self._legal_actions(node.mask, orig_img)):
                node = self._ucb_select(node)
            # expansion
            actions = [a for a in self._legal_actions(node.mask, orig_img) if a not in node.children]
            if actions:
                a = random.choice(actions)
                child_mask = node.mask.clone()
                child_mask[a]=True
                node.children[a]=Node(child_mask,parent=node)
                node = node.children[a]
            # rollout / evaluation
            value = self._rollout(orig_img, node.mask)
            # backprop
            cur=node
            while cur:
                cur.N +=1
                cur.W += value
                cur = cur.parent
        # choose best
        best_a = max(root.children.items(), key=lambda kv: kv[1].N)[0]
        return best_a

    def _rollout(self, img, mask):
        with torch.no_grad():
            masked = (img*mask.float()).unsqueeze(0).unsqueeze(0).to(self.device)
            logits = self.model(masked)
            pred = logits.argmax(dim=-1).item()
            return 1.0 if pred==self.goal else 0.0

    def _legal_actions(self, mask, img):
        nz = (img>0).nonzero(as_tuple=False)
        legal = [tuple(p.tolist()) for p in nz if not mask[tuple(p.tolist())]]
        return legal

    def _ucb_select(self,node):
        total_N = sum(c.N for c in node.children.values())
        best_score=-1e9
        best_child=None
        for a,child in node.children.items():
            u = child.Q()+self.cpuct*math.sqrt(total_N)/(1+child.N)
            if u>best_score:
                best_score=u
                best_child=child
        return best_child
