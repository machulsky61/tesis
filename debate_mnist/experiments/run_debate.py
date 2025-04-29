
import torch, argparse, tqdm
from torchvision import datasets, transforms
from debate_mnist.agents.mcts_agent   import MCTSAgent
from debate_mnist.agents.greedy_agent import GreedyAgent
from debate_mnist.experiments.train_judge import SparseCNN
import numpy as np, random

def play(img, labelA, labelB, model, sims, k, agent_type, device):
    orig = img.to(device).unsqueeze(0)         # (1,1,H,W)
    turns_each = k // 2

    if agent_type == 'greedy':
        agentA = GreedyAgent(model, labelA, turns_each, device=device)
        agentB = GreedyAgent(model, labelB, turns_each, device=device)
    else:
        agentA = MCTSAgent(model, labelA, sims=sims, k=turns_each, device=device)
        agentB = MCTSAgent(model, labelB, sims=sims, k=turns_each, device=device)

    agentA.reset(orig)
    agentB.reset(orig)
    mask = torch.zeros_like(orig, dtype=torch.bool)

    for t in range(k):
        rc = agentA.play_turn() if t % 2 == 0 else agentB.play_turn()
        if rc is None: continue
        r, c = rc
        mask[0,0,r,c] = True

    with torch.no_grad():
        pred = model((orig * mask.float())).argmax(dim=-1).item()
    return pred == labelA

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--k', type=int, default=6)
    parser.add_argument('--sims', type=int, default=3000)
    parser.add_argument('--agent', choices=['mcts', 'greedy'], default='mcts')
    parser.add_argument('--n_samples', type=int, default=10000,  # permite subset rápido
                        help='Número de imágenes del test set (por defecto 10000)')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Cargar juez
    model = SparseCNN().to(device)
    model.load_state_dict(torch.load(f'debate_mnist/models/judge_{args.k}px.pth', map_location=device))
    model.eval()

    # Dataset
    tf = transforms.Compose([transforms.Resize(16), transforms.ToTensor()])
    test_ds = datasets.MNIST(root='./data', train=False, download=True, transform=tf)

    # Sub-sample si se pidió
    if args.n_samples < len(test_ds):
        indices = random.sample(range(len(test_ds)), args.n_samples)
        test_ds = torch.utils.data.Subset(test_ds, indices)

    # Loop
    correct = 0
    for img, label in tqdm.tqdm(test_ds):
        wrong = random.choice([i for i in range(10) if i != label])
        if play(img, label, wrong, model, args.sims, args.k, args.agent, device):
            correct += 1

    print(f'Accuracy after debate ({args.agent}, k={args.k}): {correct/len(test_ds):.4f}')
if __name__ == "__main__":
    main()

