import torch, argparse, tqdm
from torchvision import datasets, transforms
from debate_mnist.agents.mcts_agent   import MCTSAgent
from debate_mnist.agents.greedy_agent import GreedyAgent
from debate_mnist.experiments.train_judge import SparseCNN
import numpy as np, random
import os
from PIL import Image
import json
import pandas as pd
from datetime import datetime
import shutil

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
    
    # Track debate moves
    debate_moves = []
    
    for t in range(k):
        agent = "A" if t % 2 == 0 else "B"
        rc = agentA.play_turn() if t % 2 == 0 else agentB.play_turn()
        if rc is None: continue
        r, c = rc
        mask[0,0,r,c] = True
        
        # Record the move
        debate_moves.append({
            "turn": t + 1,
            "agent": agent,
            "position": [int(r), int(c)],
            "pixel_value": float(orig[0,0,r,c].item())
        })

    masked_img = orig * mask.float()
    with torch.no_grad():
        pred = model(masked_img).argmax(dim=-1).item()
    
    debate_info = {
        "true_label": int(labelA),
        "agentA_label": int(labelA),
        "agentB_label": int(labelB),
        "final_prediction": int(pred),
        "is_correct": bool(pred == labelA),
        "moves": debate_moves,
        "agent_type": agent_type,
        "sims_per_move": sims,
        "total_pixels": k
    }
    
    return pred == labelA, masked_img, debate_info

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--k', type=int, default=6)
    parser.add_argument('--sims', type=int, default=3000)
    parser.add_argument('--agent', choices=['mcts', 'greedy'], default='mcts')
    parser.add_argument('--n_samples', type=int, default=10000,
                        help='Número de imágenes del test set (por defecto 10000)')
    parser.add_argument('--save_dir', type=str, default='debate_mnist/data/masked_images_after_debate',
                        help='Directory to save the masked images')
    args = parser.parse_args()

    # Create timestamped run directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.save_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, 'metadata'), exist_ok=True)

    # Save run parameters
    run_info = {
        'timestamp': timestamp,
        'k': args.k,
        'sims': args.sims,
        'agent_type': args.agent,
        'n_samples': args.n_samples,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Cargar juez
    model = SparseCNN().to(device)
    model.load_state_dict(torch.load(f'debate_mnist/models/judge_{args.k}px.pth', map_location=device))
    model.eval()

    # Dataset
    tf = transforms.Compose([transforms.Resize(16), transforms.ToTensor()])
    test_ds = datasets.MNIST(root='./debate_mnist/data', train=False, download=True, transform=tf)

    # Sub-sample si se pidió
    if args.n_samples < len(test_ds):
        indices = random.sample(range(len(test_ds)), args.n_samples)
        test_ds = torch.utils.data.Subset(test_ds, indices)

    # Loop
    correct = 0
    results = []
    
    for idx, (img, label) in enumerate(tqdm.tqdm(test_ds)):
        wrong = random.choice([i for i in range(10) if i != label])
        is_correct, masked_img, debate_info = play(img, label, wrong, model, args.sims, args.k, args.agent, device)
        if is_correct:
            correct += 1

        # Save the masked image
        masked_img = masked_img.squeeze().cpu().numpy()
        masked_img = (masked_img * 255).astype(np.uint8)
        img_pil = Image.fromarray(masked_img)
        
        # Save image with descriptive filename
        img_filename = f'debate_{idx:04d}_true_{label}_pred_{wrong}.png'
        img_pil.save(os.path.join(run_dir, img_filename))
        
        # Save metadata
        metadata_filename = f'debate_{idx:04d}_true_{label}_pred_{wrong}.json'
        with open(os.path.join(run_dir, 'metadata', metadata_filename), 'w') as f:
            json.dump(debate_info, f, indent=2)
            
        # Record result
        results.append({
            'image_idx': idx,
            'true_label': label,
            'wrong_label': wrong,
            'prediction': debate_info['final_prediction'],
            'is_correct': is_correct
        })

    # Calculate final accuracy
    accuracy = correct/len(test_ds)
    run_info['accuracy'] = accuracy
    
    # Save run results to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(run_dir, 'results.csv'), index=False)
    
    # Save run info
    with open(os.path.join(run_dir, 'run_info.json'), 'w') as f:
        json.dump(run_info, f, indent=2)
    
    # Update or create experiments log
    experiments_log_path = os.path.join(args.save_dir, 'experiments_log.csv')
    if os.path.exists(experiments_log_path):
        experiments_df = pd.read_csv(experiments_log_path)
    else:
        experiments_df = pd.DataFrame(columns=['timestamp', 'k', 'sims', 'agent_type', 'n_samples', 'accuracy'])
    
    # Add new experiment
    new_experiment = pd.DataFrame([{
        'timestamp': timestamp,
        'k': args.k,
        'sims': args.sims,
        'agent_type': args.agent,
        'n_samples': args.n_samples,
        'accuracy': accuracy
    }])
    experiments_df = pd.concat([experiments_df, new_experiment], ignore_index=True)
    experiments_df.to_csv(experiments_log_path, index=False)

    print(f'Accuracy after debate ({args.agent}, k={args.k}): {accuracy:.4f}')
    print(f'Saved run results to {run_dir}/')
    print(f'Updated experiments log at {experiments_log_path}')

if __name__ == "__main__":
    main()

