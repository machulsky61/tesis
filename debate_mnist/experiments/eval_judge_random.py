import torch, argparse, tqdm
from torchvision import datasets, transforms
from debate_mnist.experiments.train_judge import SparseCNN, random_sparse
import numpy as np
import os
from PIL import Image
import json
import pandas as pd
from datetime import datetime
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def save_confusion_matrix(y_true, y_pred, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix - Random Mask Judge Predictions')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(save_path)
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--k', type=int, default=6)
    parser.add_argument('--n_samples', type=int, default=10000,
                        help='Número de imágenes del test set (por defecto 10000)')
    parser.add_argument('--save_dir', type=str, default='debate_mnist/data/random_masked_images',
                        help='Directory to save the masked images')
    args = parser.parse_args()

    # Create timestamped run directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.save_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, 'metadata'), exist_ok=True)
    os.makedirs(os.path.join(run_dir, 'original_images'), exist_ok=True)
    os.makedirs(os.path.join(run_dir, 'masked_images'), exist_ok=True)

    # Save run parameters
    run_info = {
        'timestamp': timestamp,
        'k': args.k,
        'n_samples': args.n_samples,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'masking_type': 'random'
    }

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SparseCNN().to(device)
    model.load_state_dict(torch.load(f'debate_mnist/models/judge_{args.k}px.pth', map_location=device))
    model.eval()

    # Dataset
    tf = transforms.Compose([transforms.Resize(16), transforms.ToTensor()])
    test_ds = datasets.MNIST(root='./debate_mnist/data', train=False, download=True, transform=tf)

    # Sub-sample si se pidió
    if args.n_samples < len(test_ds):
        indices = np.random.choice(len(test_ds), args.n_samples, replace=False)
        test_ds = torch.utils.data.Subset(test_ds, indices)

    # Loop
    correct = 0
    results = []
    all_true_labels = []
    all_predictions = []

    for idx, (img, label) in enumerate(tqdm.tqdm(test_ds)):
        # Save original image
        original_img = img.unsqueeze(0).to(device)
        original_np = original_img.squeeze().cpu().numpy()
        original_np = (original_np * 255).astype(np.uint8)
        original_pil = Image.fromarray(original_np)
        original_filename = f'original_{idx:04d}_label_{label}.png'
        original_pil.save(os.path.join(run_dir, 'original_images', original_filename))

        # Create random masked image
        masked_img = random_sparse(original_img, args.k)
        
        # Get prediction
        with torch.no_grad():
            logits = model(masked_img)
            pred = logits.argmax(dim=1).item()
        
        is_correct = pred == label
        if is_correct:
            correct += 1

        # Save masked image
        masked_np = masked_img.squeeze().cpu().numpy()
        masked_np = (masked_np * 255).astype(np.uint8)
        masked_pil = Image.fromarray(masked_np)
        masked_filename = f'masked_{idx:04d}_true_{label}_pred_{pred}.png'
        masked_pil.save(os.path.join(run_dir, 'masked_images', masked_filename))

        # Save metadata
        metadata = {
            'image_idx': idx,
            'true_label': int(label),
            'prediction': int(pred),
            'is_correct': bool(is_correct),
            'masking_type': 'random',
            'k_pixels': args.k
        }
        metadata_filename = f'masked_{idx:04d}_true_{label}.json'
        with open(os.path.join(run_dir, 'metadata', metadata_filename), 'w') as f:
            json.dump(metadata, f, indent=2)

        # Record result
        results.append({
            'image_idx': idx,
            'true_label': label,
            'prediction': pred,
            'is_correct': is_correct
        })

        # Record for confusion matrix
        all_true_labels.append(label)
        all_predictions.append(pred)

    # Calculate final accuracy
    accuracy = correct/len(test_ds)
    run_info['accuracy'] = accuracy

    # Save run results to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(run_dir, 'results.csv'), index=False)

    # Save run info
    with open(os.path.join(run_dir, 'run_info.json'), 'w') as f:
        json.dump(run_info, f, indent=2)

    # Generate and save confusion matrix
    save_confusion_matrix(all_true_labels, all_predictions,
                         os.path.join(run_dir, 'confusion_matrix.png'))

    # Update or create experiments log
    experiments_log_path = os.path.join(args.save_dir, 'experiments_log.csv')
    if os.path.exists(experiments_log_path):
        experiments_df = pd.read_csv(experiments_log_path)
    else:
        experiments_df = pd.DataFrame(columns=['timestamp', 'k', 'n_samples', 'accuracy', 'masking_type'])

    # Add new experiment
    new_experiment = pd.DataFrame([{
        'timestamp': timestamp,
        'k': args.k,
        'n_samples': args.n_samples,
        'accuracy': accuracy,
        'masking_type': 'random'
    }])
    experiments_df = pd.concat([experiments_df, new_experiment], ignore_index=True)
    experiments_df.to_csv(experiments_log_path, index=False)

    print(f'Accuracy with random masking (k={args.k}): {accuracy:.4f}')
    print(f'Saved run results to {run_dir}/')
    print(f'Updated experiments log at {experiments_log_path}')

if __name__ == "__main__":
    main() 