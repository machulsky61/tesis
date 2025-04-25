import torch, argparse, tqdm
from torchvision import datasets, transforms
from debate_mnist.experiments.train_judge import SparseCNN, random_sparse

parser = argparse.ArgumentParser()
parser.add_argument('--k', type=int, default=6)
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model  = SparseCNN().to(device)
model.load_state_dict(torch.load(f'models/judge_{args.k}px.pth', map_location=device))
model.eval()

tf       = transforms.Compose([transforms.Resize(16), transforms.ToTensor()])
test_ds  = datasets.MNIST(root='./data', train=False, download=True, transform=tf)

correct = 0
for img, label in tqdm.tqdm(test_ds):
    img   = random_sparse(img.unsqueeze(0).to(device), args.k)
    logits= model(img)
    pred  = logits.argmax(dim=1).item()
    if pred == label: correct += 1
print(f'Accuracy base (k={args.k}): {correct/len(test_ds):.4f}')
