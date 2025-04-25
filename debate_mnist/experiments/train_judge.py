
import torch, torchvision, torch.nn as nn, torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
from pathlib import Path
import tqdm, argparse

class SparseCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1,16,3,padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16,32,3,padding=1), nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32*4*4,64), nn.ReLU(),
            nn.Linear(64,10)
        )
    def forward(self,x):
        return self.classifier(self.features(x))

def random_sparse(x, k=6, thr=0.5):
    """
    Deja sólo k píxeles con intensidad > thr.
    El resto se pone a 0.
    """
    x = x.clone()
    B = x.size(0)
    for i in range(B):
        nz = (x[i,0] > thr).nonzero(as_tuple=False)
        if len(nz) == 0:
            continue
        idx = nz[torch.randperm(len(nz))[:k]]
        mask = torch.zeros_like(x[i,0], dtype=torch.bool)
        mask[idx[:,0], idx[:,1]] = True
        x[i,0][~mask] = 0.0
    return x



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--k', type=int, default=6)
    parser.add_argument('--epochs', type=int, default=50)
    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tf = transforms.Compose([transforms.Resize(16), transforms.ToTensor()])
    train_ds = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=tf)
    loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=2)
    model = SparseCNN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    for epoch in range(args.epochs):
        for x,y in tqdm.tqdm(loader, leave=False):
            x,y = x.to(device), y.to(device)
            x = random_sparse(x, args.k)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
    Path('models').mkdir(exist_ok=True)
    torch.save(model.state_dict(), f'models/judge_{args.k}px.pth')
    print('Saved model.')
if __name__ == '__main__':
    main()
