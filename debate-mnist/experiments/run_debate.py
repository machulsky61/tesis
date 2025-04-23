
import torch, argparse, tqdm
from torchvision import datasets, transforms
from agents.mcts_agent import MCTSAgent
from experiments.train_judge import SparseCNN
import numpy as np, random

def play(img, labelA,labelB, model, sims, k):
    device='cuda' if torch.cuda.is_available() else 'cpu'
    orig = img.squeeze(0)
    mask = torch.zeros_like(orig, dtype=torch.bool)
    agentA = MCTSAgent(model,labelA,sims,k//2,device=device)
    agentB = MCTSAgent(model,labelB,sims,k//2,device=device)
    turn=0
    while mask.sum()<k:
        if turn%2==0:
            a=agentA.select(orig,mask)
        else:
            a=agentB.select(orig,mask)
        mask[a]=True
        turn+=1
    with torch.no_grad():
        logits=model((orig*mask.float()).unsqueeze(0).to(device))
    pred=logits.argmax(dim=-1).item()
    return pred==(labelA if pred==labelA else labelB)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--k',type=int,default=6)
    parser.add_argument('--sims',type=int,default=3000)
    args=parser.parse_args()
    device='cuda' if torch.cuda.is_available() else 'cpu'
    model=SparseCNN().to(device)
    model.load_state_dict(torch.load(f'models/judge_{args.k}px.pth',map_location=device))
    model.eval()
    tf=transforms.Compose([transforms.Resize(16), transforms.ToTensor()])
    test_ds=datasets.MNIST(root='./data',train=False,download=True, transform=tf)
    correct=0
    for img,label in tqdm.tqdm(test_ds):
        labelA=label
        # choose wrong label for B
        labelB=random.choice([i for i in range(10) if i!=label])
        if play(img,labelA,labelB,model,args.sims,args.k):
            correct+=1
    print('Accuracy after debate:', correct/len(test_ds))
if __name__=='__main__':
    main()
