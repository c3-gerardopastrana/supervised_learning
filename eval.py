import torch
import torch.nn.functional as F
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
import torch.distributed as dist
from contextlib import nullcontext
from tqdm import tqdm

def gather_tensor(tensor):
    world_size = dist.get_world_size()
    tensors_gather = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(tensors_gather, tensor)
    return torch.cat(tensors_gather, dim=0)



def run_lda_on_embeddings(train_loader, val_loader, model, device=None, use_amp=True):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    def extract_embeddings(loader):
        embeddings, labels = [], []
        rank = dist.get_rank() if dist.is_initialized() else 0
        progress = tqdm(loader) if rank == 0 else nullcontext(loader)
    
        with torch.no_grad():
            for x, y in progress:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=use_amp):
                    feats = model._forward_impl(x)
                    feats = F.normalize(feats, p=2, dim=1)
                embeddings.append(feats)
                labels.append(y)
        
        embeddings = torch.cat(embeddings)
        labels = torch.cat(labels)
    
        if dist.is_initialized():
            embeddings = gather_tensor(embeddings)
            labels = gather_tensor(labels)
    
    
            return embeddings.cpu().numpy(), labels.cpu().numpy()

    X_train, y_train = extract_embeddings(train_loader)
    X_val, y_val = extract_embeddings(val_loader)
    print("rank done",dist.get_rank())
    if dist.get_rank() == 0:
        print("LDA on ",X_train.shape)
        lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
        lda.fit(X_train, y_train)
        y_pred = lda.predict(X_val)
        acc = accuracy_score(y_val, y_pred)
        print(f"LDA Test Accuracy: {acc * 100:.2f}%")
        return acc
    else:
        return None  # only rank 0 computes LDA

