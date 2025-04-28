import os
import random
import numpy as np
np.set_printoptions(precision=4, suppress=True)
from sklearn.metrics import accuracy_score
from tqdm.notebook import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import torchvision
from torchvision import transforms, datasets
import torch.optim as optim
import wandb
from functools import partial
from lda import LDA, lda_loss, sina_loss, SphericalLDA
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, random_split
from torch.utils.checkpoint import checkpoint


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )
    
    def _forward_impl(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
        
    def forward(self, x):
        return checkpoint(self._forward_impl, x)

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=1000, lda_args=None, use_checkpoint=False):
        super(ResNet, self).__init__()
        self.lda_args = lda_args
        self.in_planes = 64
        self.use_checkpoint = use_checkpoint
        
        # ImageNet-style initial conv layer
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7,
                               stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Residual layers
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        
        # Global average pooling and output
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(512 * block.expansion, num_classes)
        
        # LDA branch (if enabled)
        if self.lda_args:
            self.lda = LDA(num_classes, lda_args['lamb'])  # your LDA class
    
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    
    def _forward_features(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)
        
        if self.use_checkpoint:
            out = checkpoint(lambda x: self.layer1(x), out)
            out = checkpoint(lambda x: self.layer2(x), out)
            out = checkpoint(lambda x: self.layer3(x), out)
            out = checkpoint(lambda x: self.layer4(x), out)
        else:
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)
            
        out = self.avgpool(out)  # output shape: [B, 512, 1, 1]
        fea = out.view(out.size(0), -1)  # flatten to [B, 512]
        return fea
    
    def forward(self, x, y=None, epoch=0):
        fea = self._forward_features(x)
        
        if self.lda_args:
            fea = F.normalize(fea, p=2, dim=1)
            hasComplexEVal, out, sigma_w_inv_b = self.lda(fea, y)
            return hasComplexEVal, fea, out, sigma_w_inv_b
        else:
            out = self.linear(fea)
            return out


def ResNet18(num_classes=1000, lda_args=None):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes, lda_args)


def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])


def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])


def ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])


def ResNet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])


class CIFAR10:
    def __init__(self, img_names, class_map, transform):
        self.img_names = img_names
        self.classes = [class_map[os.path.basename(os.path.dirname(n))] for n in img_names]
        self.transform = transform
    def __len__(self):
        return len(self.img_names)
    def __getitem__(self, idx):
        img = Image.open(self.img_names[idx])
        img = self.transform(img)
        clazz = self.classes[idx]
        return img, clazz



def get_scaled_lr_sqrt(batch_size: int, base_batch_size: int = 128, base_lr: float = 1e-3) -> float:
    """
    Scales the learning rate with sqrt of batch size increase, where batch size is passed directly.

    Args:
        batch_size (int): new batch size
        base_batch_size (int): original batch size corresponding to base_lr
        base_lr (float): base learning rate at base_batch_size

    Returns:
        float: scaled learning rate
    """
    scale = torch.tensor(batch_size / base_batch_size, dtype=torch.float32)
    return base_lr * scale.item()


class Solver:
    def __init__(self, dataloaders, model_path, n_classes, lda_args={}, local_rank=0, world_size=1, lr=1e-3):
        self.dataloaders = dataloaders
        self.local_rank = local_rank
        self.world_size = world_size
        self.device = torch.device(f'cuda:{local_rank}')
        
        self.net = ResNet18(n_classes, lda_args)
        self.net = self.net.to(self.device)
        
        # Wrap model with DDP
        if world_size > 1:
            self.net = DDP(self.net, device_ids=[local_rank], output_device=local_rank)
        
        self.use_lda = True if lda_args else False
        if self.use_lda:
            self.criterion = partial(lda_loss, n_classes=n_classes, 
                                    n_eig=lda_args['n_eig'], margin=lda_args['margin'])
            self.criterion = sina_loss
        else:
            self.criterion = nn.CrossEntropyLoss()
        
        if local_rank == 0:
            print(self.criterion)

        self.optimizer = optim.AdamW(self.net.parameters(), lr=lr, weight_decay=5e-4)
        self.model_path = model_path
        self.n_classes = n_classes

    def iterate(self, epoch, phase):
        if isinstance(self.net, DDP):
            self.net.module.train(phase == 'train')
        else:
            self.net.train(phase == 'train')
            
        dataloader = self.dataloaders[phase]
        total_loss = 0
        correct = 0
        total = 0
        entropy_sum = 0.0
        entropy_count = 0
        
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            self.optimizer.zero_grad()
        
            if self.use_lda:
                if isinstance(self.net, DDP):
                    hasComplexEVal, feas, outputs, sigma_w_inv_b = self.net.module(inputs, targets, epoch)
                else:
                    hasComplexEVal, feas, outputs, sigma_w_inv_b = self.net(inputs, targets, epoch)
                print(feas.size())
                if not hasComplexEVal:
                    #stats
                    eigvals_norm = outputs / outputs.sum()
                    eps = 1e-10 
                    max_eigval_norm = eigvals_norm.max().item()
                    min_eigval_norm = eigvals_norm.min().item()
                    quantile_25 = torch.quantile(eigvals_norm, 0.25).item()
                    quantile_50 = torch.quantile(eigvals_norm, 0.5).item()
                    quantile_75 = torch.quantile(eigvals_norm, 0.75).item()
                    eigvals_norm = torch.clamp(outputs / outputs.sum(), min=eps, max=1.0)
                    eigvals_norm /= eigvals_norm.sum()
                    entropy = -(eigvals_norm * eigvals_norm.log()).sum().item()
                    entropy_sum += entropy
                    entropy_count += 1
                    trace = torch.trace(sigma_w_inv_b)
                    rank_sigma = torch.linalg.matrix_rank(sigma_w_inv_b).item()
                    condition_sigma = torch.linalg.cond(sigma_w_inv_b).item()     
                    off_diag = sigma_w_inv_b - torch.diag(torch.diagonal(sigma_w_inv_b))
                    sum_squared_off_diag = torch.sum(off_diag ** 2).item()
                    diag_var = torch.var(torch.diagonal(sigma_w_inv_b)).item()
    
                    loss = self.criterion(sigma_w_inv_b)

                    if isinstance(self.net, DDP):
                        outputs = self.net.module.lda.predict_proba(feas)
                    else:
                        outputs = self.net.lda.predict_proba(feas)

                    if phase == 'train' and self.local_rank == 0:
                        wandb.log({
                            'loss': loss,
                            "rank simga": rank_sigma,
                            "condition simga": condition_sigma,
                            "entropy": entropy,
                            "sum_squared_off_diag": sum_squared_off_diag,
                            "diag_var": diag_var,
                            "trace": trace,
                            "max normalized eigenvalue": max_eigval_norm,
                            "min normalized eigenvalue": min_eigval_norm,
                            "quantile_25": quantile_25,
                            "quantile_50": quantile_50,
                            "quantile_75": quantile_75,
                            "epoch": epoch,
                        })
                    
                else:
                    if self.local_rank == 0:
                        print('Complex Eigen values found, skip backpropagation of {}th batch'.format(batch_idx))
                    continue
            else:
                outputs = self.net(inputs, targets, epoch)
                loss = nn.CrossEntropyLoss()(outputs, targets)
        
            if phase == 'train':
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=100.0)
                self.optimizer.step()
                if self.local_rank == 0:
                    wandb.log({"total_grad_norm_encoder": grad_norm.item()})
            total_loss += loss.item()
    
            outputs = torch.argmax(outputs.detach(), dim=1)
            total += targets.size(0)
            correct += outputs.eq(targets).sum().item()
        
        # Sync metrics across GPUs
        if self.world_size > 1:
            metrics = torch.tensor([total_loss, correct, total], dtype=torch.float32, device=self.device)
            dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
            total_loss, correct, total = metrics.tolist()
            
        total_loss /= (batch_idx + 1) * self.world_size
        if total > 0:
            total_acc = correct / total
        else:
            total_acc = 0 
        
        if self.local_rank == 0:
            if entropy_count > 0:
                average_entropy = entropy_sum / entropy_count
                print(f'Average Entropy: {average_entropy:.4f}')
            
            print('\nepoch %d: %s loss: %.3f | acc: %.2f%% (%d/%d)'
                         % (epoch, phase, total_loss, 100.*total_acc, correct, total))
            wandb.log({
                "epoch"+phase: epoch,
                "total"+phase: total_loss,
                "total_acc_train"+phase: 100.*total_acc
            }) 
        return total_loss, total_acc


    def train(self, epochs):
        best_loss = float('inf')
        for epoch in range(epochs):
            # Set epoch for distributed samplers
            
            if self.world_size > 1:
                for phase in self.dataloaders:
                    if hasattr(self.dataloaders[phase].sampler, 'set_epoch'):
                        self.dataloaders[phase].sampler.set_epoch(epoch)
                
            self.iterate(epoch, 'train')
            with torch.no_grad():
                val_loss, val_acc = self.iterate(epoch, 'val')
                
            if val_loss < best_loss and self.local_rank == 0:
                best_loss = val_loss
                if isinstance(self.net, DDP):
                    checkpoint = {'epoch': epoch, 'val_loss': val_loss, 'state_dict': self.net.module.state_dict()}
                else:
                    checkpoint = {'epoch': epoch, 'val_loss': val_loss, 'state_dict': self.net.state_dict()}
                print('best val loss found')
                torch.save(checkpoint, self.model_path)
            
            if self.local_rank == 0:
                print()
        
        # Final save on main process
        if self.local_rank == 0:
            if isinstance(self.net, DDP):
                checkpoint = {'epoch': epochs-1, 'val_loss': val_loss, 'state_dict': self.net.module.state_dict()}
            else:
                checkpoint = {'epoch': epochs-1, 'val_loss': val_loss, 'state_dict': self.net.state_dict()}
            torch.save(checkpoint, self.model_path.replace('.pth', '_final.pth'))

    def test_iterate(self, epoch, phase):
        if isinstance(self.net, DDP):
            self.net.module.eval()
        else:
            self.net.eval()
            
        dataloader = self.dataloaders[phase]
        y_pred = []
        y_true = []
        
        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                if self.use_lda:
                    if isinstance(self.net, DDP):
                        _, feas, outputs = self.net.module(inputs, targets, epoch)
                        outputs = self.net.module.lda.predict_proba(feas)
                    else:
                        _, feas, outputs = self.net(inputs, targets, epoch)
                        outputs = self.net.lda.predict_proba(feas)
                else:
                    outputs = self.net(inputs, targets, epoch)
                    
                outputs = torch.argmax(outputs, dim=1)
                y_pred.append(outputs.detach().cpu().numpy())
                y_true.append(targets.detach().cpu().numpy())
                
        # Gather predictions from all GPUs
        if self.world_size > 1:
            all_y_pred = []
            all_y_true = []
            
            # Convert lists to tensors for gathering
            local_y_pred = torch.from_numpy(np.concatenate(y_pred)).to(self.device)
            local_y_true = torch.from_numpy(np.concatenate(y_true)).to(self.device)
            
            # Get sizes from all processes
            size_tensor = torch.tensor([local_y_pred.size(0)], device=self.device)
            all_sizes = [torch.zeros_like(size_tensor) for _ in range(self.world_size)]
            dist.all_gather(all_sizes, size_tensor)
            
            # Prepare tensors for gathering
            max_size = max(size.item() for size in all_sizes)
            padded_pred = torch.zeros(max_size, dtype=torch.long, device=self.device)
            padded_true = torch.zeros(max_size, dtype=torch.long, device=self.device)
            
            # Copy data to padded tensors
            size = local_y_pred.size(0)
            padded_pred[:size] = local_y_pred
            padded_true[:size] = local_y_true
            
            # Gather padded tensors
            gathered_pred = [torch.zeros_like(padded_pred) for _ in range(self.world_size)]
            gathered_true = [torch.zeros_like(padded_true) for _ in range(self.world_size)]
            
            dist.all_gather(gathered_pred, padded_pred)
            dist.all_gather(gathered_true, padded_true)
            
            # Truncate according to original sizes and convert to numpy
            for i, size in enumerate(all_sizes):
                all_y_pred.append(gathered_pred[i][:size.item()].cpu().numpy())
                all_y_true.append(gathered_true[i][:size.item()].cpu().numpy())
                
            return np.concatenate(all_y_pred), np.concatenate(all_y_true)
        else:
            return np.concatenate(y_pred), np.concatenate(y_true)
        
    def test(self):
        if self.local_rank == 0:
            checkpoint = torch.load(self.model_path)
            epoch = checkpoint['epoch']
            val_loss = checkpoint['val_loss']
            
            if isinstance(self.net, DDP):
                self.net.module.load_state_dict(checkpoint['state_dict'])
            else:
                self.net.load_state_dict(checkpoint['state_dict'])
                
            print('load model at epoch {}, with val loss: {:.3f}'.format(epoch, val_loss))
            
        # Synchronize all processes to ensure the model is loaded
        if self.world_size > 1:
            dist.barrier()
            
        y_pred, y_true = self.test_iterate(epoch, 'test')
        
        if self.local_rank == 0:
            print(y_pred.shape, y_true.shape)
            print('total', accuracy_score(y_true, y_pred))
            for i in range(self.n_classes):
                idx = y_true == i
                if np.sum(idx) > 0:  # Only compute accuracy if there are samples
                    print('class', i, accuracy_score(y_true[idx], y_pred[idx]))


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # Initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def train_worker(rank, world_size, config):
    import torch
    from torch.utils.data import Sampler
    import random
    from collections import defaultdict
    
    from torch.utils.data import Subset

    from torch.utils.data import Sampler
    
    import random
    from collections import defaultdict

    from torch.utils.data import Sampler
    import random
    from collections import defaultdict
    
    class ClassBalancedBatchSampler(Sampler):
        def __init__(self, dataset, k_classes, n_samples, world_size=1, rank=0, seed=42, iterations_per_epoch=100):
            """
            Class-balanced batch sampler that can be used with distributed training.
            
            Args:
                dataset: Dataset to sample from
                k_classes: Number of classes to sample per batch
                n_samples: Number of samples per class
                world_size: Number of processes/GPUs in distributed training
                rank: Rank of current process
                seed: Random seed
                iterations_per_epoch: Number of batches to generate per epoch
            """
            self.dataset = dataset
            self.k_classes = k_classes
            self.n_samples = n_samples
            self.world_size = world_size
            self.rank = rank
            self.seed = seed
            self.iterations_per_epoch = iterations_per_epoch
            
            # Get targets (handle Subset)
            if isinstance(dataset, torch.utils.data.Subset):
                if hasattr(dataset.dataset, 'targets'):
                    targets = [dataset.dataset.targets[i] for i in dataset.indices]
                else:  # Handle case where targets are in another attribute or format
                    targets = [dataset.dataset.samples[i][1] for i in dataset.indices]
                indices = dataset.indices
            else:
                if hasattr(dataset, 'targets'):
                    targets = dataset.targets
                else:  # Handle case where targets are in another attribute or format
                    targets = [sample[1] for sample in dataset.samples]
                indices = list(range(len(targets)))
    
            # Build class to index mapping
            self.class_to_indices = defaultdict(list)
            for idx, target in zip(indices, targets):
                self.class_to_indices[target].append(idx)
    
            self.classes = [cls for cls in self.class_to_indices.keys() 
                           if len(self.class_to_indices[cls]) >= n_samples]
            
            if len(self.classes) < k_classes:
                raise ValueError(f"Only {len(self.classes)} classes have {n_samples} or more samples. "
                               f"Cannot sample {k_classes} classes.")
                
            self.epoch = 0
            self.samples_per_gpu = k_classes * n_samples // world_size
            
            if k_classes * n_samples % world_size != 0:
                raise ValueError(f"k_classes ({k_classes}) * n_samples ({n_samples}) = {k_classes * n_samples} "
                               f"must be divisible by world_size ({world_size})")
    
        def __iter__(self):
            # Create new random generator with seed dependent on epoch and rank
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch + self.rank)
            
            # For each batch
            for _ in range(self.iterations_per_epoch):
                batch = []
                
                # Sample k_classes with replacement
                selected_classes = random.sample(self.classes, self.k_classes)
                
                # For each class, sample n_samples instances
                for cls in selected_classes:
                    # Sample with replacement if necessary
                    available_indices = self.class_to_indices[cls]
                    if len(available_indices) < self.n_samples:
                        samples = random.choices(available_indices, k=self.n_samples)
                    else:
                        samples = random.sample(available_indices, self.n_samples)
                    batch.extend(samples)
                
                # Ensure consistent shuffling across processes
                indices = torch.tensor(batch, dtype=torch.int64)
                indices = indices[torch.randperm(len(indices), generator=g)]
                batch = indices.tolist()
                
                # Each GPU gets an equal portion of the batch
                local_batch = batch[self.rank::self.world_size]
                yield local_batch
    
        def __len__(self):
            # Return number of batches per epoch
            return self.iterations_per_epoch
    
        def set_epoch(self, epoch):
            self.epoch = epoch



    # Setup process group
    setup(rank, world_size)
    
    # Set the device
    torch.cuda.set_device(rank)
    
    if rank == 0:
        wandb.init(
            project=config['wandb_project'],
            entity=config['wandb_entity'],
            group=config['wandb_group'],
        )
    
    # Set seed for reproducibility
    torch.manual_seed(config['seed'])
    
    # ImageNet normalization
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.IMAGENET),
        transforms.ToTensor(),
        normalize,
    ])

    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    # Create datasets
    full_trainset = datasets.ImageFolder(config['train_dir'], transform=transform_train)
    N = len(full_trainset)
    Ntrain, Nval = N - int(N * config['train_val_split']), int(N * config['train_val_split'])
    
    # Use random seed for reproducibility across processes
    generator = torch.Generator().manual_seed(config['seed'])
    trainset, valset = random_split(full_trainset, [Ntrain, Nval], generator=generator)
    testset = datasets.ImageFolder(config['val_dir'], transform=transform_test)

    # Create distributed samplers
    train_sampler = ClassBalancedBatchSampler(
        dataset=trainset,
        k_classes=config['k_classes'],
        n_samples=config['n_samples'],
        world_size=world_size,
        rank=rank,
        seed=config['seed'],
        iterations_per_epoch=config.get('iterations_per_epoch', 100)  # Default to 100 batches per epoch
    )

    val_sampler = DistributedSampler(valset, num_replicas=world_size, rank=rank, shuffle=False)
    test_sampler = DistributedSampler(testset, num_replicas=world_size, rank=rank, shuffle=False)

    # Create dataloaders
    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_sampler=train_sampler,
        num_workers=config['num_workers'],
        pin_memory=True,
    )

    
    valloader = torch.utils.data.DataLoader(
        valset, 
        batch_size=config['batch_size'],
        sampler=val_sampler,
        num_workers=config['num_workers'],
        pin_memory=True,
    )
    
    testloader = torch.utils.data.DataLoader(
        testset, 
        batch_size=config['batch_size'],
        sampler=test_sampler,
        num_workers=config['num_workers'],
        pin_memory=True,
    )

    dataloaders = {'train': trainloader, 'val': valloader, 'test': testloader}

    if config['loss'] == 'LDA':
        lda_args = {'lamb': config['lamb'], 'n_eig': config['n_eig'], 'margin': config['margin']}
    else:
        lda_args = {}

    # Create solver with distributed info
    solver = Solver(
        dataloaders=dataloaders, 
        model_path=config['model_path'],
        n_classes=config['n_classes'],
        lda_args=lda_args,
        local_rank=rank,
        world_size=world_size
    )
    
    # Train
    solver.train(config['epochs'])
    
    # Test
    solver.test()
    
    # Clean up
    cleanup()


if __name__ == '__main__':
    config = {
        'wandb_project': "DELETEME",#"DeepLDA",
        'wandb_entity': "gerardo-pastrana-c3-ai",
        'wandb_group': "gapLoss",
        'seed': 42,
        'n_classes': 1000,
        'train_val_split': 0.1,
        'batch_size': 4096,
        'num_workers': 1,  
        'train_dir': 'datasets/imagenet_full_size/061417/train',
        'val_dir': 'datasets/imagenet_full_size/061417/val',
        'model_path': 'models/deeplda_best.pth',
        'loss': 'LDA',
        'lamb': 0.1,
        'n_eig': 4,
        'margin': None,
        'epochs': 100,
        'k_classes': 20,  # for example
        'n_samples': 5,   # 5 samples per class

    }
    
    # Number of available GPUs
    n_gpus = 4
    
    # Launch processes
    mp.spawn(
        train_worker,
        args=(n_gpus, config),
        nprocs=n_gpus,
        join=True
    )