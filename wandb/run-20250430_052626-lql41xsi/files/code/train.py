import os
import random
import gc
from collections import defaultdict
from functools import partial

import numpy as np
np.set_printoptions(precision=4, suppress=True)

from PIL import Image
from tqdm.notebook import tqdm
from sklearn.metrics import accuracy_score

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.checkpoint import checkpoint, checkpoint_sequential
from torch.utils.data import DataLoader, random_split, Sampler, Subset
from torch.utils.data.distributed import DistributedSampler

import torchvision
from torchvision import transforms, datasets

import wandb

from lda import LDA, lda_loss, sina_loss, SphericalLDA
from models import ResNet, BasicBlock
from utils import compute_wandb_metrics

def ResNet18(num_classes=1000, lda_args=None, use_checkpoint=True, segments=4):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes, lda_args, use_checkpoint, segments)


class Solver:
    def __init__(self, dataloaders, model_path, n_classes, lda_args={}, local_rank=0, world_size=1, lr=1e-3, 
                 gradient_accumulation_steps=1, use_amp=True, use_checkpoint=True):
        self.dataloaders = dataloaders
        self.local_rank = local_rank
        self.world_size = world_size
        self.device = torch.device(f'cuda:{local_rank}')
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.use_amp = use_amp
        
        # Create model with checkpointing enabled
        self.net = ResNet18(n_classes, lda_args, use_checkpoint=use_checkpoint)
        self.net = self.net.to(self.device)
        
        # Wrap model with DDP
        if world_size > 1:
            self.net = DDP(self.net, device_ids=[local_rank], output_device=local_rank,
                           find_unused_parameters=False)  # Set to True only if needed
        
        self.use_lda = True if lda_args else False
        if self.use_lda:
            self.criterion = sina_loss  # Assuming this is defined elsewhere
        else:
            self.criterion = nn.CrossEntropyLoss()
        
        if local_rank == 0:
            print(f"Using criterion: {self.criterion}")
            print(f"Using checkpoint: {use_checkpoint}")
            print(f"Using mixed precision: {use_amp}")
            print(f"Gradient accumulation steps: {gradient_accumulation_steps}")

        self.optimizer = optim.AdamW(self.net.parameters(), lr=lr, weight_decay=5e-4)
        self.scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
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

        # Clear CUDA cache before each epoch
        torch.cuda.empty_cache()
        gc.collect()
        
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            # Move data to device
            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            
            # For training with gradient accumulation
            if phase == 'train':
                effective_batch_idx = batch_idx % self.gradient_accumulation_steps
                
                #if effective_batch_idx == 0:
                self.optimizer.zero_grad(set_to_none=True)  # More memory efficient
                
                # Apply mixed precision for training
                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    if self.use_lda:
                        if isinstance(self.net, DDP):
                            hasComplexEVal, feas, outputs, sigma_w_inv_b = self.net.module(inputs, targets, epoch)
                        else:
                            hasComplexEVal, feas, outputs, sigma_w_inv_b = self.net(inputs, targets, epoch)
                        
                        if not hasComplexEVal:
                            # Stats calculation (same as original)
                            metrics = compute_wandb_metrics(outputs, sigma_w_inv_b)
                            entropy_sum += metrics["entropy"]
                            entropy_count += 1
                            loss = self.criterion(sigma_w_inv_b)
                            
                            if isinstance(self.net, DDP):
                                outputs = self.net.module.lda.predict_proba(feas)
                            else:
                                outputs = self.net.lda.predict_proba(feas)
                            
                            # Only log on rank 0 for efficiency
                            if phase == 'train' and self.local_rank == 0:
                                wandb.log(metrics, commit=False)
                                wandb.log({
                                    'loss': loss.item(),
                                    'epoch': epoch,
                                }, commit=False)
                        else:
                            if self.local_rank == 0:
                                print(f'Complex Eigenvalues found, skipping batch {batch_idx}')
                            continue
                    else:
                        outputs = self.net(inputs, targets, epoch)
                        loss = self.criterion(outputs, targets)
                
                # Scale loss for gradient accumulation
                #loss = loss / self.gradient_accumulation_steps
                
                if phase == 'train':
                    # Use gradient scaler for mixed precision
                    self.scaler.scale(loss).backward()
                    
                    # Step optimizer at effective batch boundaries
                    #if (effective_batch_idx == self.gradient_accumulation_steps - 1) or (batch_idx == len(dataloader) - 1):
                    # Unscale before clipping
                    self.scaler.unscale_(self.optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=100.0)
                    
                    # Update with scaler
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    
                    if self.local_rank == 0:
                        wandb.log({"grad_norm": grad_norm.item()})
            else:
                # Validation phase - no gradients needed
                with torch.no_grad():
                    if self.use_lda:
                        if isinstance(self.net, DDP):
                            hasComplexEVal, feas, outputs, sigma_w_inv_b = self.net.module(inputs, targets, epoch)
                        else:
                            hasComplexEVal, feas, outputs, sigma_w_inv_b = self.net(inputs, targets, epoch)
                        
                        if not hasComplexEVal:
                            loss = self.criterion(sigma_w_inv_b)
                            
                            if isinstance(self.net, DDP):
                                outputs = self.net.module.lda.predict_proba(feas)
                            else:
                                outputs = self.net.lda.predict_proba(feas)
                        else:
                            continue
                    else:
                        outputs = self.net(inputs, targets, epoch)
                        loss = self.criterion(outputs, targets)
            
            # Accumulate metrics
            total_loss += loss.item()  if phase == 'train' else loss.item()
            
            outputs = torch.argmax(outputs.detach(), dim=1)
            total += targets.size(0)
            correct += outputs.eq(targets).sum().item()
            
            # Free memory after each batch
            del inputs, targets, outputs
            if phase == 'train' and self.use_lda and not hasComplexEVal:
                del feas, sigma_w_inv_b
            torch.cuda.empty_cache()
        
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
        
        # Log metrics
        if self.local_rank == 0:
            if entropy_count > 0:
                average_entropy = entropy_sum / entropy_count
                print(f'Average Entropy: {average_entropy:.4f}')
            
            print(f'\nepoch {epoch}: {phase} loss: {total_loss:.3f} | acc: {100.*total_acc:.2f}% ({correct}/{total})')
            wandb.log({
                f"epoch_{phase}": epoch,
                f"loss_{phase}": total_loss,
                f"acc_{phase}": 100.*total_acc
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
            
            # Training phase
            self.iterate(epoch, 'train')
            
            # Validation phase
            with torch.no_grad():
                val_loss, val_acc = self.iterate(epoch, 'val')
                
                
            # Save best model
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

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # Initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()
    
def train_worker(rank, world_size, config):
    class ClassBalancedBatchSampler(Sampler):
        def __init__(self, dataset, k_classes, n_samples,
                     world_size=1, rank=0, seed=42):
            """
            Class-balanced batch sampler for distributed training.
            
            Args:
                dataset: Dataset to sample from
                k_classes: Number of classes per batch
                n_samples: Number of samples per class
                world_size: Number of processes (GPUs)
                rank: Local rank of this process
                seed: Random seed
            """
            super().__init__(dataset)
            self.dataset = dataset
            self.k_classes = k_classes
            self.n_samples = n_samples
            self.world_size = world_size
            self.rank = rank
            self.seed = seed
            self.epoch = 0  # must be set each epoch manually!
    
            # Build mapping from class to list of indices
            if isinstance(dataset, torch.utils.data.Subset):
                targets = [dataset.dataset.targets[i] for i in dataset.indices]
            else:
                targets = dataset.targets
            
            self.class_to_indices = {}
            for idx, target in enumerate(targets):
                if target not in self.class_to_indices:
                    self.class_to_indices[target] = []
                self.class_to_indices[target].append(idx)
    
            # Only keep classes that have enough samples
            self.available_classes = [cls for cls, idxs in self.class_to_indices.items()
                                      if len(idxs) >= n_samples]
            
            assert len(self.available_classes) >= k_classes, \
                f"Only {len(self.available_classes)} classes have {n_samples}+ samples, but need {k_classes}"
    
            # Compute approximately how many batches can fit
            total_samples = sum(len(self.class_to_indices[cls]) for cls in self.available_classes)
            batch_size = self.k_classes * self.n_samples
            self.batches_per_epoch = total_samples // batch_size
    
        def set_epoch(self, epoch):
            self.epoch = epoch
    
        def __iter__(self):
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch + self.rank)

            num_batches = 0
            while num_batches < self.batches_per_epoch:
                selected_classes = torch.tensor(self.available_classes)
                selected_classes = selected_classes[torch.randperm(len(selected_classes), generator=g)][:self.k_classes]
            
                batch = []
                for cls in selected_classes.tolist():
                    indices = self.class_to_indices[cls]
                    indices_tensor = torch.tensor(indices)
                    chosen_indices = indices_tensor[torch.randperm(len(indices_tensor), generator=g)][:self.n_samples]
                    batch.extend(chosen_indices.tolist())
            
                # Shard based on rank
                if num_batches % self.world_size == self.rank:
                    yield batch
            
                num_batches += 1

    
            # all_batches = []
    
            # while len(all_batches) < self.batches_per_epoch:
            #     # Pick k_classes randomly
            #     selected_classes = torch.tensor(self.available_classes)
            #     selected_classes = selected_classes[torch.randperm(len(selected_classes), generator=g)][:self.k_classes]
    
            #     batch = []
            #     for cls in selected_classes.tolist():
            #         indices = self.class_to_indices[cls]
            #         indices_tensor = torch.tensor(indices)
            #         chosen_indices = indices_tensor[torch.randperm(len(indices_tensor), generator=g)][:self.n_samples]
            #         batch.extend(chosen_indices.tolist())
    
            #     all_batches.append(batch)
    
            # # Shard batches across GPUs
            # local_batches = all_batches[self.rank::self.world_size]
    
            # for batch in local_batches:
            #     yield batch
    
        def __len__(self):
            return self.batches_per_epoch // self.world_size
            
    # Configure CUDA
    #os.environ['CUDA_VISIBLE_DEVICES'] = config.get('cuda_visible_devices', '')  # Optional GPU ID restrictions
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'  # Help with fragmentation
    
    # Setup process group
    setup(rank, world_size)
    
    # Set the device
    torch.cuda.set_device(rank)
    
    # Initialize wandb only on rank 0
    if rank == 0:
        wandb.init(
            project=config['wandb_project'],
            entity=config['wandb_entity'],
            group=config['wandb_group'],
            config=config,  # Track configuration
        )
    
    # Set seeds for reproducibility
    seed = config['seed'] + rank  # Different seed per process
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # Calculate effective batch size and adjust learning rate
    global_batch_size = config['k_classes'] * config['n_samples'] * world_size
    gradient_accumulation_steps = config.get('gradient_accumulation_steps', 1)
    effective_batch_size = global_batch_size * gradient_accumulation_steps
    base_lr = config.get('base_lr', 1e-3)
    lr = base_lr#get_scaled_lr_sqrt(effective_batch_size, base_batch_size=config.get('base_batch_size', 128), base_lr=base_lr)
    
    if rank == 0:
        print(f"Global batch size: {global_batch_size}")
        print(f"Gradient accumulation steps: {gradient_accumulation_steps}")
        print(f"Effective batch size: {effective_batch_size}")
        print(f"Learning rate: {lr}")
    
    # Data loading code (same as original)
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
    trainset = datasets.ImageFolder(config['train_dir'], transform=transform_train)
    valset = datasets.ImageFolder(config['val_dir'], transform=transform_test)
    testset = datasets.ImageFolder(config['test_dir'], transform=transform_test)

    # Create distributed samplers
    train_sampler = ClassBalancedBatchSampler(
        dataset=trainset,
        k_classes=config['k_classes'],
        n_samples=config['n_samples'],
        world_size=world_size,
        rank=rank,
        seed=config['seed']
    )


    val_sampler = DistributedSampler(valset, num_replicas=world_size, rank=rank, shuffle=False)
    test_sampler = DistributedSampler(testset, num_replicas=world_size, rank=rank, shuffle=False)

    # Create dataloaders
    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_sampler=train_sampler,
        num_workers=config['num_workers'],
        pin_memory=True,
        #persistent_workers=False
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
        
    # Create solver with optimized parameters
    solver = Solver(
        dataloaders=dataloaders, 
        model_path=config['model_path'],
        n_classes=config['n_classes'],
        lda_args=lda_args if config['loss'] == 'LDA' else {},
        local_rank=rank,
        world_size=world_size,
        lr=lr,
        gradient_accumulation_steps=gradient_accumulation_steps,
        use_amp=config.get('use_amp', True),
        use_checkpoint=config.get('use_checkpointing', True)
    )
    
    # Train
    solver.train(config['epochs'])
    
    # Test
    solver.test()
    
    # Clean up
    cleanup()


if __name__ == '__main__':
    # Configuration with memory optimizations
    config = {
        'wandb_project': "DELETEME",
        'wandb_entity': "gerardo-pastrana-c3-ai",
        'wandb_group': "gapLoss",
        'seed': 42,
        'n_classes': 1000,
        'train_val_split': 0.1,
        'batch_size': 4096,  # Global batch size
        'num_workers': 1,  # Adjust based on CPU cores
        'train_dir': '/data/datasets/imagenet_full_size/061417/train',
        'val_dir': '/data/datasets/imagenet_full_size/061417/val',
        'test_dir': '/data/datasets/imagenet_full_size/061417/test',
        'model_path': 'models/deeplda_best.pth',
        'loss': 'LDA',
        'lamb': 0.1,
        'n_eig': 4,
        'margin': None,
        'epochs': 20,
        'k_classes':128 ,
        'n_samples': 64,
        # Memory optimization parameters
        'gradient_accumulation_steps': 1,  # Accumulate gradients to save memory
        'use_amp': True,                   # Use automatic mixed precision
        'use_checkpointing': True,         # Use gradient checkpointing
        'base_lr': 1e-3,                   # Base learning rate
        'base_batch_size': 128,            # Reference batch size for LR scaling
        'cuda_visible_devices': '',        # Optional GPU restrictions
    }
    
    # Number of available GPUs
    n_gpus = torch.cuda.device_count()
    print(f"Using {n_gpus} GPUs")
    
    # Launch processes
    mp.spawn(
        train_worker,
        args=(n_gpus, config),
        nprocs=n_gpus,
        join=True
    )