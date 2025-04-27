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
import torchvision.transforms as transforms
import torch.optim as optim
import wandb
from functools import partial
from lda import LDA, lda_loss, sina_loss, SphericalLDA

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

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=1000, lda_args=None):
        super(ResNet, self).__init__()
        self.lda_args = lda_args
        self.in_planes = 64

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

    def forward(self, x, y=None, epoch=0):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgpool(out)  # output shape: [B, 512, 1, 1]
        fea = out.view(out.size(0), -1)  # flatten to [B, 512]

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


class Solver:
    def __init__(self, dataloaders, model_path, n_classes, lda_args={}, gpu=-1):
        self.dataloaders = dataloaders
        if gpu >= 0:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
            self.device = torch.device('cuda:0')
        else:
            self.device = torch.device('cpu:0')
        self.net = ResNet18(n_classes, lda_args)
        self.net = self.net.to(self.device)
        self.use_lda = True if lda_args else False
        if self.use_lda:
            self.criterion = partial(lda_loss, n_classes=n_classes, 
                                    n_eig=lda_args['n_eig'], margin=lda_args['margin'])
            self.criterion = sina_loss
        else:
            self.criterion = nn.CrossEntropyLoss()
        print(self.criterion)

        self.optimizer = optim.AdamW(self.net.parameters(), lr=1e-3, weight_decay=5e-4)
        self.model_path = model_path
        self.n_classes = n_classes

    def iterate(self, epoch, phase):
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
                hasComplexEVal, feas, outputs, sigma_w_inv_b = self.net(inputs, targets, epoch)
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
    
                    loss =  self.criterion(sigma_w_inv_b)

                    outputs = self.net.lda.predict_proba(feas)

                    if phase == 'train':
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
                            "quantile_75": quantile_75})
                    
                else:
                    print('Complex Eigen values found, skip backpropagation of {}th batch'.format(batch_idx))
                    continue
            else:
                outputs = self.net(inputs, targets, epoch)
                loss = nn.CrossEntropyLoss()(outputs, targets)
        
            if phase == 'train':
               
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=5.0)
                self.optimizer.step()
                wandb.log({"total_grad_norm_encoder":grad_norm.item()})
            total_loss += loss.item()
    
            outputs = torch.argmax(outputs.detach(), dim=1)
            total += targets.size(0)
            correct += outputs.eq(targets).sum().item()
        
        total_loss /= (batch_idx + 1)
        if total > 0:
            total_acc = correct / total
        else:
            total_acc = 0 
        
        if entropy_count > 0:
            average_entropy = entropy_sum / entropy_count
            print(f'Average Entropy: {average_entropy:.4f}')
        
        print('\nepoch %d: %s loss: %.3f | acc: %.2f%% (%d/%d)'
                     % (epoch, phase, total_loss, 100.*total_acc, correct, total))
        wandb.log({
            "epoch"+phase:epoch,
             "total"+phase:total_loss,
             "total_acc_train"+phase: 100.*total_acc
        }) 
        return total_loss, total_acc


    def train(self, epochs):
        best_loss = float('inf')
        for epoch in range(epochs):
            self.iterate(epoch, 'train')
            with torch.no_grad():
                val_loss, val_acc = self.iterate(epoch, 'val')
            if val_loss < best_loss:
                best_loss = val_loss
                checkpoint = {'epoch':epoch, 'val_loss':val_loss, 'state_dict':self.net.state_dict()}
                print('best val loss found')
            print()
        torch.save(checkpoint, self.model_path)

    def test_iterate(self, epoch, phase):
        self.net.eval()
        dataloader = self.dataloaders[phase]
        y_pred = []
        y_true = []
        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                if self.use_lda:
                    _, feas, outputs = self.net(inputs, targets, epoch)
                    outputs = self.net.lda.predict_proba(feas)
                else:
                    outputs = self.net(inputs, targets, epoch)
                outputs = torch.argmax(outputs, dim=1)
                y_pred.append(outputs.detach().cpu().numpy())
                y_true.append(targets.detach().cpu().numpy())
        return np.array(y_pred).flatten(), np.array(y_true).flatten()
        
    def test(self):
        checkpoint = torch.load(self.model_path)
        epoch = checkpoint['epoch']
        val_loss = checkpoint['val_loss']
        self.net.load_state_dict(checkpoint['state_dict'])
        print('load model at epoch {}, with val loss: {:.3f}'.format(epoch, val_loss))
        y_pred, y_true = self.test_iterate(epoch, 'test')
        print(y_pred.shape, y_true.shape)

        print('total', accuracy_score(y_true, y_pred))
        for i in range(self.n_classes):
            idx = y_true == i
            print('class', i, accuracy_score(y_true[idx], y_pred[idx]))


def parse_dir(img_dir, classes, randnum=-1):
    img_names = []
    ids = []
    for clazz in classes:
        sub_dir = os.path.join(img_dir, clazz)
        sub_files = [os.path.join(sub_dir, f) for f in os.listdir(sub_dir)]
        if len(sub_files) > randnum > 0:
            sub_files = random.sample(sub_files, randnum)
        img_names += sub_files
    for img_name in img_names:
        clazz = os.path.basename(os.path.dirname(img_name))
        id = clazz + '+' + os.path.basename(img_name)
        ids.append(id)
    return ids


if __name__ == '__main__':
    import os
    import wandb
    import torch
    from torchvision import transforms, datasets
    from torch.utils.data import DataLoader, random_split

    wandb.init(
        project="DeepLDA",
        entity="gerardo-pastrana-c3-ai",
        group="gapLoss",
    )

    seed = 42
    torch.manual_seed(seed)

    n_classes = 1000
    train_val_split = 0.1
    batch_size = 128
    num_workers = 8
    gpu = 0

    train_dir = 'datasets/imagenet_full_size/061417/train'
    val_dir = 'datasets/imagenet_full_size/061417/val'
    model_path = 'models/deeplda_best.pth'

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
    full_trainset = datasets.ImageFolder(train_dir, transform=transform_train)
    N = len(full_trainset)
    Ntrain, Nval = N - int(N * train_val_split), int(N * train_val_split)
    trainset, valset = random_split(full_trainset, [Ntrain, Nval])

    testset = datasets.ImageFolder(val_dir, transform=transform_test)

    # Dataloaders
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    valloader = DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    dataloaders = {'train': trainloader, 'val': valloader, 'test': testloader}

    loss = 'CE'
    lamb = 0.0001
    n_eig = 4
    margin = None
    lda_args = {'lamb': lamb, 'n_eig': n_eig, 'margin': margin} if loss == 'LDA' else {}

    solver = Solver(dataloaders, model_path, n_classes, lda_args, gpu)
    solver.train(100)
    solver.test()
