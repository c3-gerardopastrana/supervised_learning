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

class Lamb(optim.Optimizer):
    r"""Implements Lamb algorithm.

    It has been proposed in `Large Batch Optimization for Deep Learning: Training BERT in 76 minutes`_.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        adam (bool, optional): always use trust ratio = 1, which turns this into
            Adam. Useful for comparison purposes.

    .. _Large Batch Optimization for Deep Learning: Training BERT in 76 minutes:
        https://arxiv.org/abs/1904.00962
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-6,
                 weight_decay=0, adam=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay)
        self.adam = adam
        super(Lamb, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Lamb does not support sparse gradients, consider SparseAdam instad.')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                # Decay the first and second moment running average coefficient
                # m_t
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                # v_t
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Paper v3 does not use debiasing.
                # bias_correction1 = 1 - beta1 ** state['step']
                # bias_correction2 = 1 - beta2 ** state['step']
                # Apply bias to lr to avoid broadcast.
                step_size = group['lr'] # * math.sqrt(bias_correction2) / bias_correction1

                weight_norm = p.data.pow(2).sum().sqrt().clamp(0, 10)

                adam_step = exp_avg / exp_avg_sq.sqrt().add(group['eps'])
                if group['weight_decay'] != 0:
                    adam_step.add_(p.data, alpha=group['weight_decay'])

                adam_norm = adam_step.pow(2).sum().sqrt()
                if weight_norm == 0 or adam_norm == 0:
                    trust_ratio = 1
                else:
                    trust_ratio = weight_norm / adam_norm
                state['weight_norm'] = weight_norm
                state['adam_norm'] = adam_norm
                state['trust_ratio'] = trust_ratio
                if self.adam:
                    trust_ratio = 1

                p.data.add_(adam_step, alpha=-step_size * trust_ratio)

        return loss
class LARS(optim.Optimizer):
    def __init__(
        self,
        params,
        lr,
        weight_decay=0,
        momentum=0.9,
        eta=0.001,
        weight_decay_filter=None,
        lars_adaptation_filter=None,
    ):
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            eta=eta,
            weight_decay_filter=weight_decay_filter,
            lars_adaptation_filter=lars_adaptation_filter,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g["params"]:
                dp = p.grad

                if dp is None:
                    continue

                if g["weight_decay_filter"] is None or not g["weight_decay_filter"](p):
                    dp = dp.add(p, alpha=g["weight_decay"])

                if g["lars_adaptation_filter"] is None or not g[
                    "lars_adaptation_filter"
                ](p):
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)
                    one = torch.ones_like(param_norm)
                    q = torch.where(
                        param_norm > 0.0,
                        torch.where(
                            update_norm > 0, (g["eta"] * param_norm / update_norm), one
                        ),
                        one,
                    )
                    dp = dp.mul(q)

                param_state = self.state[p]
                if "mu" not in param_state:
                    param_state["mu"] = torch.zeros_like(p)
                mu = param_state["mu"]
                mu.mul_(g["momentum"]).add_(dp)

                p.add_(mu, alpha=-g["lr"])




import torch
import torch.nn as nn
import torch.nn.functional as F


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
    def __init__(self, block, num_blocks, num_classes=10, lda_args=None):
        super(ResNet, self).__init__()
        self.lda_args = lda_args
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)  # CIFAR-style
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)
        self.projector = nn.Linear(512 * block.expansion, 12)
        # self.projector = nn.Sequential(
        #         nn.Linear(512 * block.expansion, 128),
        #         nn.BatchNorm1d(128),
        #         nn.ReLU(),
        #         nn.Linear(128,64)
        # )
        
        if self.lda_args:
            self.lda = SphericalLDA(num_classes, lda_args['lamb'])  # your LDA class
        else:
            self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, y=None, epoch = 0):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = F.avg_pool2d(out, 4)  # CIFAR final feature map is 4x4
        fea = out.view(out.size(0), -1)  # Nx512

        
        

        
    # self.projector = nn.Sequential(
    #             nn.Linear(512 * block.expansion, 128),
    #             nn.BatchNorm1d(128),
    #             nn.Relu(),
    #             nn.Linear(128,64)
    #     )
        
        if self.lda_args:
            scale = 30.0  # common in ArcFace
            logits = scale * F.normalize(fea, dim=-1) @ F.normalize(self.linear.weight, dim=-1).T
            
            fea = self.projector(fea)        # Nx128
            fea = F.normalize(fea, p=2, dim=1)  # L2 normalization
            hasComplexEVal, out, sigma_w_inv_b = self.lda(fea, y)
            return hasComplexEVal, fea, out, sigma_w_inv_b, logits
        else:
            out = self.linear(fea)
            return out


def ResNet18(num_classes=10, lda_args=None):
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
        
        #self.optimizer = optim.SGD(self.net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
        
        def exclude_bias_and_norm(p):
            return p.ndim == 1  # biases and BN params are 1D
        self.optimizer = optim.AdamW(self.net.parameters(), lr=1e-3, weight_decay=5e-4)
        # self.optimizer = Lamb(
        #    self.net.parameters(),
        #     lr=1e-2,
        #     weight_decay=5e-4, #Try decreasing it to 1e-6
        # )

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
                hasComplexEVal, feas, outputs, sigma_w_inv_b, ce = self.net(inputs, targets, epoch)
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
        
                    #Loss
                    import math
                    
                    def get_ce_weight(epoch, warmup_epochs=5, decay_epochs=30):
                        """
                        Returns the weight for the cross-entropy loss:
                        - Warms up linearly over `warmup_epochs`
                        - Decays with cosine until `decay_epochs`
                        """
                        warmup = min(epoch / warmup_epochs, 1.0)
                        decay = 0.5 * (1 + math.cos(min(epoch, decay_epochs) / decay_epochs * math.pi))
                        return warmup * decay

                    ce_weight = get_ce_weight(epoch)
                    loss = self.criterion(sigma_w_inv_b) + max(0.0, 1.0 - epoch / 30) * F.cross_entropy(ce, targets)


                    #loss = (eigvals_norm * eigvals_norm.log()).sum()
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
    wandb.init(
    project="DeepLDA",
    entity="gerardo-pastrana-c3-ai",
    group="gapLoss",
    )
    
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=2),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    seed = 42
    n_classes = 10
    train_val_split = 0.2
    batch_size = 5000
    num_workers = 4
    gpu = 2

    train_dir = '../data/cifar10/imgs/train'
    test_dir = '../data/cifar10/imgs/test'
    model_path = '../data/cifar10/exp1015/deeplda_best.pth'

    loss = 'LDA' # CE or LDA
    lamb = 0.1#0.0001
    n_eig = 4
    margin = None
    lda_args = {'lamb':lamb, 'n_eig':n_eig, 'margin':margin} if loss == 'LDA' else {}
    

    class_map = {'airplane':0, 'automobile':1, 'bird':2, 'cat':3, 'deer':4, 
                 'dog':5, 'frog':6, 'horse':7, 'ship':8, 'truck':9}
    
    ids = parse_dir(train_dir, os.listdir(train_dir))
    train_img_names = [os.path.join(train_dir, *f.split('+')) for f in ids]
    trainset = CIFAR10(train_img_names, class_map, transform_train)
    N = len(trainset)
    Ntrain, Nval = N - int(N * train_val_split), int(N * train_val_split)
    trainset, valset = torch.utils.data.random_split(trainset, [Ntrain, Nval])
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valloader = torch.utils.data.DataLoader(
        valset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    test_ids = parse_dir(test_dir, os.listdir(test_dir))
    test_img_names = [os.path.join(test_dir, *f.split('+')) for f in test_ids]
    testset = CIFAR10(test_img_names, class_map, transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    dataloaders = {'train':trainloader, 'val':valloader, 'test':testloader}
    solver = Solver(dataloaders, model_path, n_classes, lda_args, gpu)
    
    solver.train(500)
    solver.test()
