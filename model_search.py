import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import ceil

from operations import *
from torch.autograd import Variable
from genotypes import PRIMITIVES
from genotypes import Genotype

k_list = [4, 4, 4, 4, 4, 4, 4, 4]


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        # MLP
        # self.fc1 = nn.Conv2d(in_planes*2, in_planes // 2, 1, bias=False)
        # self.relu1 = nn.ReLU()
        # self.fc2 = nn.Conv2d(in_planes // 2, in_planes, 1, bias=False)
        # self.sigmoid = nn.Sigmoid()
        self.fc1 = nn.Conv2d(in_planes * 2, len(PRIMITIVES), 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(10, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, weights_all):
        out = self.fc2(self.relu1((torch.mm(weights_all.t(), weights_all).expand(len(x), -1, -1) @ (
            self.fc1(torch.cat((self.avg_pool(x), self.max_pool(x)), dim=1))[:, :, :, 0])).unsqueeze(-1)))
        # avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        # max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        # out = avg_out + max_out
        # weights_all.t()@weights_all@
        return self.sigmoid(out)


class MixedOp(nn.Module):

    def __init__(self, C, stride, layers):
        super(MixedOp, self).__init__()
        self._ops = nn.ModuleList()
        self.mp = nn.MaxPool2d(2, 2)

        self.k = k_list[layers]
        self.ca = ChannelAttention(C)
        for primitive in PRIMITIVES:

            op = OPS[primitive](C // self.k, stride, False)
            if 'pool' in primitive:
                op = nn.Sequential(op, nn.BatchNorm2d(C // self.k, affine=False))
            self._ops.append(op)

    def forward(self, x, weights, weights_all):
        dim_2 = x.shape[1]
        num_list = self.ca(x, weights_all)
        x = x * num_list
        num_dict = []
        slist = torch.sum(num_list, dim=0, keepdim=True)
        values, max_num_index = slist.topk(dim_2 // self.k, dim=1, largest=True, sorted=True)

        for i in range(0, len(max_num_index[0])):
            num_dict.append(max_num_index[0][i])

        xtemp = torch.index_select(x, 1, torch.tensor(num_dict).cuda())
        temp1 = sum(w * op(xtemp) for w, op in zip(weights, self._ops))

        if temp1.shape[2] == x.shape[2]:
            x[:, num_dict, :, :] = temp1[:, :, :, :]
        else:
            x = self.mp(x)
            x[:, num_dict, :, :] = temp1[:, :, :, :]

        return x


class Cell(nn.Module):

    def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev, layers):
        super(Cell, self).__init__()
        self.reduction = reduction

        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)
        self._steps = steps
        self._multiplier = multiplier

        self._ops = nn.ModuleList()
        self._bns = nn.ModuleList()
        for i in range(self._steps):
            for j in range(2 + i):
                stride = 2 if reduction and j < 2 else 1
                op = MixedOp(C, stride, layers)
                self._ops.append(op)

    def forward(self, s0, s1, weights, weights2):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        offset = 0
        for i in range(self._steps):
            s = sum(weights2[offset + j] * self._ops[offset + j](h, weights[offset + j], weights) for j, h in
                    enumerate(states))
            offset += len(states)
            states.append(s)

        return torch.cat(states[-self._multiplier:], dim=1)


class Network(nn.Module):

    def __init__(self, C, num_classes, layers, criterion, steps=4, multiplier=4, stem_multiplier=3):
        super(Network, self).__init__()
        self._C = C
        self._num_classes = num_classes
        self._layers = layers
        self._criterion = criterion
        self._steps = steps
        self._multiplier = multiplier

        C_curr = stem_multiplier * C
        self.stem = nn.Sequential(
            nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
            nn.BatchNorm2d(C_curr)
        )

        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = nn.ModuleList()
        reduction_prev = False
        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev, i)
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, multiplier * C_curr

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)

        self._initialize_alphas()

    def new(self):
        model_new = Network(self._C, self._num_classes, self._layers, self._criterion).cuda()
        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.data)
        return model_new

    def forward(self, input):
        s0 = s1 = self.stem(input)
        for i, cell in enumerate(self.cells):
            if cell.reduction:
                weights = F.softmax(self.alphas_reduce, dim=-1)
                n = 3
                start = 2
                weights2 = F.softmax(self.betas_reduce[0:2], dim=-1)
                for i in range(self._steps - 1):
                    end = start + n
                    tw2 = F.softmax(self.betas_reduce[start:end], dim=-1)
                    start = end
                    n += 1
                    weights2 = torch.cat([weights2, tw2], dim=0)
            else:
                weights = F.softmax(self.alphas_normal, dim=-1)
                n = 3
                start = 2
                weights2 = F.softmax(self.betas_normal[0:2], dim=-1)
                for i in range(self._steps - 1):
                    end = start + n
                    tw2 = F.softmax(self.betas_normal[start:end], dim=-1)
                    start = end
                    n += 1
                    weights2 = torch.cat([weights2, tw2], dim=0)
            s0, s1 = s1, cell(s0, s1, weights, weights2)
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits

    def _loss(self, input, target):
        logits = self(input)
        return self._criterion(logits, target)

    def _initialize_alphas(self):
        k = sum(1 for i in range(self._steps) for n in range(2 + i))
        num_ops = len(PRIMITIVES)
        an = torch.randn(k, num_ops).cuda()*0.5
        ar = torch.randn(k, num_ops).cuda()*0.5
        an[:, 7] = 1
        ar[:, 7] = 1
        self.alphas_normal = Variable(1e-3 * an, requires_grad=True)
        self.alphas_reduce = Variable(1e-3 * ar, requires_grad=True)
        self.betas_normal = Variable(1e-3 * torch.randn(k).cuda(), requires_grad=True)
        self.betas_reduce = Variable(1e-3 * torch.randn(k).cuda(), requires_grad=True)
        self._arch_parameters = [
            self.alphas_normal,
            self.alphas_reduce,
            self.betas_normal,
            self.betas_reduce,
        ]

    def arch_parameters(self):
        return self._arch_parameters

    def genotype(self):

        def _parse(weights, weights2):
            gene = []
            n = 2
            start = 0

            for i in range(self._steps):
                end = start + n
                W = weights[start:end].copy()
                W2 = weights2[start:end].copy()
                for j in range(n):
                    W[j, :] = W[j, :] * W2[j]
                edges = sorted(range(i + 2),
                               key=lambda x: -max(W[x][k] for k in range(len(W[x])) if k != PRIMITIVES.index('none')))[
                        :2]

                for j in edges:
                    k_best = None
                    for k in range(len(W[j])):
                        if k != PRIMITIVES.index('none'):
                            if k_best is None or W[j][k] > W[j][k_best]:
                                k_best = k
                    gene.append((PRIMITIVES[k_best], j))
                start = end
                n += 1
            return gene

        n = 3
        start = 2
        weightsr2 = F.softmax(self.betas_reduce[0:2], dim=-1)
        weightsn2 = F.softmax(self.betas_normal[0:2], dim=-1)
        for i in range(self._steps - 1):
            end = start + n
            tw2 = F.softmax(self.betas_reduce[start:end], dim=-1)
            tn2 = F.softmax(self.betas_normal[start:end], dim=-1)
            start = end
            n += 1
            weightsr2 = torch.cat([weightsr2, tw2], dim=0)
            weightsn2 = torch.cat([weightsn2, tn2], dim=0)
        gene_normal = _parse(F.softmax(self.alphas_normal, dim=-1).data.cpu().numpy(), weightsn2.data.cpu().numpy())
        gene_reduce = _parse(F.softmax(self.alphas_reduce, dim=-1).data.cpu().numpy(), weightsr2.data.cpu().numpy())

        concat = range(2 + self._steps - self._multiplier, self._steps + 2)
        genotype = Genotype(
            normal=gene_normal, normal_concat=concat,
            reduce=gene_reduce, reduce_concat=concat
        )
        return genotype
