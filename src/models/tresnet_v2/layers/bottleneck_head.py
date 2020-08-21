import torch
import torch.nn as nn
import torch.nn.functional as F


################ palitao FCC layer  ################
class GroupFC(nn.Module):
    def __init__(self, C_prev, num_classes, num_groups, remove_bias=False):
        super(GroupFC, self).__init__()
        assert C_prev % num_groups == 0  # why make our life difficult?
        self.num_groups = num_groups
        len_group = C_prev // self.num_groups
        len_classes = num_classes // self.num_groups
        self.classifier_group = nn.ModuleList(
            [nn.Linear(len_group, len_classes) for i in range(self.num_groups - 1)])
        self.classifier_group.append(
            nn.Linear(C_prev - len_group * (self.num_groups - 1),
                      num_classes - len_classes * (self.num_groups - 1),
                      bias=not remove_bias))

    def forward(self, x):
        chunks = torch.chunk(x, self.num_groups, dim=1)
        logits = torch.cat([op(g) for op, g in zip(self.classifier_group, chunks)], dim=1)
        return logits


class fcc_bottleneck(nn.Module):
    def __init__(self, C_prev, reduce_factor=4):
        super(fcc_bottleneck, self).__init__()

        self.bottleneck_fcc = nn.Sequential(
            nn.Linear(C_prev, C_prev // reduce_factor, bias=False),
            nn.BatchNorm1d(C_prev // reduce_factor),
            nn.ReLU(inplace=True),
            nn.Linear(C_prev // reduce_factor, C_prev, bias=False))

    def forward(self, x):
        return x + self.bottleneck_fcc(x)


class bottleneck_head(nn.Module):
    def __init__(self, num_features, num_classes, bottleneck_features=200):
        super(bottleneck_head, self).__init__()
        self.embedding_generator = nn.ModuleList()
        self.embedding_generator.append(nn.Linear(num_features, bottleneck_features))
        self.embedding_generator = nn.Sequential(*self.embedding_generator)
        self.FC = nn.Linear(bottleneck_features, num_classes)

    def forward(self, x):
        self.embedding = self.embedding_generator(x)
        logits = self.FC(self.embedding)
        return logits
################################################################
