import torch
import torch.nn as nn
import torch.nn.functional as F



class FastGlobalAvgPool2d(nn.Module):
    def __init__(self, flatten=False):
        super(FastGlobalAvgPool2d, self).__init__()
        self.flatten = flatten

    def forward(self, x):
        if self.flatten:
            in_size = x.size()
            return x.view((in_size[0], in_size[1], -1)).mean(dim=2)
        else:
            return x.view(x.size(0), x.size(1), -1).mean(-1).view(x.size(0), x.size(1), 1, 1)


class FastGlobalAvgMaxPool2d(nn.Module):
    def __init__(self, flatten=False):
        super(FastGlobalAvgMaxPool2d, self).__init__()
        self.flatten = flatten

    def forward(self, x):
        in_size = x.size()
        x = x.view((in_size[0], in_size[1], -1))
        x = (x.mean(dim=2).add_(x.max(dim=2))).mul_(0.5)
        if self.flatten:
            return x
        else:
            return x.view(x.size(0), x.size(1), 1, 1)


def adaptive_avgmax_pool2d(x, output_size=1):
    x_avg = F.adaptive_avg_pool2d(x, output_size)
    x_max = F.adaptive_max_pool2d(x, output_size)
    return 0.5 * (x_avg + x_max)


class TestTimePoolHead(nn.Module):
    def __init__(self, model, original_pool=7):
        super(TestTimePoolHead, self).__init__()
        self.model = model
        self.original_pool = original_pool
        # base_fc = self.model.head[0]
        # self.avgmax_pool2d = FastGlobalAvgMaxPool2d()

        self.fc = nn.Conv2d(
            self.model.num_features, self.model.num_classes, kernel_size=1, bias=True)
        self.fc.weight.data.copy_(self.model.head[0].weight.data.view(self.fc.weight.size()))
        self.fc.bias.data.copy_(self.model.head[0].bias.data.view(self.fc.bias.size()))

        # delete original fc layer
        del self.model.head
        del self.model.self.global_pool

    def forward(self, x):
        x = self.model.body(x)
        x = F.avg_pool2d(x, kernel_size=self.original_pool, stride=1)
        x = self.fc(x)
        x = adaptive_avgmax_pool2d(x, 1)
        return x.view(x.size(0), -1)
