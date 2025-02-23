import torch.nn as nn
import torch.nn.functional as F
from layers import GatedMultiHeadGATLayer, GRUSubLayer, GATLayer


# 定义GAT模型
class GGAT_with_1_block(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout, num_heads):
        super(GGAT_with_1_block, self).__init__()
        self.gru_layer = GRUSubLayer(hidden_dim)

        self.layer1 = GatedMultiHeadGATLayer(in_dim, hidden_dim, self.gru_layer, dropout, num_heads, require_gru=False)
        self.layer2 = GatedMultiHeadGATLayer(hidden_dim, hidden_dim, self.gru_layer, dropout, 1)
        self.layer3 = nn.Linear(hidden_dim, out_dim, bias=True)

    def forward(self, g, h):
        h = self.layer1(g, h)
        h = self.layer2(g, h)
        h = self.layer3(h)
        h = F.sigmoid(h)
        return h


class GGAT_with_2_blocks(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout, num_heads):
        super(GGAT_with_2_blocks, self).__init__()
        self.gru_layer = GRUSubLayer(hidden_dim)

        self.layer1 = GatedMultiHeadGATLayer(in_dim, hidden_dim, self.gru_layer, dropout, num_heads, require_gru=False)
        self.layer2 = GatedMultiHeadGATLayer(hidden_dim, hidden_dim, self.gru_layer, dropout, num_heads)
        self.layer3 = GatedMultiHeadGATLayer(hidden_dim, hidden_dim, self.gru_layer, dropout, 1)
        self.layer4 = nn.Linear(hidden_dim, out_dim, bias=True)

    def forward(self, g, h):
        h = self.layer1(g, h)
        h = self.layer2(g, h)
        h = self.layer3(g, h)
        h = self.layer4(h)
        h = F.sigmoid(h)
        return h


class GGAT_with_3_blocks(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout, num_heads):
        super(GGAT_with_3_blocks, self).__init__()
        self.gru_layer = GRUSubLayer(hidden_dim)

        self.layer1 = GatedMultiHeadGATLayer(in_dim, hidden_dim, self.gru_layer, dropout, num_heads, require_gru=False)
        self.layer2 = GatedMultiHeadGATLayer(hidden_dim, hidden_dim, self.gru_layer, dropout, num_heads)
        self.layer3 = GatedMultiHeadGATLayer(hidden_dim, hidden_dim, self.gru_layer, dropout, num_heads)
        self.layer4 = GatedMultiHeadGATLayer(hidden_dim, hidden_dim, self.gru_layer, dropout, 1)
        self.layer5 = nn.Linear(hidden_dim, out_dim, bias=True)

    def forward(self, g, h):
        h = self.layer1(g, h)
        h = self.layer2(g, h)
        h = self.layer3(g, h)
        h = self.layer4(g, h)
        h = self.layer5(h)
        h = F.sigmoid(h)
        return h


class GGAT_with_4_blocks(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout, num_heads):
        super(GGAT_with_4_blocks, self).__init__()
        self.gru_layer = GRUSubLayer(hidden_dim)

        self.layer1 = GatedMultiHeadGATLayer(in_dim, hidden_dim, self.gru_layer, dropout, num_heads, require_gru=False)
        self.layer2 = GatedMultiHeadGATLayer(hidden_dim, hidden_dim, self.gru_layer, dropout, num_heads)
        self.layer3 = GatedMultiHeadGATLayer(hidden_dim, hidden_dim, self.gru_layer, dropout, num_heads)
        self.layer4 = GatedMultiHeadGATLayer(hidden_dim, hidden_dim, self.gru_layer, dropout, num_heads)
        self.layer5 = GatedMultiHeadGATLayer(hidden_dim, hidden_dim, self.gru_layer, dropout, 1)
        self.layer6 = nn.Linear(hidden_dim, out_dim, bias=True)

    def forward(self, g, h):
        h = self.layer1(g, h)
        h = self.layer2(g, h)
        h = self.layer3(g, h)
        h = self.layer4(g, h)
        h = self.layer5(g, h)
        h = self.layer6(h)
        h = F.sigmoid(h)
        return h



class GGAT_with_5_blocks(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout, num_heads):
        super(GGAT_with_5_blocks, self).__init__()
        self.gru_layer = GRUSubLayer(hidden_dim)

        self.layer1 = GatedMultiHeadGATLayer(in_dim, hidden_dim, self.gru_layer, dropout, num_heads, require_gru=False)
        self.layer2 = GatedMultiHeadGATLayer(hidden_dim, hidden_dim, self.gru_layer, dropout, num_heads)
        self.layer3 = GatedMultiHeadGATLayer(hidden_dim, hidden_dim, self.gru_layer, dropout, num_heads)
        self.layer4 = GatedMultiHeadGATLayer(hidden_dim, hidden_dim, self.gru_layer, dropout, num_heads)
        self.layer5 = GatedMultiHeadGATLayer(hidden_dim, hidden_dim, self.gru_layer, dropout, num_heads)
        self.layer6 = GatedMultiHeadGATLayer(hidden_dim, hidden_dim, self.gru_layer, dropout, 1)
        self.layer7 = nn.Linear(hidden_dim, out_dim, bias=True)

    def forward(self, g, h):
        h = self.layer1(g, h)
        h = self.layer2(g, h)
        h = self.layer3(g, h)
        h = self.layer4(g, h)
        h = self.layer5(g, h)
        h = self.layer6(g, h)
        h = self.layer7(h)
        h = F.sigmoid(h)
        return h