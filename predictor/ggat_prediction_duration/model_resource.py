import torch.nn as nn
import torch.nn.functional as F
from layers import GatedMultiHeadGATLayer, GRUSubLayer


# 定义GAT模型
class GGAT_RES(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout, num_heads):
        super(GGAT_RES, self).__init__()
        self.gru_layer = GRUSubLayer(hidden_dim)
        #self.gru_layer_for_layer2 = GRUSubLayer(hidden_dim // 2)
        #self.gru_layer_for_layer3 = GRUSubLayer(hidden_dim // 4)

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
