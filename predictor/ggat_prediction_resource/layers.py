import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# 定义多头注意力机制的GAT层
class GatedMultiHeadGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, gru_layer, gat_dropout, num_heads, merge='cat', require_gru=True):
        super(GatedMultiHeadGATLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.require_gru = require_gru
        self.gat_dropout = gat_dropout
        self.num_heads = num_heads

        if num_heads > 1:
            self.attentions = [GATLayer(in_dim, out_dim) for _ in range(num_heads)]
            for i, attention in enumerate(self.attentions):
                self.add_module('gat_attention_{}'.format(i), attention)

            self.gat_out_att = GATLayer(out_dim * num_heads, out_dim)
            self.add_module('gat_out_attention', self.gat_out_att)
        else:
            self.gat_out_att = GATLayer(in_dim, out_dim)
            self.add_module('gat_out_attention', self.gat_out_att)

        self.merge = merge  # 使用拼接的方法，否则取平均

        if in_dim != out_dim and require_gru:
            self.transform = nn.Linear(in_dim, out_dim)

        if require_gru:
            self.gru_layer = gru_layer

    def forward(self, g, h):
        gru_in = self.transform(h) if (self.in_dim != self.out_dim and self.require_gru) else h
        # x = F.dropout(h, self.gat_dropout, training=True)
        x = h

        if self.num_heads > 1:
            # 获取每套注意力机制得到的hi
            head_outs = [attn_head(g, x) for attn_head in self.attentions]
            if self.merge == 'cat':
                # 每套的hi拼接
                x = torch.cat(head_outs, dim=1)
            else:
                # 所有的hi对应元素求平均
                x = torch.mean(torch.stack(head_outs))

            # New added
            x = F.leaky_relu(x)

        x = F.leaky_relu(self.gat_out_att(g, x))

        if self.require_gru:
            x = self.gru_layer(x, gru_in)
            x = F.leaky_relu(x)

        return x


# 定义GAT神经层
class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(GATLayer, self).__init__()
        # 对应公式中1的 W，用于特征的线性变换
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        # 对应公式2中的 a, 输入拼接的zi和zj（2 * out_dim），输出eij（一个数值）
        self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)
        self.edge_fc = nn.Linear(1, 1, bias=False)
        self.m_fc = nn.Linear(1, 1, bias=False)
        # 初始化参数
        self.reset_parameters()

    def reset_parameters(self):
        # 随机初始化需要学习的参数
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.fc.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_fc.weight, gain=gain)
        nn.init.xavier_normal_(self.edge_fc.weight, gain=gain)
        nn.init.xavier_normal_(self.m_fc.weight, gain=gain)

    def edge_attention(self, edges):
        # 对应公式2中的拼接操作，即zi || zj
        z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)
        # 拼接之后对应公式2中激活函数里的计算操作，即a(zi || zj)
        a = self.attn_fc(z2)

        ev = self.edge_fc(edges.data['performance_degradation'])

        new_a = a * ev
        # 算出来的值经过leakyReLU激活得到eij,保存在e变量中
        return {'e': F.leaky_relu(new_a)}

    def message_func(self, edges):
        # 汇聚信息，传递之前计算好的z（对应节点的特征） 和 e
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        # 对应公式3，eij们经过softmax即可得到特征的权重αij
        new_e = self.m_fc(nodes.mailbox['e'])

        alpha = F.softmax(new_e, dim=1)
        # 计算出权重之后即可通过 权重αij * 变换后的特征zj 求和计算出节点更新后的特征
        # 不过激活函数并不在这里，代码后面有用到ELU激活函数
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'h': h}

    # 正向传播方式
    def forward(self, g, h):
        # 对应公式1，先转换特征
        z = self.fc(h)

        z = F.leaky_relu(z)
        # 将转换好的特征保存在z
        g.ndata['z'] = z
        # 对应公式2，得出e
        g.apply_edges(self.edge_attention)
        # 对应公式3、4计算出注意力权重α并且得出最后的hi
        g.update_all(self.message_func, self.reduce_func)
        # 返回并清除hi
        return g.ndata.pop('h')


class GRUSubLayer(nn.Module):
    def __init__(self, n_features):
        """
        GRU sub layer
        :param n_features: number of features
        """
        super(GRUSubLayer, self).__init__()
        self.n_features = n_features

        """GRU: Reset Gate"""
        self.reset_gate = nn.Sequential(
            nn.Linear(2 * n_features, n_features),
            # 激活函数为 sigmoid
            nn.Sigmoid()
        )
        """GRU: Update Gate"""
        self.update_gate = nn.Sequential(
            # 线性变换就相当于矩阵乘
            nn.Linear(2 * n_features, n_features),
            nn.Sigmoid()
        )
        """GRU: The output transform"""
        self.transform = nn.Sequential(
            nn.Linear(2 * n_features, n_features),
            nn.Tanh()
        )

    def forward(self, h, h_in):
        a = torch.cat((h, h_in), 1)

        r = self.reset_gate(a)
        z = self.update_gate(a)

        joined_input = torch.cat((h, r * h_in), 1)
        h_hat = self.transform(joined_input)

        output = (1 - z) * h_in + z * h_hat
        return output
