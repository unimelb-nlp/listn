import torch
from torch import nn

class MatFact(nn.Module):
    """ Matrix factorization + user & item bias, weight init., sigmoid_range """

    def __init__(self, N, M, D, K, T):
        super().__init__()
        self.U = nn.Parameter(torch.zeros(T, N, K))
        self.W = nn.Parameter(torch.zeros(T, D, K))
        self.V = nn.Parameter(torch.zeros(T, M, K))

        self.U_bias = nn.Parameter(torch.zeros(T, N, 1))
        self.V_bias = nn.Parameter(torch.zeros(T, M, 1))
        self.W_bias = nn.Parameter(torch.zeros(T, D, 1))

        self.U.data.uniform_(0., 0.05)
        self.W.data.uniform_(0., 0.05)
        self.V.data.uniform_(0., 0.05)

        self.cx_dict = nn.ParameterDict({'A':{'emb':self.V,'bias':self.V_bias},
                                         'C':{'emb':self.W,'bias':self.W_bias}})

    def l2(self, x):
        return torch.sum(torch.pow(x, 2))

    def weight_regularization(self):
        V_reg = self.l2(self.V)
        W_reg = self.l2(self.W)
        U_reg = self.l2(self.U)
        return V_reg + W_reg + U_reg

    def alignment_regularization(self):
        V_reg = self.l2(self.V[1:] - self.V[:-1])
        W_reg = self.l2(self.W[1:] - self.W[:-1])
        U_reg = self.l2(self.U[1:] - self.U[:-1])

        return V_reg + W_reg + U_reg

    def forward(self, t, user_ind, source, item_inds=None):
        cx_emb = self.cx_dict[source[0]]['emb'][t]
        cx_bias = self.cx_dict[source[0]]['bias'][t]

        if item_inds is not None:
            # cx_bias = cx_bias[item_inds]
            # cx_emb = cx_emb[item_inds]
            item_inds = item_inds.unsqueeze(-1)
            cx_bias = torch.gather(cx_bias, 1, item_inds)
            cx_emb = torch.gather(cx_emb, 1, item_inds.expand(-1, -1, cx_emb.size(2)))

        user_emb = self.U[t, user_ind].unsqueeze(1)
        user_bias = self.U_bias[t, user_ind].unsqueeze(1)

        element_product = torch.matmul(cx_emb,
                                       torch.permute(user_emb, [0, 2, 1]))
        element_product += user_bias + cx_bias

        return element_product.squeeze(-1)
