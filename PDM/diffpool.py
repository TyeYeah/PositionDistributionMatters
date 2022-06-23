import torch
import torch.nn as nn
from torch.nn import functional as F

from torch_geometric.nn import GCNConv, SAGEConv
from utils_config import device

class DiffPool(nn.Module):

    def __init__(self, feature_size, output_dim, device=device, final_layer=False):
        super(DiffPool, self).__init__()
        self.device = device
        self.feature_size = feature_size
        self.output_dim = output_dim
        self.embed = SAGEConv(self.feature_size, self.feature_size)
        self.pool = SAGEConv(self.feature_size, self.output_dim)
        self.final_layer = final_layer

    def forward(self, x, a):
        z = self.embed(x, a)
        if self.final_layer:
            s = torch.ones(x.size(0), self.output_dim, device=self.device)
        else:
            s = F.softmax(self.pool(x, a), dim=1)
        x_new = s.t() @ z 
        
        # map to matrix [node,node]
        a_tmp = torch.zeros([x.shape[0],x.shape[0]]).to(device)
        for edg in a.t():
            a_tmp[edg[0],edg[1]] = 1
        a_tmp2 = s.t() @ a_tmp @ s
        # map back to [2,x]
        a_new = []
        for i in range(a_tmp2.shape[0]):
            for j in range(a_tmp2.shape[1]):
                # print(a_tmp[i][j])
                if a_tmp2[i][j]*10 >= 1:
                    a_new.append([i,j])
        a_new = torch.tensor(a_new).to(device).t()
        if a_new.shape[0] == 0:
            a_new = torch.tensor([[0,0]]).to(device).t()

        EPS = 1e-15
        link_loss = a_tmp - torch.matmul(s, s.t())
        link_loss = torch.norm(link_loss, p=2)
        link_loss = link_loss / a.numel()
        ent_loss = (-s * torch.log(s + EPS)).sum(dim=-1).mean()

        # print(a_tmp,a_tmp2, a_new)
        # print(x_new.shape, a_new.shape)
        # print(x.shape,x_new.shape,'==========')
        # print(a.shape,s.shape,z.shape,'==========')
        # input()

        # mapping phase
        x_new = torch.cat((x,x_new))
        a_new = torch.cat((a.t(),a_new.t()+x.shape[0])).t()
        
        return x_new, a_new, link_loss, ent_loss

