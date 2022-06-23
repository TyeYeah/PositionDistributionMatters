import torch.nn as nn
import torch.nn.functional as F

class SiameseNet(nn.Module):
    def __init__(self, embedding_net):
        super(SiameseNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2):
        # output1 = self.embedding_net(x1)
        # output2 = self.embedding_net(x2)
        output1,rec1, loss1 = self.embedding_net(x1)
        output2,rec2, loss2 = self.embedding_net(x2)
        return output1, output2, loss1+loss2

    def get_embedding(self, x):
        return self.embedding_net(x) # output, reconstruction, loss

class TripletNet(nn.Module):
    def __init__(self, embedding_net):
        super(TripletNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2, x3):
        # output1 = self.embedding_net(x1)
        # output2 = self.embedding_net(x2)
        # output3 = self.embedding_net(x3)
        output1,rec1, loss1 = self.embedding_net(x1)
        output2,rec2, loss2 = self.embedding_net(x2)
        output3,rec, loss3 = self.embedding_net(x3)
        return output1, output2, output3, loss1+loss2+loss3

    def get_embedding(self, x):
        return self.embedding_net(x) # output, reconstruction, loss
