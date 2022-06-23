import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from cal_dis import cosine

class ContrastiveLoss(nn.Module):
    """
    Contrastive loss
    Takes embeddings of two samples and a target label == 1 if samples are from the same class and label == 0 otherwise
    """

    def __init__(self, margin):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.eps = 1e-9

    def forward(self, output1, output2, target, size_average=True):
        log1 = -torch.log10(abs(output1[0])).round()
        output1 = output1*10**log1
        log2 = -torch.log10(abs(output2[0])).round()
        output2 = output2*10**log2

        distances = torch.cosine_similarity(output1,output2,dim=0)
        losses = torch.mean((1-target) * torch.pow(distances, 2) + 
                                      (target) * torch.pow(torch.clamp(self.margin - distances, min=0.0), 2)) + 0.001
        # print(target, distances, losses,'--------------------') 
        if torch.isnan(losses):
            print(output1,output2)
            print(distances, target)
            print((1-target) * torch.pow(distances, 2), (target) * torch.pow(torch.clamp(self.margin - distances, min=0.0), 2))
            print(losses,losses.mean())
            # print("set losses as 0.")
            # losses = torch.tensor(0.)
            # losses.requires_grad_(True)
            print("------------------------------")
            
        return losses.mean() if size_average else losses.sum()
    


class TripletLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative, size_average=True):
        log1 = -torch.log10(abs(anchor[0])).round()
        anchor = anchor*10**log1
        log2 = -torch.log10(abs(positive[0])).round()
        positive = positive*10**log2
        log3 = -torch.log10(abs(negative[0])).round()
        negative = negative*10**log3

        distance_positive = torch.cosine_similarity(anchor, positive, dim=0)
        distance_negative = torch.cosine_similarity(anchor, negative, dim=0)
        losses = F.relu(- distance_positive + distance_negative + self.margin)+self.margin*0.001

        if torch.isnan(losses):
            print(anchor, positive, negative)
            print(distance_positive, distance_negative)
            print("------------------------------")

        return losses.mean() if size_average else losses.sum()

