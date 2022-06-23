import random
import torch
import torch.utils.data.dataset as Dataset
import torch.utils.data.dataloader as DataLoader
import numpy as np
 
from utils_config import device
import Levenshtein, glob

class TripletDataset(Dataset.Dataset):
    def __init__(self, Data):
        self.json_path = []
        self.dict_data = []
        self.max_edge_len = 0
        for data in Data:
            self.json_path.append(data[0])
            self.dict_data.append(data[1])
            if len(data[1]['edges']) > self.max_edge_len:
                self.max_edge_len = len(data[1]['edges'])
    def __len__(self):
        return len(self.json_path) # self.dict_data
    def __getitem__(self, index):
        data = self.dict_data[index]
        split_path = self.json_path[index].split('/')
        func_name = split_path[-1]
        bin_name = split_path[-2]

        data_path = self.json_path[index]
        pos_path = ''
        neg_path = ''

        # create positive/negative samples
        sample_num = 0
        indflag = [0,0]
        while True:
            index1 = random.sample(range(self.__len__()),1)[0]
            data1 = self.dict_data[index1]
            split_path1 = self.json_path[index1].split('/')
            func_name1 = split_path1[-1]
            bin_name1 = split_path1[-2]
            if ('libcrypto' in bin_name and 'libcrypto' in bin_name1) or ('libssl' in bin_name and 'libssl' in bin_name1) or ('busybox' in bin_name and 'busybox' in bin_name1):
                if func_name1 == func_name:
                    posdata = data1
                    pos_path = self.json_path[index1]
                    indflag[0]=1
                elif self.json_path[index1].replace(func_name1,func_name) in self.json_path:
                    posdata = self.dict_data[self.json_path.index(self.json_path[index1].replace(func_name1,func_name))]
                    pos_path = self.json_path[index1].replace(func_name1,func_name)
                    indflag[0]=1
                if func_name1 != func_name and indflag[1]==0:
                    negdata = data1
                    neg_path = self.json_path[index1]
                    indflag[1]=1
            else: # if different source/program
                negdata = data1
                neg_path = self.json_path[index1]
                indflag[1]=1
            if indflag == [1,1]: # 
                break
        # print(data_path, pos_path, neg_path,'++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        return data, posdata, negdata, data_path, pos_path, neg_path


if __name__ == '__main__':
    ps = glob.glob('input/train/*')
    print(ps)