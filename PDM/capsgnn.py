"""CapsGNN Trainer."""

import glob,os,time,re
import json
import random
import torch
import numpy as np
from tqdm import tqdm, trange
from torch_geometric.nn import GCNConv
from utils_config import *
from layers import ListModule, PrimaryCapsuleLayer, Attention, SecondaryCapsuleLayer

from networks import SiameseNet, TripletNet
from losses import ContrastiveLoss, TripletLoss
from datasets import TripletDataset, DataLoader
from diffpool import DiffPool
from cal_dis import cosine

# torch.autograd.set_detect_anomaly(True) 

class CapsGNN(torch.nn.Module):
    """
    An implementation of themodel described in the following paper:
    https://openreview.net/forum?id=Byl8BnRcYm
    """
    def __init__(self, args, number_of_features, number_of_targets):
        super(CapsGNN, self).__init__()
        """
        :param args: Arguments object.
        :param number_of_features: Number of vertex features.
        :param number_of_targets: Number of classes.
        """
        self.args = args
        self.number_of_features = number_of_features
        self.number_of_targets = number_of_targets
        self._setup_layers()

    def _setup_base_layers(self):
        """
        Creating GCN layers.
        """
        self.base_layers = [GCNConv(self.number_of_features, self.args.gcn_filters)]
        # self.base_layers.append(GCNConv(self.args.gcn_filters, self.args.gcn_filters))
        self.base_layers.append(DiffPool(self.args.gcn_filters, self.args.diffpool_dim))
        for _ in range(self.args.gcn_layers-1):
            self.base_layers.append(GCNConv(self.args.gcn_filters, self.args.gcn_filters))
            self.base_layers.append(DiffPool(self.args.gcn_filters, self.args.diffpool_dim))
        self.base_layers = ListModule(*self.base_layers)

    def _setup_primary_capsules(self):
        """
        Creating primary capsules.
        """
        self.first_capsule = PrimaryCapsuleLayer(in_units=self.args.gcn_filters, # self.args.gcn_layers,
                                                 in_channels=self.args.gcn_layers, # self.args.gcn_filters,
                                                 num_units=self.args.gcn_layers,
                                                 capsule_dimensions=self.args.capsule_dimensions).to(device)

    def _setup_attention(self):
        """
        Creating attention layer.
        """
        self.attention = Attention(self.args.gcn_layers*self.args.capsule_dimensions,
                                   self.args.inner_attention_dimension).to(device)

    def _setup_graph_capsules(self):
        """
        Creating graph capsules.
        """
        self.graph_capsule = SecondaryCapsuleLayer(self.args.gcn_layers,
                                                   self.args.capsule_dimensions,
                                                   self.args.number_of_capsules,
                                                   self.args.capsule_dimensions).to(device)

    def _setup_class_capsule(self):
        """
        Creating class capsules.
        """
        self.class_capsule = SecondaryCapsuleLayer(self.args.capsule_dimensions,
                                                   self.args.number_of_capsules,
                                                #    self.number_of_targets,
                                                   self.args.number_of_capsules,
                                                   self.args.capsule_dimensions).to(device)

    def _setup_reconstruction_layers(self):
        """
        Creating histogram reconstruction layers.
        """
        self.reconstruction_layer_1 = torch.nn.Linear(self.args.number_of_capsules*self.args.capsule_dimensions,
                                                      int((self.number_of_features*2)/3)).to(device)

        self.reconstruction_layer_2 = torch.nn.Linear(int((self.number_of_features*2)/3),
                                                      int((self.number_of_features*3)/2)).to(device)

        self.reconstruction_layer_3 = torch.nn.Linear(int((self.number_of_features*3)/2),
                                                      self.number_of_features).to(device)

    def _setup_layers(self):
        """
        Creating layers of model.
        1. GCN layers.
        2. Primary capsules.
        3. Attention
        4. Graph capsules.
        5. Class capsules.
        6. Reconstruction layers.
        """
        self._setup_base_layers()
        self._setup_primary_capsules()
        self._setup_attention()
        self._setup_graph_capsules()
        self._setup_class_capsule()
        self._setup_reconstruction_layers()

    def calculate_reconstruction_loss(self, capsule_input, features):
        """
        Calculating the reconstruction loss of the model.
        :param capsule_input: Output of class capsule.
        :param features: Feature matrix.
        :return reconstrcution_loss: Loss of reconstruction.
        """
        # mask processing

        # v_mag = torch.sqrt((capsule_input**2).sum(dim=1)).to(device)
        # _, v_max_index = v_mag.max(dim=0)
        # v_max_index = v_max_index.data

        # capsule_masked = torch.autograd.Variable(torch.zeros(capsule_input.size()).to(device))
        # capsule_masked[v_max_index, :] = capsule_input[v_max_index, :]
        # capsule_masked = capsule_masked.view(1, -1)

        feature_counts = features.sum(dim=0)
        feature_counts = feature_counts/feature_counts.sum()

        reconstruction_output = torch.nn.functional.relu(self.reconstruction_layer_1(capsule_input.view(1, -1)))
        reconstruction_output = torch.nn.functional.relu(self.reconstruction_layer_2(reconstruction_output))
        reconstruction_output = torch.softmax(self.reconstruction_layer_3(reconstruction_output), dim=1)
        reconstruction_output = reconstruction_output.view(1, self.number_of_features)
        reconstruction_loss = torch.sum((features-reconstruction_output)**2)
        return reconstruction_output, reconstruction_loss

    def forward(self, data):
        """
        Forward propagation pass.
        :param data: Dictionary of tensors with features and edges.
        :return class_capsule_output: Class capsule outputs.
        """
        features = data["features"]
        edges = data["edges"]
        hidden_representations = []
        link_loss, ent_loss = 0, 0
        for layer in self.base_layers:
            if isinstance(layer, DiffPool):
                res = layer(features, edges)
                features, edges = torch.nn.functional.relu(res[0]), torch.nn.functional.relu(res[1])
                link_loss, ent_loss = res[2], res[3]
            else:
                features = torch.nn.functional.relu(layer(features, edges))
            hidden_representations.append(features)

        hidden_representations = torch.cat(tuple(hidden_representations))
        hidden_representations = hidden_representations.view(1, self.args.gcn_layers, self.args.gcn_filters, -1)
        # hidden_representations = hidden_representations.view(1, self.args.gcn_filters, self.args.gcn_layers, -1)
        first_capsule_output = self.first_capsule(hidden_representations)
        first_capsule_output = first_capsule_output.view(-1, self.args.gcn_layers*self.args.capsule_dimensions)
        rescaled_capsule_output = self.attention(first_capsule_output)
        rescaled_first_capsule_output = rescaled_capsule_output.view(-1, self.args.gcn_layers,
                                                                     self.args.capsule_dimensions)
        graph_capsule_output = self.graph_capsule(rescaled_first_capsule_output)
        reshaped_graph_capsule_output = graph_capsule_output.view(-1, self.args.capsule_dimensions,
                                                                  self.args.number_of_capsules)
        class_capsule_output = self.class_capsule(reshaped_graph_capsule_output)
        class_capsule_output = class_capsule_output.view(-1, self.args.number_of_capsules*self.args.capsule_dimensions)
        class_capsule_output = torch.mean(class_capsule_output, dim=0).view(1,
                                                                            self.args.number_of_capsules,
                                                                            self.args.capsule_dimensions)

        recon = class_capsule_output.view(self.args.number_of_capsules, self.args.capsule_dimensions)
        reconstruction_output, reconstruction_loss = self.calculate_reconstruction_loss(recon, data["features"])

        return class_capsule_output, reconstruction_output, link_loss + ent_loss + reconstruction_loss


class CapsGNNTrainer(object):
    """
    CapsGNN training and scoring.
    """
    def __init__(self, args):
        """
        :param args: Arguments object.
        """
        self.args = args
        self.setup_model()

    def enumerate_unique_labels_and_targets(self):
        """
        Enumerating the features and targets in order to setup weights later.
        """
        print("\nEnumerating feature and target values.\n")
        ending = "*.json"

        self.train_graph_paths = glob.glob(self.args.train_graph_folder+'*/'+ending)
        self.test_graph_paths = glob.glob(self.args.test_graph_folder+'*/'+ending)

        self.number_of_features = 64 # 128
        self.number_of_targets = 2

    def setup_model(self):
        """
        Enumerating labels and initializing a CapsGNN.
        """
        self.enumerate_unique_labels_and_targets()
        self.model = CapsGNN(self.args, self.number_of_features, self.number_of_targets).to(device)
        self.siamese = SiameseNet(self.model).to(device)
        self.triplet = TripletNet(self.model).to(device)

    def create_data_dictionary(self, target, edges, features):
        """
        Creating a data dictionary.
        :param target: Target vector.
        :param edges: Edge list tensor.
        :param features: Feature tensor.
        """
        to_pass_forward = dict()
        to_pass_forward["target"] = target
        to_pass_forward["edges"] = edges
        to_pass_forward["features"] = features
        return to_pass_forward

    def create_target(self, data):
        """
        Target createn based on data dicionary.
        :param data: Data dictionary.
        :return : Target vector.
        """
        return  torch.FloatTensor([0.0 if i != data["target"] else 1.0 for i in range(self.number_of_targets)])

    def create_edges(self, data):
        """
        Create an edge matrix.
        :param data: Data dictionary.
        :return : Edge matrix.
        """
        edges = [[edge[0], edge[1]] for edge in data["edges"]]
        edges = edges + [[edge[1], edge[0]] for edge in data["edges"]]
        return torch.t(torch.LongTensor(edges))

    def create_features(self, data):
        """
        Create feature matrix.
        :param data: Data dictionary.
        :return features: Matrix of features.
        """
        features = np.zeros((len(data["labels"]), self.number_of_features))
        # node_indices = [node for node in range(len(data["labels"]))]
        # feature_indices = [self.feature_map[label] for label in data["labels"].values()]
        # features[node_indices, feature_indices] = 1.0
        for fea_label in data["labels"]:
            for label_i in range(len(data["labels"][fea_label])):
                features[int(fea_label)][int(label_i)]=data["labels"][fea_label][int(label_i)]
        features = torch.FloatTensor(features)
        return features

    def create_input_data(self, path):
        """
        Creating tensors and a data dictionary with Torch tensors.
        :param path: path to the data JSON.
        :return to_pass_forward: Data dictionary.
        """
        data = json.loads(open(path).read().replace("\'",'\"').replace('\n',''))
        
        target = self.create_target(data).to(device)
        edges = self.create_edges(data).to(device)
        features = self.create_features(data).to(device)
        to_pass_forward = self.create_data_dictionary(target, edges, features)
        return to_pass_forward

    def fit_s(self):
        """
        Training a model on the training set.
        """
        print("\nTraining started.\n")
        # torch.autograd.set_detect_anomaly(True) 
        loss_func = ContrastiveLoss(1.).to(device)
        loss_func2 = TripletLoss(1.).to(device)
        self.siamese.train()
        optimizer = torch.optim.Adam(self.siamese.parameters(),
                                     lr=self.args.learning_rate,
                                     weight_decay=self.args.weight_decay)
        # prepare dataset
        datas = []
        for json_path in self.train_graph_paths:
            dict_data = self.create_input_data(json_path)
            datas.append([json_path,dict_data])
        dataset = TripletDataset(datas)
        batch_s = self.args.batch_size
        dataloader = DataLoader.DataLoader(dataset,batch_size= batch_s, shuffle = False, num_workers= 0)
        model_path = 'model'
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        for _ in tqdm(range(self.args.epochs), desc="Epochs: ", leave=True):
            tt = trange(int(dataset.__len__()/batch_s))
            for i in tt: # an iteration
                accumulated_losses = 0
                for j in range(batch_s):
                    data, data1, data2, path0, path1, path2 = dataset.__getitem__(i*batch_s+j)
                    outputs_1 = self.siamese(data, data1)
                    outputs_2 = self.siamese(data, data2)
                    output0 = outputs_1[0].view(-1)
                    outputO = outputs_2[0].view(-1)
                    output1 = outputs_1[1].view(-1)
                    output2 = outputs_2[1].view(-1)
                    losses = outputs_1[2]+outputs_2[2]
                    # print(path0,'\n', path1,'\n', path2)
                    # print(data,'\n', data1,'\n', data2,'+-+-+-+-+-+-+-+-+-')
                    # print(output0,'\n', output1,'\n', output2,'+++++++++++++++++++++++++++++')
                    # input()
                    loss1 = loss_func(output0,output1,torch.tensor(1.).to(device))
                    loss2 = loss_func(outputO,output2,torch.tensor(0.).to(device))
                    loss3 = loss_func2(output0,output1,output2,torch.tensor(1.).to(device))
                    if torch.isnan(loss1) or torch.isnan(loss2):
                        print(loss1, loss2)
                        print(path0, data)
                        print(path1, data1)
                        print(path2, data2)
                        # print(datas[i*batch_s+j][0])
                        input("=================================")
                    accumulated_losses = accumulated_losses + loss1 + loss2 + losses + loss3
                accumulated_losses = accumulated_losses/batch_s
                accumulated_losses.backward()
                optimizer.step()
                tt.set_description("Siamese CapsGNN (Loss=%g)" % round(accumulated_losses.item(), 4)) 
            torch.save(self.siamese,'model/model_siamese_'+str(_)+'.pt')
            pass

    def fit_t(self):
        """
        Training a model on the training set.
        """
        print("\nTraining started.\n")
        loss_func = TripletLoss(1.).to(device)
        self.triplet.train()
        optimizer = torch.optim.Adam(self.triplet.parameters(),
                                     lr=self.args.learning_rate,
                                     weight_decay=self.args.weight_decay)
        # prepare dataset
        datas = []
        for json_path in self.train_graph_paths:
            dict_data = self.create_input_data(json_path)
            datas.append([json_path,dict_data])
        dataset = TripletDataset(datas)
        batch_s = self.args.batch_size
        dataloader = DataLoader.DataLoader(dataset,batch_size= batch_s, shuffle = False, num_workers= 0)
        model_path = 'model'
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        for _ in tqdm(range(self.args.epochs), desc="Epochs: ", leave=True):
            tt = trange(int(dataset.__len__()/batch_s))
            for i in tt: # an iteration
                accumulated_losses = 0
                for j in range(batch_s):
                    data, data1, data2, path0, path1, path2 = dataset.__getitem__(i*batch_s+j)
                    output = self.triplet(data, data1, data2)
                    output0 = output[0].view(-1)
                    output1 = output[1].view(-1)
                    output2 = output[2].view(-1)
                    losses = output[3]
                    loss = loss_func(output0,output1,output2,torch.tensor(1.).to(device))
                    if torch.isnan(loss) :
                        print(loss)
                        print(path0, data)
                        print(path1, data1)
                        print(path2, data2)
                        # print(datas[i*batch_s+j][0])
                        input("=================================")
                    accumulated_losses = accumulated_losses + loss + losses
                accumulated_losses = accumulated_losses/batch_s
                accumulated_losses.backward()
                optimizer.step()
                tt.set_description("Triplet CapsGNN (Loss=%g)" % round(accumulated_losses.item(), 4)) 
            torch.save(self.triplet,'model/model_triplet_'+str(_)+'.pt')
            pass

    def score_s(self):
        """
        Scoring on the test set.
        """
        print("\n\nScoring.\n")
        self.siamese.eval()
        # output_file = open('pecker_siamese_score_res.log','a+')

        # for roc drawing
        labels = []
        predicts = []

        test_sum = 1
        test_precision10 = 1
        test_precision50 = 1
        test_recall = 1
        mrr_value = 0
        
        XO_res = [];XC_res = [];XA_res = [];

        while test_sum in trange(128):
            bin1, bin2 = random.sample(glob.glob(self.args.test_graph_folder+'*/'),2)
            if bin1 != bin2 and (('libcrypto' in bin1 and 'libcrypto' in bin2) or ('libssl' in bin1 and 'libssl' in bin2) or ('busybox' in bin1 and 'busybox' in bin2)): # same program
                path1s = glob.glob(bin1+'*')
                path2s = glob.glob(bin2+'*')
                func1 = random.sample(path1s,1)[0]
                if not os.path.exists('/'.join(path2s[0].split('/')[:-1])+'/'+func1.split('/')[-1]): # have a same name function
                    continue
                temp_res = []
                func2_num = 0
                func2_flag = 0
                # for func2 in tqdm(path2s):
                for func2 in path2s:
                    if func2_num >= 96:
                        func2 = '/'.join(path2s[0].split('/')[:-1])+'/'+func1.split('/')[-1]
                        func2_flag = 1
                    data1 = self.create_input_data(func1)
                    data2 = self.create_input_data(func2)
                    out = self.siamese(data1, data2)
                    out1, out2 = out[0].view(-1).cpu().detach().numpy(), out[1].view(-1).cpu().detach().numpy()
                    cos_sim = 1-cosine(out1,out2)
                    temp_res.append((cos_sim,func1.split('/')[-1],func2.split('/')[-1]))
                    func2_num+=1
                    if func1.split('/')[-1] == func2.split('/')[-1]:
                        func2_flag = 1
                    if func2_flag == 1:
                        break
                            
                temp_res.sort(key=lambda u:(-u[0]))
                
                for i in range(len(temp_res)):
                    # print(temp_res[i],i,'==============',bin1,bin2,file=output_file)
                    if temp_res[i][1]==temp_res[i][2]:
                        if i <=50 :
                            test_precision50+=1
                        if i <=10 :
                            test_precision10+=1
                        if i<=1 :
                            test_recall+=1
                            i=1 # for mrr calculation
                        mrr_value+=1/i
                        if 'clang' in bin1 and 'gcc' in bin2 or 'clang' in bin2 and 'gcc' in bin1:
                            XC_res.append(i)
                        if re.findall('O.',bin1)[0] != re.findall('O.',bin2)[0]:
                            XO_res.append(i)
                        if 'x86_64' in bin1 and 'i386' in bin2 or 'x86_64' in bin2 and 'i386' in bin1:
                            XA_res.append(i)
                        break
                # print('--------------------------------------'+time.asctime(time.localtime(time.time()))+'----------------------------------',file=output_file)
                test_sum+=1
                print("\nPrecision: " , test_precision10/test_sum, test_precision50/test_sum," in Top 10: ",test_precision10," in Top 50: ",test_precision50," Summary: ",test_sum)
                print("Recall: " , test_recall/test_sum," at Top 1: ",test_recall,"MRR: " , mrr_value/test_sum," value: ",mrr_value," Summary: ",test_sum)

            else: # different program, or same binary
                pass # continue, reloop
        # print('=================================='+time.asctime(time.localtime(time.time()))+'======================================',file=output_file)
        # output_file.close()

        test_cmp_num = 1
        test_cmp_true = 1
        # prepare dataset
        datas = []
        for json_path in self.test_graph_paths:
            dict_data = self.create_input_data(json_path)
            datas.append([json_path,dict_data])
        dataset = TripletDataset(datas)
        batch_s = self.args.batch_size
        model_path = 'model'
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        for _ in tqdm(range(3), desc="Turns: ", leave=True):
            tt = trange(int(dataset.__len__()/batch_s))
            for i in tt: # an iteration
                if test_cmp_num>=512:
                    break
                for j in range(batch_s):
                    data, data1, data2, path0, path1, path2 = dataset.__getitem__(i*batch_s+j)
                    outputs_1 = self.siamese(data, data1)
                    outputs_2 = self.siamese(data, data2)
                    output0 = outputs_1[0].view(-1).cpu().detach().numpy()
                    outputO = outputs_2[0].view(-1).cpu().detach().numpy()
                    output1 = outputs_1[1].view(-1).cpu().detach().numpy()
                    output2 = outputs_2[1].view(-1).cpu().detach().numpy()
                    similarity1 = 1-cosine(output0, output1)
                    similarity2 = 1-cosine(outputO, output2)

                    labels.append(1);labels.append(0)
                    predicts.append(similarity1);predicts.append(similarity2)
                    labels.append(1)

                    test_cmp_num+=1
                    
                    if similarity1>similarity2:
                        test_cmp_true+=1
                        predicts.append(1)
                    else: 
                        predicts.append(0)

                    if test_cmp_num>=512:
                        break
                tt.set_description("Siamese CapsGNN (CMP Acc=%g, True:%d, Sum:%d)" % (round(test_cmp_true/test_cmp_num, 4), test_cmp_true, test_cmp_num)) 
            pass

        time_stamp = str(time.time())
        statistics_file = "./statistics/"
        if not os.path.exists(statistics_file) :
            os.makedirs(statistics_file)
        np.save(statistics_file+'labels_s_'+time_stamp+'.npy',labels)
        np.save(statistics_file+'predicts_s_'+time_stamp+'.npy',predicts)
        print("CMP result saved.")

        print("\nComparing Accuracy: " , test_cmp_true/test_cmp_num,"True cmp:",test_cmp_true,"Summary:",test_cmp_num)

        os.system('echo "'+time.asctime(time.localtime(time.time()))+'" >> pecker_evaluate_siamese.log')
        os.system('echo "Precision:'+str(test_precision10/test_sum)+';'+str(test_precision50/test_sum)+' Recall:'+str(test_recall/test_sum)+' MRR:'+str(mrr_value/test_sum)+' CMP Acc:'+str(test_cmp_true/test_cmp_num)+'" >> pecker_evaluate_siamese.log')
        os.system('echo "Precision num:'+str(test_precision10)+';'+str(test_precision50)+' Recall num:'+str(test_recall)+' MRR num:'+str(mrr_value)+' Summary num:'+str(test_sum)+' CMP Acc num:'+str(test_cmp_true)+' CMP Sum num:'+str(test_cmp_num)+'" >> pecker_evaluate_siamese.log')
        XO_mrr, XO_recall = calmrr(XO_res);XC_mrr, XC_recall = calmrr(XC_res);XA_mrr, XA_recall = calmrr(XA_res)
        os.system('echo "XO MRR:'+str(XO_mrr)+', Recall:'+str(XO_recall)+' XC MRR:'+str(XC_mrr)+', Recall:'+str(XC_recall)+' XA MRR:'+str(XA_mrr)+', Recall:'+str(XA_recall)+'" >> pecker_evaluate_siamese.log')
        
    def score_t(self):
        """
        Scoring on the test set.
        """
        print("\n\nScoring.\n")
        self.triplet.eval()
        # output_file = open('pecker_triplet_score_res.log','a+')

        # for roc drawing
        labels = []
        predicts = []

        test_sum = 1
        test_precision10 = 1
        test_precision50 = 1
        test_recall = 1
        mrr_value = 0
        
        XO_res = [];XC_res = [];XA_res = [];

        while test_sum in trange(128):
            bin1, bin2 = random.sample(glob.glob(self.args.test_graph_folder+'*/'),2)
            if bin1 != bin2 and (('libcrypto' in bin1 and 'libcrypto' in bin2) or ('libssl' in bin1 and 'libssl' in bin2) or ('busybox' in bin1 and 'busybox' in bin2)): # same program
                path1s = glob.glob(bin1+'*')
                path2s = glob.glob(bin2+'*')
                func1 = random.sample(path1s,1)[0]
                if not os.path.exists('/'.join(path2s[0].split('/')[:-1])+'/'+func1.split('/')[-1]): # have a same name function
                    continue
                temp_res = []
                func2_num = 0
                func2_flag = 0
                # for func2 in tqdm(path2s):
                for func2 in path2s:
                    if func2_num >= 96:
                        func2 = '/'.join(path2s[0].split('/')[:-1])+'/'+func1.split('/')[-1]
                        func2_flag = 1
                    data1 = self.create_input_data(func1)
                    data2 = self.create_input_data(func2)
                    out = self.triplet(data1,data1,data2)
                    out1, out2 = out[1].view(-1).cpu().detach().numpy(), out[2].view(-1).cpu().detach().numpy()
                    cos_sim = 1-cosine(out1,out2)
                    temp_res.append((cos_sim,func1.split('/')[-1],func2.split('/')[-1]))
                    func2_num+=1
                    if func1.split('/')[-1] == func2.split('/')[-1]:
                        func2_flag = 1
                    if func2_flag == 1:
                        break
                            
                temp_res.sort(key=lambda u:(-u[0]))
                
                for i in range(len(temp_res)):
                    # print(temp_res[i],i,'==============',bin1,bin2,file=output_file)
                    if temp_res[i][1]==temp_res[i][2]:
                        if i <=50 :
                            test_precision50+=1
                        if i <=10 :
                            test_precision10+=1
                        if i<=1 :
                            test_recall+=1
                            i=1 # for mrr calculation
                        mrr_value+=1/i
                        if 'clang' in bin1 and 'gcc' in bin2 or 'clang' in bin2 and 'gcc' in bin1:
                            XC_res.append(i)
                        if re.findall('O.',bin1)[0] != re.findall('O.',bin2)[0]:
                            XO_res.append(i)
                        if 'x86_64' in bin1 and 'i386' in bin2 or 'x86_64' in bin2 and 'i386' in bin1:
                            XA_res.append(i)
                        break
                # print('-----------------------------------'+time.asctime(time.localtime(time.time()))+'-------------------------------------',file=output_file)
                test_sum+=1
                print("\nPrecision: " , test_precision10/test_sum, test_precision50/test_sum," in Top 10: ",test_precision10," in Top 50: ",test_precision50," Summary: ",test_sum)
                print("Recall: " , test_recall/test_sum," at Top 1: ",test_recall,"MRR: " , mrr_value/test_sum," value: ",mrr_value," Summary: ",test_sum)

            else: # different program, or same binary
                pass # continue, reloop
        # print('================================'+time.asctime(time.localtime(time.time()))+'========================================',file=output_file)
        # output_file.close()

        test_cmp_num = 1
        test_cmp_true = 1
        # prepare dataset
        datas = []
        for json_path in self.test_graph_paths:
            dict_data = self.create_input_data(json_path)
            datas.append([json_path,dict_data])
        dataset = TripletDataset(datas)
        batch_s = self.args.batch_size
        model_path = 'model'
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        for _ in tqdm(range(3), desc="Turns: ", leave=True):
            tt = trange(int(dataset.__len__()/batch_s))
            for i in tt: # an iteration
                if test_cmp_num>=512:
                    break
                for j in range(batch_s):
                    data, data1, data2, path0, path1, path2 = dataset.__getitem__(i*batch_s+j)
                    output = self.triplet(data, data1, data2)
                    output0 = output[0].view(-1).cpu().detach().numpy()
                    output1 = output[1].view(-1).cpu().detach().numpy()
                    output2 = output[2].view(-1).cpu().detach().numpy()
                    similarity1 = 1-cosine(output0, output1)
                    similarity2 = 1-cosine(output0, output2)
                    
                    labels.append(1);labels.append(0)
                    predicts.append(similarity1);predicts.append(similarity2)
                    labels.append(1)

                    test_cmp_num+=1
                    
                    if similarity1>similarity2:
                        test_cmp_true+=1
                        predicts.append(1)
                    else: 
                        predicts.append(0)

                    if test_cmp_num>=512:
                        break
                tt.set_description("Triplet CapsGNN (CMP Acc=%g, True:%d, Sum:%d)" % (round(test_cmp_true/test_cmp_num, 4), test_cmp_true, test_cmp_num)) 
            pass

        time_stamp = str(time.time())
        statistics_file = "./statistics/"
        if not os.path.exists(statistics_file) :
            os.makedirs(statistics_file)
        np.save(statistics_file+'labels_t_'+time_stamp+'.npy',labels)
        np.save(statistics_file+'predicts_t_'+time_stamp+'.npy',predicts)
        print("CMP result saved.")

        print("\nComparing Accuracy: " , test_cmp_true/test_cmp_num,"True cmp:",test_cmp_true,"Summary:",test_cmp_num)
        
        os.system('echo "'+time.asctime(time.localtime(time.time()))+'" >> pecker_evaluate_triplet.log')
        os.system('echo "Precision:'+str(test_precision10/test_sum)+';'+str(test_precision50/test_sum)+' Recall:'+str(test_recall/test_sum)+' MRR:'+str(mrr_value/test_sum)+' CMP Acc:'+str(test_cmp_true/test_cmp_num)+'" >> pecker_evaluate_triplet.log')
        os.system('echo "Precision num:'+str(test_precision10)+';'+str(test_precision50)+' Recall num:'+str(test_recall)+' Summary num:'+str(test_sum)+' MRR num:'+str(mrr_value)+' Summary num:'+str(test_sum)+' CMP Acc num:'+str(test_cmp_true)+' CMP Sum num:'+str(test_cmp_num)+'" >> pecker_evaluate_triplet.log')
        XO_mrr, XO_recall = calmrr(XO_res);XC_mrr, XC_recall = calmrr(XC_res);XA_mrr, XA_recall = calmrr(XA_res)
        os.system('echo "XO MRR:'+str(XO_mrr)+', Recall:'+str(XO_recall)+' XC MRR:'+str(XC_mrr)+', Recall:'+str(XC_recall)+' XA MRR:'+str(XA_mrr)+', Recall:'+str(XA_recall)+'" >> pecker_evaluate_triplet.log')
        
    def vul_rank_s(self):
        """
        Ranking on the vulnerability set.
        """
        print("\n\nRanking.\n")
        self.siamese.eval()
        # output_file = open('pecker_siamese_rank_res.log','a+')

        cve_path = './input/cve/'
        cve_paths = glob.glob(cve_path+'*/*')
        test_path = './input/test_cve/'
        test_paths = glob.glob(test_path+'*/')

        cve_funcs = set([i.split('/')[-1] for i in cve_paths])
        candidate = []
        for test_bin in test_paths:
            candidate+=random.sample(glob.glob(test_bin+'*'),50)
            for cve_func in cve_funcs:
                if os.path.exists(test_bin+cve_func):
                    candidate.append(test_bin+cve_func)

        temp_res = []
        for cve in tqdm(cve_paths):
            for can in tqdm(candidate):
                data1 = self.create_input_data(cve)
                data2 = self.create_input_data(can)
                out = self.siamese(data1, data2)
                out1, out2 = out[0].view(-1).cpu().detach().numpy(), out[1].view(-1).cpu().detach().numpy()
                cos_sim = 1-cosine(out1,out2)
                temp_res.append((cos_sim,cve,can))
        temp_res.sort(key=lambda u:(-u[0]))

        top5 = 0;top10 = 0;top20 = 0;top50 = 0;top100 = 0;top200 = 0;
        for i in range(len(temp_res)):
            if temp_res[i][1].split('/')[-1]==temp_res[i][2].split('/')[-1]:
                print(temp_res[i][0],i,'==============\n',temp_res[i][1],'\n',temp_res[i][2])
                if i <=5 :
                    top5+=1
                if i <=10 :
                    top10+=1
                if i<=20 :
                    top20+=1
                if i <=50 :
                    top50+=1
                if i <=100 :
                    top100+=1
                if i <=200 :
                    top200+=1
            # print(temp_res[i][0],i,'==============\n',temp_res[i][1],'\n',temp_res[i][2],file=output_file)
        # print('---------------------------------'+time.asctime(time.localtime(time.time()))+'---------------------------------------',file=output_file)
        # output_file.close()
        output_file2 = open('pecker_evaluate_siamese.log','a+')
        print("===== Result =====")
        print("Top-5 num:",top5)
        print("Top-10 num:",top10)
        print("Top-20 num:",top20)
        print("Top-50 num:",top50)
        print("Top-100 num:",top100)
        print("Top-200 num:",top200)
        # and store in file
        print("Top-5 num:",top5,file=output_file2)
        print("Top-10 num:",top10,file=output_file2)
        print("Top-20 num:",top20,file=output_file2)
        print("Top-50 num:",top50,file=output_file2)
        print("Top-100 num:",top100,file=output_file2)
        print("Top-200 num:",top200,file=output_file2)
        output_file2.close()
    
    def vul_rank_t(self):
        """
        Ranking on the vulnerability set.
        """
        print("\n\nRanking.\n")
        self.triplet.eval()
        # output_file = open('pecker_triplet_rank_res.log','a+')

        cve_path = './input/cve/'
        cve_paths = glob.glob(cve_path+'*/*')
        test_path = './input/test_cve/'
        test_paths = glob.glob(test_path+'*/')

        cve_funcs = set([i.split('/')[-1] for i in cve_paths])
        candidate = []
        for test_bin in test_paths:
            candidate+=random.sample(glob.glob(test_bin+'*'),50)
            for cve_func in cve_funcs:
                if os.path.exists(test_bin+cve_func):
                    candidate.append(test_bin+cve_func)

        temp_res = []
        for cve in tqdm(cve_paths):
            for can in tqdm(candidate):
                data1 = self.create_input_data(cve)
                data2 = self.create_input_data(can)
                out = self.triplet(data1, data2, data2)
                out1, out2 = out[0].view(-1).cpu().detach().numpy(), out[1].view(-1).cpu().detach().numpy()
                cos_sim = 1-cosine(out1,out2)
                temp_res.append((cos_sim,cve,can))
        temp_res.sort(key=lambda u:(-u[0]))

        top5 = 0;top10 = 0;top20 = 0;top50 = 0;top100 = 0;top200 = 0;
        for i in range(len(temp_res)):
            if temp_res[i][1].split('/')[-1]==temp_res[i][2].split('/')[-1]:
                print(temp_res[i][0],i,'==============\n',temp_res[i][1],'\n',temp_res[i][2])
                if i <=5 :
                    top5+=1
                if i <=10 :
                    top10+=1
                if i<=20 :
                    top20+=1
                if i <=50 :
                    top50+=1
                if i <=100 :
                    top100+=1
                if i <=200 :
                    top200+=1
            # print(temp_res[i][0],i,'==============\n',temp_res[i][1],'\n',temp_res[i][2],file=output_file)
        # print('---------------------------------'+time.asctime(time.localtime(time.time()))+'---------------------------------------',file=output_file)
        # output_file.close()
        output_file2 = open('pecker_evaluate_triplet.log','a+')
        print("===== Result =====")
        print("Top-5 num:",top5)
        print("Top-10 num:",top10)
        print("Top-20 num:",top20)
        print("Top-50 num:",top50)
        print("Top-100 num:",top100)
        print("Top-200 num:",top200)
        # and store in file
        print("Top-5 num:",top5,file=output_file2)
        print("Top-10 num:",top10,file=output_file2)
        print("Top-20 num:",top20,file=output_file2)
        print("Top-50 num:",top50,file=output_file2)
        print("Top-100 num:",top100,file=output_file2)
        print("Top-200 num:",top200,file=output_file2)
        output_file2.close()
