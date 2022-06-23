"""Data reading and printing utils."""

from sklearn import manifold
import numpy as np
from matplotlib import colors, markers
import matplotlib.pyplot as plt
from sklearn.metrics import auc, roc_curve
from texttable import Texttable
import torch,os

if torch.cuda.is_available:
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("CUDA ON !!!!!!")
else:
    device=torch.device("cpu")
    print("CPU RUN !!!!!!")

# device=torch.device("cpu")
# device=torch.device("cuda:1")

def calmrr(ll):
    llen = len(ll)
    res = 0
    rec=0
    for lll in ll:
        res+=1/lll
        if lll ==1:
            rec+=1
    return res/llen,rec/llen

def tab_printer(args):
    """
    Function to print the logs in a nice tabular format.
    :param args: Parameters used for the model.
    """
    args = vars(args)
    keys = sorted(args.keys())
    t = Texttable() 
    t.add_rows([["Parameter", "Value"]])
    t.add_rows([[k.replace("_", " ").capitalize(), args[k]] for k in keys])
    print(t.draw())

def create_numeric_mapping(node_properties):
    """
    Create node feature map.
    :param node_properties: List of features sorted.
    :return : Feature numeric map.
    """
    return {value:i for i, value in enumerate(node_properties)}

def calstr(s="sss"):
    bias = 100
    if str!=type(s):
        if int == type(s):
            return s+1
        else: return 1
    t = 0
    for i in s:
        t+=ord(i)
    return t+1+bias

def draw_tsne(X=np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]]),y=[12,34,56,78], dim_ind = 'i'):
    # X:features    ;   y:labels
    '''X:features and y:labels'''
    X=np.array(X)
    
    yy = [ calstr(i)%148 for i in y]
    # print(yy)
    tsne = manifold.TSNE(n_components=3, init='pca', random_state=501)
    X_tsne = tsne.fit_transform(X)
    print("Org data dimension is {}.Embedded data dimension is {}".format(X.shape[-1], X_tsne.shape[-1]))

    colors_list = []
    for colorrr in colors.cnames.keys():
        colors_list.append(colorrr)
    marker_list = []
    for mark in markers.MarkerStyle.markers.keys():
        marker_list.append(mark)
        
    '''嵌入空间可视化'''
    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化
    plt.figure(figsize=(8, 8))

    hist = []
    for i in range(X_norm.shape[0]):
        if y[i] in hist :
            plt.scatter(X_norm[i, 0], X_norm[i, 1], color=colors_list[yy[i]], marker=marker_list[yy[i]%len(marker_list)])
        else: 
            plt.scatter(X_norm[i, 0], X_norm[i, 1], color=colors_list[yy[i]],label = y[i], marker=marker_list[yy[i]%len(marker_list)])
            hist.append(y[i])
        # plt.text(X_norm[i, 0], X_norm[i, 1], str(y[i]), color=plt.cm.Set1(y[i]), 
        #         fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([])
    plt.yticks([])
    plt.legend()
    plt.show()
    plt.savefig("Tsne-"+str(dim_ind)+".png") # plt.show()
    pass

def calculate_auc(labels, predicts, name_list, note=''):
    # print('--debug-- ',labels,predicts)
    fpr_list = []
    tpr_list = []
    AUC_list = []
    for i in range(len(labels)):
        fpr, tpr, thresholds = roc_curve(labels[i], predicts[i], pos_label=1)
        fpr_list.append(fpr)
        tpr_list.append(tpr)
    for i in range(len(fpr_list)):
        AUC = auc(fpr_list[i], tpr_list[i])
        AUC_list.append(AUC)
        print ("auc"+note+" : ",AUC)

    colors_list = []
    for colorrr in colors.cnames.keys():
        colors_list.append(colorrr)

    plt.figure()
    lw=2
    #plt.figure(figsize(10,10))
    for i in range(len(fpr_list)):
        plt.plot(fpr_list[i],tpr_list[i],color=colors_list[i+10],lw=lw,label='%s ROC curve (area=%0.2f)'%(name_list[i],AUC_list[i]))
    plt.plot([0,1],[0,1],color='navy',lw=lw,linestyle='--')
    plt.xlim([0.0,1.0])
    plt.ylim([0.0,1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    #plt.show()
    statistics_file = "./statistics/"
    if not os.path.exists(statistics_file) :
        os.makedirs(statistics_file)
    plt.savefig(statistics_file+"roc"+str(note)+".png")
#=======================================
    return AUC

if __name__ == "__main__":

    # draw AUC
    
    labels = []
    predicts = []
    name_list = []

    labels.append(np.load('statistics/our_model/labels_s_1653357134.5027738.npy'))
    predicts.append(np.load('statistics/our_model/predicts_s_1653357134.5027738.npy'))
    name_list.append("PDM-s")
    labels.append(np.load('statistics/our_model/labels_t_1653357133.3738678.npy'))
    predicts.append(np.load('statistics/our_model/predicts_t_1653357133.3738678.npy'))
    name_list.append("PDM-t")
    labels.append(np.load('statistics/our_model/labels_s_1653357340.2507641.npy'))
    predicts.append(np.load('statistics/our_model/predicts_s_1653357340.2507641.npy'))
    name_list.append("PDM-diffpool-s")
    labels.append(np.load('statistics/our_model/labels_t_1653357309.4367876.npy'))
    predicts.append(np.load('statistics/our_model/predicts_t_1653357309.4367876.npy'))
    name_list.append("PDM-diffpool-t")
    
    labels.append(np.load('statistics/palmtree-vulseeker-npy/valid_labels_1653358145.1035924.npy').reshape(-1))
    predicts.append(np.load('statistics/palmtree-vulseeker-npy/valid_predicts_1653358156.5318065.npy').reshape(-1))
    name_list.append("PalmTree+VulSeeker")
    labels.append(np.load('statistics/binseeker-npy/train_labels_1653358751.5349321.npy').reshape(-1))
    predicts.append(np.load('statistics/binseeker-npy/train_predicts_1653358759.1476328.npy').reshape(-1))
    name_list.append("VulSeeker")

    calculate_auc(labels, predicts,name_list)
