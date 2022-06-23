from torch.nn.functional import embedding
from config import *
from torch import nn, rand
from scipy.ndimage.filters import gaussian_filter1d
from torch.autograd import Variable
import torch
import numpy as np
import eval_utils as utils

import r2pipe,glob,os,random,tqdm
import matplotlib.pyplot as plt
from sklearn import manifold,datasets
import matplotlib.colors as colors
from r2exp import is_number, deal_with_OOV, makesuredirs

def palmtree_main():
    palmtree = utils.UsableTransformer(model_path="./transformer.ep19", vocab_path="./vocab")
    # tokens has to be seperated by spaces.

    text = ["mov rbp rdi", 
            "mov ebx 0x1", 
            "mov rdx rbx", 
            "call memcpy", 
            "mov [ rcx + rbx ] 0x0", 
            "mov rcx rax", 
            "mov [ rax ] 0x2e"]

    # it is better to make batches as large as possible.
    embeddings = palmtree.encode(text)
    print("usable embedding of this basicblock:", embeddings)
    print("the shape of output tensor: ", embeddings.shape)

def opcode_emb():
    bird_model = utils.UsableTransformer(model_path="./cdfg_bert_1/transformer_asm.ep19", vocab_path="./cdfg_bert_1/vocab_asm")
    opcode = produce_corpus()
    embeddings = bird_model.encode(opcode)
    print("usable embedding of this basicblock:", embeddings)
    print("the shape of output tensor: ", embeddings.shape)
    similarity = torch.cosine_similarity(torch.from_numpy(embeddings[0]), torch.from_numpy(embeddings[1]), dim=0).numpy()
    print("similarity: ",similarity)

def produce_corpus(bin_path='/bin/sh'):
    r2 = r2pipe.open(bin_path)
    result = {}
    r2.cmd("aaaa")
    print(bin_path,'in produce_corpus =================================')
    for func in tqdm.tqdm(r2.cmdj('aflj')):
        # print('func:',func)
        for inst in r2.cmdj('s '+hex(func['offset'])+';pdfj')['ops']:
            # print('inst:',inst)
            if 'opcode' in inst.keys():
                opcode = inst['opcode']
                optype = inst['type']
                opcode_text = deal_with_OOV(r2,opcode)
                if len(opcode_text.split())<=2:
                    opcode_text = optype
                # os.system("echo '"+opcode_text+"' >> ops/"+optype+".txt") # write to file to store
                if optype not in result.keys():
                    result[optype]=set()
                    result[optype].add(opcode_text)
                else:
                    result[optype].add(opcode_text)
    return result

def calstr(s="sss"):
    # print(s,type(s))
    if str!=type(s):
        if int == type(s):
            return s
        else: return 1
    t = 0
    for i in s:
        t+=ord(i)
    # print(t)
    # input()
    return t

def draw_tsne(X_path='./eval_opcode_X.npy', y_path='./eval_opcode_y.npy', mod='bird'):
    # X means features, X_tsne means features after dimension reduction'''
    X = np.load(X_path)
    y = np.load(y_path)
    # y means labels
    yy = [ calstr(str(i))%148 for i in y]
    tsne = manifold.TSNE(n_components=3, init='pca', random_state=501)
    X_tsne = tsne.fit_transform(X)
    print("Org data dimension is {}. Embedded data dimension is {}".format(X.shape[-1], X_tsne.shape[-1]))

    colors_list = []
    for colorrr in colors.cnames.keys():
        colors_list.append(colorrr)
        
    # visualization of embedding space
    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)  # normalization
    plt.figure(figsize=(8, 8))

    hist = []
    for i in range(X_norm.shape[0]):
        if y[i] in hist:
            plt.scatter(X_norm[i, 0], X_norm[i, 1], color=colors_list[yy[i]])
        elif len(hist)<=100:     # control the number of classes
            plt.scatter(X_norm[i, 0], X_norm[i, 1], color=colors_list[yy[i]],label = y[i])
            hist.append(y[i])
        # plt.text(X_norm[i, 0], X_norm[i, 1], str(y[i]), color=plt.cm.Set1(y[i]), 
        #         fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([])
    plt.yticks([])
    plt.legend() # (loc='upper right') # default: loc="best"
    plt.savefig("tsne_"+mod+".png") # plt.show()
    pass

def evaluate_opcode_classify(mod = 'bird'):
    if mod == 'bird':
        bird_model = utils.UsableTransformer(model_path="./cdfg_bert_1/transformer_asm.ep19", vocab_path="./cdfg_bert_1/vocab_asm")
    elif mod == 'palmtree':
        bird_model = utils.UsableTransformer(model_path="./transformer_x86.ep19", vocab_path="./vocab_x86")
    
    test_paths = glob.glob("./bin_test/*")
    tested_dict = {}
    for test_bin in test_paths:
        test_dict = produce_corpus(test_bin)
        if len(tested_dict) == 0:
            tested_dict = test_dict
        else:
            for opkode in test_dict:
                if opkode in tested_dict:
                    test_dict[opkode] = test_dict[opkode].union(tested_dict[opkode])
            tested_dict.update(test_dict)
    # print(tested_dict)
    # input()
    makesuredirs('ops')
    for optype in tested_dict:
        for opcode_text in tested_dict[optype]:
            os.system("echo '"+opcode_text+"' >> ops/"+optype+".txt") # write to file to store
    X = []; y = []; X_path = './eval_opcode_X.npy'; y_path = './eval_opcode_y.npy'
    sum_num = 0; acc_num = 0

    for opkode in tqdm.tqdm(tested_dict):
        for opcode in tqdm.tqdm(tested_dict):
            if opkode == opcode:
                continue
            inzts = list(tested_dict[opkode])
            insts = list(tested_dict[opcode])
            for inst0 in inzts:
                if len(insts)>=2:
                    inst1, inst2 = random.sample(insts,2)
                else: 
                    if len(inzts)>=2:
                        inst0 = insts[0]
                        opkode, opcode = opcode, opkode
                        inst1, inst2 = random.sample(inzts,2)
                    else: 
                        continue
                emb0, emb1, emb2 = bird_model.encode(inst0), bird_model.encode(inst1), bird_model.encode(inst2)
                sim01 = torch.cosine_similarity(torch.from_numpy(emb0[0]), torch.from_numpy(emb1[0]), dim=0)
                sim02 = torch.cosine_similarity(torch.from_numpy(emb0[0]), torch.from_numpy(emb2[0]), dim=0)
                sim12 = torch.cosine_similarity(torch.from_numpy(emb1[0]), torch.from_numpy(emb2[0]), dim=0)
                X.append(emb0.tolist()[0]);y.append(opkode)
                sum_num+=1
                if sim12>=sim02 or sim12>=sim01:
                    acc_num+=1
    print(acc_num/sum_num)
    print("Accuracy of distinguishing opcodes -->","Summary:",sum_num,"Accurate:",acc_num,'percent: {:.2%}'.format(acc_num/sum_num))
    np.save(X_path,X)
    np.save(y_path,y)
    # draw_tsne(X_path,y_path)
    return


def block_vectorization(block_inst_list,bird_model):
    bb_embeddings = bird_model.encode(block_inst_list)
    bb_emb = np.mean(bb_embeddings,axis=0)
    return bb_emb

def bin_block_emblist_gen(paths,bird_model):
    result=[]
    for path in tqdm.tqdm(paths):
        r2 = r2pipe.open(path)
        r2.cmd("aaaa")
        aflj = r2.cmdj('aflj')
        print(path,'in bin_block_emblist_gen =================================')
        for fj in tqdm.tqdm(aflj):
            # print(fj['name'])
            offset = fj['offset']
            r2.cmd('s '+hex(offset))
            agfj = r2.cmdj('agfj')
            blk_ind=0
            if not agfj:
                continue
            for bj in agfj[0]['blocks']:
                blk_ind+=1
                sym_num=0
                str_num=0
                ops_l = []
                for ops in bj['ops']:
                    if 'disasm' not in ops.keys():
                        ops_l.append("invalid")
                        continue
                    elif 'str' in ops['disasm']:
                        str_num+=1
                    elif 'sym' in ops['disasm']:
                        sym_num+=1
                    ops_l.append(ops['disasm']) 
                bb_emb = block_vectorization(ops_l,bird_model)
                result.append((fj['name'],blk_ind,sym_num,str_num,bb_emb))
    return result

def evaluate_block_search(mod = 'bird'):
    if mod == 'bird':
        bird_model = utils.UsableTransformer(model_path="./cdfg_bert_1/transformer_asm.ep19", vocab_path="./cdfg_bert_1/vocab_asm")
    elif mod == 'palmtree':
        bird_model = utils.UsableTransformer(model_path="./transformer_x86.ep19", vocab_path="./vocab_x86")
    
    test_path = glob.glob("./bin_test/*") # binary inside has to be from same source
    bb_embs = bin_block_emblist_gen(test_path,bird_model)

    precision = 0
    recall = 0
    top_list_num = 0
    for b1 in tqdm.tqdm(bb_embs):
        list_tmp = []
        for b2 in bb_embs:
            sim = torch.cosine_similarity(torch.from_numpy(b1[4]), torch.from_numpy(b2[4]), dim=0).numpy()
            if b1[0]==b2[0] and b1[3]==b2[3] and b1[2]==b2[2]:
                label = 'same'
                # if b1[1]!=b2[0] and b1[3]==0 and b2[2]==0:
                #     label = 'diff'
                #     real_same-=1
            else: 
                label = 'diff'
            list_tmp.append((b1[0:4],b2[0:4],label,sim))
        list_tmp.sort(key=lambda u:(-u[3]))
        if list_tmp[0][3] == 1:
            # print(list_tmp[0])
            # input("pop it? pop it! ")
            list_tmp.pop(0)  # to remove the exactly same pair
            pass
        top1 = list_tmp[0]
        # print('top1:',top1)
        top5 = list_tmp[0:5]
        # print('top5:',top5)
        top_list_num+=1
        if top1[2]=='same':
            recall+=1
        for t5 in top5:
            if t5[2]=='same':
                precision+=1
                break
        # print(precision,recall,top_list_num)
        # input()
    print("Accuracy of block searching -->","Block searching times",top_list_num,"Precision: {:.2%}".format(precision/top_list_num),"Recall: {:.2%}".format(recall/top_list_num))
    return precision,recall


def main():
    print("main")
    pass

if __name__ == '__main__':
    # opcode_emb()
    # main()
    mod = 'palmtree' # 
    evaluate_opcode_classify(mod)
    draw_tsne('eval_opcode_X.npy','eval_opcode_y.npy',mod)
    evaluate_block_search(mod)