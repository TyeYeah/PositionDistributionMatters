from pprint import pprint
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable
from config import *
import numpy as np
import bird
from bird import dataset
from bird import trainer
import pickle as pkl
import os,re

import r2pipe,glob,os,shutil,json
from multiprocessing import Pool
from tqdm import tqdm, trange
import eval_utils as utils

train_cfg_dataset = "data/training/cfg_train_asm.txt"
train_dfg_dataset = "data/training/dfg_train_asm.txt"
test_dataset = "data/training/test_asm.txt"
vocab_path = "data/output/vocab_asm"
output_path = "data/output/transformer_asm"

lang_mode = "opcode" # "esil" or "disasm" or "opcode"
# example: 
# "esil": "0x204,rip,-,rdi,="
# "opcode": "lea rdi, [rip - 0x204]"
# "disasm": "lea rdi, [main]"

bin_bird_paths = glob.glob('./bin_bird/*')
bin_pdm_paths = glob.glob('./bin_pdm/*')

multiple_process_switch = True

def makesuredirs(target_dir):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

def removesuredirs(target_dir):
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)

def is_number(s):
    # first deal with corner case
    if s.startswith('0x') or s.startswith('-0x'):
        return True
    # use try-catch to check
    try:  # if float(s) works, return True
        float(s)
        return True
    except ValueError:  # ValueError: passing invalid parameters
        pass  # pass this error
    try:
        import unicodedata 
        unicodedata.numeric(s)  # like float(s)
        return True
    except (TypeError, ValueError):
        pass
    return False

def deal_with_OOV(r2,sentence):
    iSj = r2.cmdj('iSj')
    sentence = sentence.replace('[',' [ ').replace(']',' ] ').replace('{','{ ').replace('}',' }').replace('(',' ( ').replace(')',' )')
    sent = re.split(' |,',sentence)
    for i in range(len(sent)):
        if is_number(sent[i]):
            section_flag = 0
            for isj in iSj:
                if eval(sent[i]) in range(isj['paddr'],isj['paddr']+isj['size']+1):
                    sent[i] = '['+isj['name']+']'
                    section_flag = 1
                    break
            if not section_flag:
                if eval(sent[i]) > 0x9000000:
                    sent[i] = '[res]'
                elif eval(sent[i]) > 0x10000:
                    sent[i] = '[addr]'
                else: 
                    sent[i] = '[offset]'
    return ' '.join(sent)

def bird_gen_cfg_train(r2, func_name,train_cfg_dataset):
    r2.cmd('s '+func_name)
    fcfg = open(train_cfg_dataset,'a')
    agfj = r2.cmdj('agfj') # afbj, aflj, agfj, agdj
    cfg_result = []

    blocks = agfj[0]['blocks']
    
    for i in range(len(blocks)):
        j = -1
        for j in range(len(blocks[i]['ops'])-1):
            if blocks[i]['ops'][j+1]['type'] == "invalid" or blocks[i]['ops'][j+1]['type'] == "invalid": # if so, no 'disasm' key 
                j-=1
                break
            inst1 = deal_with_OOV(r2,blocks[i]['ops'][j][lang_mode])
            inst2 = deal_with_OOV(r2,blocks[i]['ops'][j+1][lang_mode])
            cfg_line = inst1+'\t'+inst2+'\n'
            if len(inst1.split())>0 and len(inst2.split())>0:
                cfg_result.append(cfg_line)
                fcfg.write(cfg_line)
        
        if 'jump' in blocks[i].keys():
            r2.cmd('s '+hex(blocks[i]['jump']))
            pdj = r2.cmdj('pdj')
            if lang_mode in pdj[0].keys():
                inst1 = deal_with_OOV(r2,blocks[i]['ops'][j+1][lang_mode])
                inst2 = deal_with_OOV(r2,pdj[0][lang_mode])
                cfg_line = inst1+'\t'+inst2+'\n'
                if len(inst1.split())>0 and len(inst2.split())>0:
                    cfg_result.append(cfg_line)
                    fcfg.write(cfg_line)
        if 'fail' in blocks[i].keys():
            r2.cmd('s '+hex(blocks[i]['fail']))
            pdj = r2.cmdj('pdj')
            if lang_mode in pdj[0].keys():
                inst1 = deal_with_OOV(r2,blocks[i]['ops'][j+1][lang_mode])
                inst2 = deal_with_OOV(r2,pdj[0][lang_mode])
                cfg_line = inst1+'\t'+inst2+'\n'
                if len(inst1.split())>0 and len(inst2.split())>0:
                    cfg_result.append(cfg_line)
                    fcfg.write(cfg_line)
        r2.cmd('s '+func_name)
    # print(cfg_result)
    # input('pause...............')
    return cfg_result

def bird_gen_dfg_train(r2, func_name,train_dfg_dataset):
    r2.cmd('s '+func_name)
    fdfg = open(train_dfg_dataset,'a')
    axvj = r2.cmdj('axvj')
    agfj = r2.cmdj('agfj') # afbj, aflj, agfj, agdj

    dfg_result = []
    data_names = []

    for reads in axvj['reads']:
        rname = reads['name']
        raddrs = reads['addrs']
        if len(raddrs) >0:
            for writes in axvj['writes']:
                wname = writes['name']
                waddrs = writes['addrs']
                if wname == rname and len(waddrs)>0:
                    data_names.append((wname,raddrs,waddrs))

    for data in data_names:
        for w in data[2]:
            for r in data[1]:
                wpdj = r2.cmdj('s '+hex(w)+';pdj')
                wdisasm = deal_with_OOV(r2,wpdj[0][lang_mode])
                rpdj = r2.cmdj('s '+hex(r)+';pdj')
                rdisasm = deal_with_OOV(r2,rpdj[0][lang_mode])
                r2.cmd('s '+func_name)
                dfg_line = wdisasm+"\t"+rdisasm+"\n"
                dfg_result.append(dfg_line)
                fdfg.write(dfg_line)

    return dfg_result

def bird_gen_test(r2, func_name,test_dataset):
    bird_gen_cfg_train(r2,func_name,test_dataset)
    bird_gen_dfg_train(r2,func_name,test_dataset)

def bird_r2task(bin_path): # sym.xxx
    print("radare2 starts analysing "+bin_path+" ......")
    r2 = r2pipe.open(bin_path)
    r2.cmd('aaaa')
    bin_info = r2.cmdj('ij')
    aflj = r2.cmdj("aflj")
    print("radare2 analysing finished √")

    ### examples
    # print(r2.cmd("afl"))
    # print(r2.cmdj("aflj"))            # evaluates JSONs and returns an object
    # print(r2.cmdj("ij")['core']['format']) # print(r2.cmdj("ij").core.format)  # shows file format

    ### tasks
    num_tick = 0
    for i in tqdm(range(len(aflj))):
        current_func_name = aflj[i]['name']
        
        current_func_addr = hex(aflj[i]['offset']) # kinda duplicated
        if len(r2.cmdj('s '+current_func_addr+';'+'agfj'))==0:  # some fcn.xxx is too enormous to get json result back
            continue
            
        # print('======'+bin_path+" -> "+current_func_name+'======',random.random())
        if 'callrefs' not in aflj[i].keys():    # check if it is a real function
            continue                            # or a single code snippet
        
        # train:test = 8:2
        num_tick+=1
        if num_tick % 10 == 0 :
            # print("generate testing dataset")
            bird_gen_test(r2,current_func_name,test_dataset)
            continue
        else:
            # print("generate training dataset")
            bird_gen_cfg_train(r2,current_func_name,train_cfg_dataset)
            bird_gen_dfg_train(r2,current_func_name,train_dfg_dataset)
        
    os.system('echo "'+time.asctime(time.localtime(time.time()))+': '+bin_path+'" >> bird_train_data_gen.log')
    ### task end
    r2.quit()

def bird_prepare_dataset():
    global train_cfg_dataset,train_dfg_dataset,test_dataset,vocab_path,output_path,lang_mode
    
    # print("clean up remainings ......")
    # removesuredirs('/'.join(train_cfg_dataset.split('/')[:-1]))
    # # removesuredirs('/'.join(train_dfg_dataset.split('/')[:-1])) # duplicated
    # # removesuredirs('/'.join(test_dataset.split('/')[:-1]))
    # print("clean up finished √")

    print("make dirs ......")
    makesuredirs('/'.join(train_cfg_dataset.split('/')[:-1]))
    # makesuredirs('/'.join(train_dfg_dataset.split('/')[:-1])) # duplicated
    # makesuredirs('/'.join(test_dataset.split('/')[:-1]))
    print("make dirs finished √")

    if multiple_process_switch:
        with Pool(processes=20) as pool:
            for in_path in tqdm(bin_bird_paths):
                print("in_path:",in_path)
                pool.apply_async(bird_r2task, (in_path,))
            pool.close()
            pool.join()
    else:
        for in_path in tqdm(bin_bird_paths):
            print("in_path:",in_path)
            bird_r2task(in_path)

def bird_train():
    print(bird.__file__)
    global train_cfg_dataset,train_dfg_dataset,test_dataset,vocab_path,output_path,lang_mode,trainer
    
    with open(train_cfg_dataset, "r", encoding="utf-8") as f1:
        with open(train_dfg_dataset, "r", encoding="utf-8") as f2:
            vocab = dataset.WordVocab([f1, f2], max_size=13000, min_freq=1)

    print("VOCAB SIZE:", len(vocab))
    if not os.path.exists(vocab_path.split('/')[0]):
        os.makedirs(vocab_path.split('/')[0])
    vocab.save_vocab(vocab_path)


    print("Loading Vocab", vocab_path)
    vocab = dataset.WordVocab.load_vocab(vocab_path)
    print("Vocab Size: ", len(vocab))
    # print(vocab.itos)


    print("Loading Train Dataset")
    train_dataset = dataset.BERTDataset(train_cfg_dataset, train_dfg_dataset, vocab, seq_len=20,
                                corpus_lines=None, on_memory=True)

    print("Loading Test Dataset", test_dataset)
    test_dataset = bird.dataset.BERTDataset(test_dataset, test_dataset, vocab, seq_len=20, on_memory=True) \
        if test_dataset is not None else None

    print("Creating Dataloader")
    train_data_loader = DataLoader(train_dataset, batch_size=256, num_workers=10)



    test_data_loader = DataLoader(test_dataset, batch_size=256, num_workers=10) \
        if test_dataset is not None else None

    print("Building BERT model")
    bert = bird.BERT(len(vocab), hidden=64, n_layers=12, attn_heads=8, dropout=0.0)

    print("Creating BERT Trainer")
    trainer = trainer.BERTTrainer(bert, len(vocab), train_dataloader=train_data_loader, test_dataloader=test_data_loader,
                            lr=1e-5, betas=(0.9, 0.999), weight_decay=0.0,
                            with_cuda=True, cuda_devices=[0], log_freq=100)


    print("Training Start, ===")
    for epoch in range(20):
        trainer.train(epoch)
        trainer.save(epoch, output_path)
        if test_data_loader is not None:
            print("Testing Start, ===")
            trainer.test(epoch)     
    pass

def bird_embedding(bird_model,inst_list):
    if not isinstance(inst_list,list):
        if isinstance(inst_list,str):
            inst_list = list(inst_list)
        else: return "illegal bird input, so ......"
    else:
        embeddings = bird_model.encode(inst_list)
        return embeddings

def pdm_gen_cfg_txt(r2, func_name,to_path):
    r2.cmd('s '+func_name)
    # fcfg = open(to_path+os.sep+func_name+'_cfg.txt','w')
    agfj = r2.cmdj('agfj') # afbj, aflj, agfj, agdj
    cfg_result = []

    blocks = agfj[0]['blocks']
    
    bb_addrs = []
    for bb in blocks:
        bb_addrs.append(bb['offset']) 

    for i in range(len(blocks)):
        bb = []
        # print('=======cfg-block-'+str(i)+'=======')
        # print(hex(blocks[i]['offset']))
        bb.append(hex(blocks[i]['offset']))
        # print(blocks[i])
        if 'jump' in blocks[i].keys():
            if blocks[i]['jump'] in bb_addrs:
                # print('jump:'+hex(blocks[i]['jump']))
                bb.append(' '+hex(blocks[i]['jump']))
        if 'fail' in blocks[i].keys():
            if blocks[i]['fail'] in bb_addrs:
                # print('fail:'+hex(blocks[i]['fail']))
                bb.append(' '+hex(blocks[i]['fail']))
        # print('=======cfg-block-'+str(i)+'=======')
        # print(bb)
        # fcfg.write(''.join(bb)+'\n')
        cfg_result.append(''.join(bb)+'\n')
    return cfg_result

def pdm_gen_dfg_txt(r2, func_name,to_path):
    r2.cmd('s '+func_name)
    # fdfg = open(to_path+os.sep+func_name+'_dfg.txt','w')
    axvj = r2.cmdj('axvj')
    agfj = r2.cmdj('agfj') # afbj, aflj, agfj, agdj
    dfg_result = []

    blocks = agfj[0]['blocks']
    blocks_dict = []
    for i in range(len(blocks)):
        blocks_dict.append({})
        blocks_dict[i]['index'] = i
        blocks_dict[i]['bstart'] = blocks[i]['offset']
        blocks_dict[i]['bsize'] = blocks[i]['size']
        blocks_dict[i]['bend'] = blocks_dict[i]['bstart'] + blocks_dict[i]['bsize'] - 2
    # print(blocks_dict)
    # print(axvj)
    ### preprocess reads and writes
    for reads in axvj['reads']:
        temp=[]
        for addrs in reads['addrs']:
            for bbb in blocks_dict:
                # print(addrs,bbb['bstart'],"=====================")
                # print(addrs in range(bbb['bstart'],bbb['bend']+1))
                if addrs in range(bbb['bstart'],bbb['bend']+1):
                    temp.append(bbb['bstart'])
        reads['addrs'] = list(set(temp))
    for writes in axvj['writes']:
        temp=[]
        for addrs in writes['addrs']:
            for bbb in blocks_dict:
                # print(addrs,bbb['bstart'],"=====================")
                # print(addrs in range(bbb['bstart'],bbb['bend']+1))
                if addrs in range(bbb['bstart'],bbb['bend']+1):
                    temp.append(bbb['bstart'])
        writes['addrs'] = list(set(temp))
    # print("----------split-line---------")
    # print(axvj)
    ### make dfg list
    dfg_dict = {}
    for bb in blocks:
        dfg_dict[bb['offset']] = []
        for writes in axvj['writes']:
            for addr in writes['addrs']:
                # print("if1:",bb['offset'] == addr,bb['offset'] , addr)
                if bb['offset'] == addr:
                    for reads in axvj['reads']:
                        # print("if2",reads['name'] == writes['name'], reads['addrs'] != [], reads['name'], writes['name'], reads['addrs'])
                        if reads['name'] == writes['name'] and reads['addrs'] != []:
                            for bto in reads['addrs']:
                                if bto != bb['offset']:
                                    dfg_dict[bb['offset']].append(bto)
    # print(dfg_dict)
    for kk in dfg_dict:
        dfg_dict[kk] = [" "+hex(i) for i in  dfg_dict[kk]]
        # fdfg.write(hex(kk)+''.join(dfg_dict[kk])+'\n')
        dfg_result.append(hex(kk)+''.join(dfg_dict[kk])+'\n')

    return dfg_result

def pdm_gen_fea_csv(r2, func_name,to_path,bird_model):
    r2.cmd('s '+func_name)
    # fcsv = open(to_path+os.sep+func_name+'_fea.csv','w')
    fea_result = []
    agfj = r2.cmdj('agfj') # afbj, aflj, agfj, agdj

    # calculate numbers of different type instructions 
    for bb in agfj[0]['blocks']:

        bb_disasm = []
        for inst in bb['ops']:
            itp = inst['type']
            if itp != 'invalid':
                bb_disasm.append(deal_with_OOV(r2,inst[lang_mode]))
            else: bb_disasm.append('invalid')

        bb_embeddings = bird_embedding(bird_model,bb_disasm)
        # bb_embeddings = bird_model.encode(bb_disasm)
        bb_emb = np.mean(bb_embeddings,axis=0)
        bb_emb_list = bb_emb.tolist()
        bb_emb_list = [str(i) for i in bb_emb_list]
        # print(hex(bb['offset'])+','+','.join(bb_emb_list)+',\n')
        # fcsv.write(hex(bb['offset'])+','+','.join(bb_emb_list)+',\n')
        fea_result.append(hex(bb['offset'])+','+','.join(bb_emb_list)+',\n')
    
    return fea_result   

def pdm_r2task(bin_path,bird_model): # sym.xxx
    print("radare2 starts analysing "+bin_path+" ......")
    r2 = r2pipe.open(bin_path)
    r2.cmd('aaaa')
    bin_info = r2.cmdj('ij')
    aflj = r2.cmdj("aflj")
    print("radare2 analysing finished √")

    ### examples
    # print(r2.cmd("afl"))
    # print(r2.cmdj("aflj"))            # evaluates JSONs and returns an object
    # print(r2.cmdj("ij")['core']['format']) # print(r2.cmdj("ij").core.format)  # shows file format

    ### tasks
    to_path = './binfeature'
    input_path = '../PDM/input/train'   # for training
    input_path = '../PDM/input/test'    # fro test
    # makesuredirs(to_path)
    makesuredirs(input_path)
    # print(program,version,bin_name,to_path)
        
    for i in trange(len(aflj)):
        current_func_name = aflj[i]['name']

        # ############################## to generate features #################################################
        # if "SSL_shutdown" not in current_func_name :
        #     continue
        # ###############################################################################

        print('\n======'+bin_path+" -> "+current_func_name+'======\n')

        current_func_addr = hex(aflj[i]['offset']) # kinda duplicated
        if len(r2.cmdj('s '+current_func_addr+';'+'agfj'))==0:
            continue
        if 'callrefs' not in aflj[i].keys(): # check if it is a real function
            continue    # or a single code snippet
        # if 'fcn.' in current_func_name: # as it is training process, we need symbols
        #     continue
        if 'sym.imp' in current_func_name: # sym.imp.* is just symbol link to library
            continue
        
        # input("waiting...")
        # print("+++++++++++++++++++++++generating-CFG+++++++++++++++++++++++\n")
        cfg_res = pdm_gen_cfg_txt(r2,current_func_name,to_path)
        # print("+++++++++++++++++++++++generating-DFG+++++++++++++++++++++++\n")
        dfg_res = pdm_gen_dfg_txt(r2,current_func_name,to_path)
        # print("+++++++++++++++++++++generating-features+++++++++++++++++++++\n")
        fea_res = pdm_gen_fea_csv(r2,current_func_name,to_path,bird_model)
        # input("Pause at function:"+current_func_name+','+current_func_addr+', press enter to continue ...')

        json_path = input_path+os.sep+bin_path.split('/')[-1]+os.sep+current_func_name+'.json'
        makesuredirs('/'.join(json_path.split('/')[:-1]))
        mapping = {}
        for i,line in enumerate(cfg_res):
            # print(i,addr)
            addr = line.split()[0]
            mapping[addr] = i
        # print(mapping)
        # print('======',cfg_res,'\n',dfg_res,'\n',fea_res,'======')
        # input()
        result = {}
        result["labels"] = {}
        if len(cfg_res) == 1:
            result["edges"] = [[0,0]];
            result["labels"][str(mapping[cfg_res[0].split()[0]])]=eval('['+fea_res[0].split()[0]+']')[1:];
            result["target"]=1
            with open(json_path,'w')as jf:
                jf.write(json.dumps(result))
            continue

        # cfg edges
        result["edges"] = []
        for cfg_line in cfg_res:
            cfg_blocks = cfg_line.split()
            fromblk = mapping[cfg_blocks[0]]
            for blk in cfg_blocks[1:]:
                toblk = mapping[blk]
                result["edges"].append([fromblk,toblk])
        # dfg edges
        for dfg_line in dfg_res:
            dfg_blocks = dfg_line.split()
            fromblk = mapping[dfg_blocks[0]]
            for blk in dfg_blocks[1:]:
                toblk = mapping[blk]
                result['edges'].append([fromblk,toblk])
        for fea_line in fea_res:
            fea_vec = eval('['+fea_line.split()[0]+']')[1:]
            result['labels'][str(mapping[fea_line.split(',')[0]])] = fea_vec
        result["target"] = 0
        with open(json_path,'w')as jf:
            print(len(result['labels']))
            if len(result['edges']): # prevent from "edges": [] , causing error in process json data
                jf.write(json.dumps(result))
        
    os.system('echo "'+time.asctime(time.localtime(time.time()))+': '+bin_path+'" >> pdm_train_data_gen.log')
    ### task end
    r2.quit()

def pdm_gen_input():
    ctx = torch.multiprocessing.get_context("spawn")
    # palmtree = utils.UsableTransformer(model_path="./transformer.ep19", vocab_path="./vocab")
    bird_model = utils.UsableTransformer(model_path=output_path+'.ep19', vocab_path=vocab_path)
    
    if multiple_process_switch:
        ### multiple process
        with ctx.Pool(processes=5) as pool:
            for in_path in tqdm(bin_pdm_paths):
                # print("in_path:",in_path)
                pool.apply_async(pdm_r2task, (in_path,bird_model,))
            pool.close()
            pool.join()
    else:
        ### single process
        for in_path in tqdm(bin_pdm_paths):
            print(in_path)
            pdm_r2task(in_path,bird_model)

    pass

def main():
    print("main")

if __name__ == '__main__':
    # main()

    bird_prepare_dataset()
    bird_train()
    pdm_gen_input()
    