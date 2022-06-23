"""Running CapsGNN."""

from utils_config import tab_printer, device, draw_tsne
from capsgnn import CapsGNNTrainer
from param_parser import parameter_parser
from cal_dis import cosine

import torch,sys,glob,re,time,os,random

def train_s():
    args = parameter_parser()
    tab_printer(args)
    model = CapsGNNTrainer(args)
    
    model.fit_s()
    torch.save(model.siamese,'model/model_siamese.pt')
    pass

def evaluate_s():
    tooltrainer = CapsGNNTrainer(parameter_parser())
    mod_paths = glob.glob('model/model_siamese*')
    
    tooltrainer.siamese = torch.load('model/model_siamese.pt')
    print("Choose siamese model:",'model/model_siamese.pt')

    tooltrainer.score_s()
    tooltrainer.vul_rank_s()
    pass

def embed_s(path1 = 'input/temp/api.cgi/fcn.0041120c.json', path2 = 'input/temp/login.cgi/fcn.0040530c.json', model_path = './model/model_siamese_diffpool_0519.pt'):
    model = torch.load(model_path, map_location = torch.device('cuda:0'))
    tooltrainer = CapsGNNTrainer(parameter_parser())
    data1 = tooltrainer.create_input_data(path1)
    data2 = tooltrainer.create_input_data(path2)
    emb1, emb2, losses = model.forward(data1,data2) # emb = model(data)
    embed1, rec1, losses1 = model.get_embedding(data1)
    embed2, rec2, losses2 = model.get_embedding(data2)
    emb1 = emb1.view(-1).cpu().detach().numpy()
    emb2 = emb2.view(-1).cpu().detach().numpy()
    print(1-cosine(emb1, emb2))
    print(torch.cosine_similarity(embed1.view(-1), embed2.view(-1),dim=0))

def train_t():
    args = parameter_parser()
    tab_printer(args)
    model = CapsGNNTrainer(args)
    
    model.fit_t()
    # model.score_t()
    torch.save(model.triplet,'model/model_triplet.pt')
    pass

def evaluate_t():
    tooltrainer = CapsGNNTrainer(parameter_parser())
    mod_paths = glob.glob('model/model_triplet*')

    tooltrainer.triplet = torch.load('model/model_triplet.pt')
    print("Choose triplet model:",'model/model_triplet.pt')

    tooltrainer.score_t()
    tooltrainer.vul_rank_t()
    pass

def embed_t(path1 = 'input/test_patch/busybox-1.34.1-clang-x86_64-O0_unstripped/sym.seek_to_zero.json', path2 = 'input/test/busybox-1_21_stable_clang-7.0_x86_64_O1_busybox_unstripped.elf/dbg.add_cmd.json', path3 = 'input/test/busybox-1_21_stable_clang-7.0_x86_64_O0_busybox_unstripped.elf/dbg.add_cmd.json', model_path = './model/model_triplet_0.pt'):
    model = torch.load(model_path, map_location = torch.device('cuda:1'))
    tooltrainer = CapsGNNTrainer(parameter_parser())
    data1 = tooltrainer.create_input_data(path1)
    data2 = tooltrainer.create_input_data(path2)
    data3 = tooltrainer.create_input_data(path3)
    emb1, emb2, emb3, losses = model.forward(data1,data2,data3) # emb = model(data)
    
    embed1, rec1, losses1 = model.get_embedding(data1)

    print(embed1.shape, rec1.shape)
    recon_input1 = embed1.view(8, 32)
    calculate_reconstruction = model.embedding_net.calculate_reconstruction_loss
    recon_output = calculate_reconstruction(recon_input1, data1['features'])
    print(embed1, rec1 == recon_output, rec1, data1['features'])
    

if __name__ == "__main__":
    args = parameter_parser()
    arg = args.expmode
    if arg == 'train_s':
        train_s()
    elif arg == 'evaluate_s':
        evaluate_s()
    elif arg == 'embed_s':
        embed_s()
    elif arg == 'train_t':
        train_t()
    elif arg == 'evaluate_t':
        evaluate_t()
    elif arg == 'embed_t':
        embed_t()
    else:
        print('wrong/no enough args, pass')
        print(arg,'is not supported')
        print('only support: train_s, evaluate_s, embed_s, train_t, evaluate_t, embed_t')
    