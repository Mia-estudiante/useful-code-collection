import os
import torch

def load_param(restore_from, model):

    if os.path.exists(restore_from):
        print('Resume training from {}'.format(restore_from))

    # Step1. load model parameters
    checkpoint = torch.load(restore_from, map_location='cpu')

    ###########################################################
    #Load checkpoint & Fit model parameters - Resume
    new_params = model.state_dict().copy()

    model_keys = [name for name, _ in model.named_parameters()]
    weight_file = [i for i in checkpoint['state_dict']]      

    for i in weight_file:
        if i not in model_keys: continue
        if 'module' in i:   #'module.'이 추가됨을 확인
            wo_i = i[7:]
        new_params[wo_i] = checkpoint[i]

    checkpoint['state_dict'] = new_params
    ###########################################################

    model.load_state_dict(checkpoint['state_dict'])
    # start_epoch = checkpoint['epoch']
    # model.to(device)

def load_param(net, trained_path):
    
    param_dict = torch.load(trained_path)
    
    if 'state_dict' in param_dict:
        param_dict = param_dict['state_dict']

    for name, param in net.named_parameters():
        if name in param_dict:
            try:
                net.state_dict()[name].copy_(param_dict[name])
            except:
                print(name, net.state_dict()[name].size(), param_dict[name].size())
        else:
            print(name)
        print("Loading pretrained model from {}".format(trained_path))
    