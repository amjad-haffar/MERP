'''
since we are randomly selected segments of each song in each epoch, 
the number of epochs is explosive so by a high probability, every entire song is accounted for. lol.
'''

import os
import sys
sys.path.append(os.path.abspath(''))
sys.path.append(os.path.abspath('processing'))
print(sys.path)
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

import torch
from torch import optim, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

import util
from util_method import save_model, pearson_corr_loss
from testing_kfold import single_test, plot_pred_n_gts
### to edit accordingly.
from dataset import rdm_dataset as dataset_class

from networks import Three_FC_profile_gated as archi_linear
from networks import lstm_double_adaptive_gating as late_fusion_gating

from ave_exp_by_prof import ave_exps_by_profile

#####################
####    Train    ####
#####################
def train(train_loader, model, test_loader, fold_i, args):
    
    
    loss_log = {
        'train_mse' : [],
        'train_r' : [],
        'test_mse' : [],
        'test_r' : []
    }
    '''
        intial round
    '''
    with torch.no_grad():
        model.eval()
        epoch_loss_log = {'mse' : [],'r' : []}

        for batchidx, (feature, label) in enumerate(train_loader):
            numbatches = len(train_loader)
            # Transfer to GPU
            feature = feature.to(device).float()
            label = label.to(device).float()
            audio = feature[:, :, :audio_dim]
            fused_feature = feature[:, 0, audio_dim:]
            # clear gradients 
            optimizer.zero_grad()
            # forward pass
            output = model(audio, fused_feature)
            # output = model.forward(feature)
            output = output.squeeze(1)
            
            # MSE Loss calculation
            loss_mse = F.mse_loss(output, label)
            loss_r = pearson_corr_loss(output, label)

            epoch_loss_log['mse'].append(loss_mse.item())
            epoch_loss_log['r'].append(loss_r.item())
        
        aveloss_mse = np.average(epoch_loss_log['mse'])
        aveloss_r = np.average(epoch_loss_log['r'])
        print(f'Initial round without training || mse = {aveloss_mse:.2f} || r = {aveloss_r:.2f}')
        loss_log['train_mse'].append(aveloss_mse)
        loss_log['train_r'].append(aveloss_r)
    
    '''
        actual training
    '''

    for epoch in np.arange(args.num_epochs):
        model.train()
        start_time = time.time()
        epoch_loss_log = {'mse' : [],'r' : []}

        for batchidx, (feature, label) in enumerate(train_loader):
            
            numbatches = len(train_loader)
            # Transfer to GPU
            feature = feature.to(device).float()
            label = label.to(device).float()
            audio = feature[:, :, :audio_dim]
            fused_feature = feature[:, 0, audio_dim:]
            # clear gradients 
            optimizer.zero_grad()
            # forward pass
            output = model(audio, fused_feature)
            # print('out: ', output)

            output = output.squeeze(1)
            # print(f'output stuffs: max: {output.max()} min: {output.min()} mean: {output.mean()} shape: {output.shape}')
            # print(f'label stuffs: max: {label.max()} min: {label.min()} mean: {label.mean()} shape: {label.shape}')
            # torch.save(output, os.path.join(args.dir_path, save_models_foldername, f'{args.model_name}', 'output'))
            # torch.save(label, os.path.join(args.dir_path, save_models_foldername, f'{args.model_name}', 'label'))
            
            # loss
            loss_mse = F.mse_loss(output, label)
            loss_r = pearson_corr_loss(output, label)
            # loss = loss_mse*args.mse_weight + loss_r*args.r_weight
            
            # backward pass
            # loss.backward()
            if args.use_r_loss:
                loss = loss_mse - 0.1 * loss_r
            else:
                loss = loss_mse

            loss.backward()
            # loss_mse.backward()
            # torch.nn.utils.clip_grad_norm_(
            #     model.parameters(),
            #     max_norm=1.0
            # )
            # update parameters
            optimizer.step()

            # record training loss
            epoch_loss_log['mse'].append(loss_mse.item())
            epoch_loss_log['r'].append(loss_r.item())

            print(f'Epoch: {epoch} || Batch: {batchidx}/{numbatches} || mse = {loss_mse.item():5f} || r = {loss_r.item():5f}', end = '\r')
            
        # log average loss
        if args.use_sched:
            scheduler.step()
        aveloss_mse = np.average(epoch_loss_log['mse'])
        aveloss_r = np.average(epoch_loss_log['r'])
        # print(f'Initial round without training || mse = {aveloss_mse:.2f} || r = {aveloss_r:.2f}')
        loss_log['train_mse'].append(aveloss_mse)
        loss_log['train_r'].append(aveloss_r)
        print(' '*200)

        epoch_duration = time.time() - start_time
        print(f'Fold: {fold_i} || Epoch: {epoch:3} || mse: {aveloss_mse:8.5f} || r: {aveloss_r:8.5f} || time taken (s): {epoch_duration:8f}')
        if (epoch + 1) % 1 == 0:
            test_ave_mse, test_ave_r = test(model, test_loader)

            print(
                f'Fold: {fold_i} || Epoch: {epoch:3} || '
                f'test mse: {test_ave_mse:.4f} || '
                f'test r: {test_ave_r:.4f}'
            )

            loss_log['test_mse'].append(test_ave_mse)
            loss_log['test_r'].append(test_ave_r)

            model.train()

    # test_ave_mse, test_ave_r = test(model, test_loader)
    # print(f'test loss || mse: {test_ave_mse:.4f} || r: {test_ave_r:.4f}')

    # loss_log['test_mse'].append(test_ave_mse)
    # loss_log['test_r'].append(test_ave_r)

    return model, aveloss_mse, aveloss_r, test_ave_mse, test_ave_r

####################
####    Test    ####
####################

def test(model, test_loader):
    model.eval()
    losses = {'mse' : [], 'r' : []}
    with torch.no_grad():
        for feature, label in test_loader:
            feature = feature.to(device).float()
            label = label.to(device).float()
            audio = feature[:, :, :audio_dim]
            fused_feature = feature[:, 0, audio_dim:]
            # forward pass
            output = model(audio, fused_feature)
            output = output.squeeze(1)

            # loss
            mse = F.mse_loss(output, label)
            r = pearson_corr_loss(output, label)

            losses['mse'].append(mse.item())
            losses['r'].append(r.item())
        
    return np.mean(losses['mse']), np.mean(losses['r'])


if __name__ == "__main__":
    
    ########################
    ####    Argparse    ####
    ########################

    dir_path = os.path.dirname(os.path.realpath(__file__))

    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_path', type=str, default=dir_path)
    parser.add_argument('--linear', type=bool, default=False)
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--master', type=int, default=1) # use int instead of boolean.
    parser.add_argument('--affect_type', type=str, default='arousals', help='Can be either "arousals" or "valences"')
    parser.add_argument('--num_epochs', type=int, default=2)
    parser.add_argument('--model_name', type=str, default='test', help='Name of folder plots and model will be saved in')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=10)
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--num_timesteps', type=int, default=30)
    # parser.add_argument('--step_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.0001)
    # parser.add_argument('--mse_weight', type=float, default=1.0)
    # parser.add_argument('--r_weight', type=float, default=0.1)
    parser.add_argument('--conditions', nargs='+', type=str, default=['age'])
    parser.add_argument(
        '--use_r_loss',
        action='store_true'
    )
    parser.add_argument(
        '--use_sched',
        action='store_true'
    )
    
    

    args = parser.parse_args()
    # kinda messy but this ensures the model_names can be the same but saved separately.
    if args.master == 0:
        save_models_foldername = 'saved_models'
    else:
        save_models_foldername = 'saved_models_m'

    if args.linear:
        setattr(args, 'model_name', f'linear_{args.affect_type[0]}_p_{args.model_name}')
        exp_log_filepath = os.path.join(dir_path,save_models_foldername,'test_log_linear.pkl')
        archi = archi_linear
    else:
        setattr(args, 'model_name', f'{args.affect_type[0]}_p_{args.model_name}')
        exp_log_filepath = os.path.join(dir_path,save_models_foldername,'test_log_lstm.pkl')
        archi = late_fusion_gating
    print(args)

    # check if folder with same model_name exists. if not, create folder.
    savepath = os.path.join(dir_path,save_models_foldername, args.model_name)
    os.makedirs(savepath, exist_ok=True)
    os.makedirs(os.path.join(savepath, 'predictions'), exist_ok=True)
    
    #########################
    ####    Load Data    ####
    #########################

    # read labels from pickle
    
    # exps = pd.read_pickle('data/exps_std_a_profile_ave.pkl')
    pinfo = util.load_pickle('data/pinfo_numero.pkl')
    exps = pd.read_pickle('data/exps_ready3.pkl')
    
    if args.master == 1: # retrieve only master pinfos.
        pinfo = pinfo[pinfo['master'] == 1.0]
        
    exps = ave_exps_by_profile(exps, pinfo, args.affect_type, args.conditions)
    # print(exps.head())
    
    ####################
    ####    Cuda    ####
    ####################

    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    torch.manual_seed(42)
    torch.backends.cudnn.benchmark = True
    print('cuda: ', use_cuda)
    print('device: ', device)

    ###########################
    ####    Dataloading    ####
    ###########################

    def dataloader_prep(feat_dict, exps, args, test=False):
        params = {
            'shuffle': not test,
            'num_workers': args.num_workers,
            'batch_size': args.batch_size}

        if test:
            params['batch_size'] = 1
            seq_len = None
        else:
            seq_len=args.num_timesteps

        dataset = dataset_class(feat_dict, exps, seq_len=seq_len)
        loader = DataLoader(dataset, **params)
        return loader

    '''
    5 FOLD CROSS VALIDATION LEGGO
    '''
    loss_log_folds = {'train_loss_mse':[], 'train_loss_r':[], 'test_loss_mse':[], 'test_loss_r':[]}
    num_folds = 5
    for fold_i in range(num_folds):


        ########################
        ####    Training    ####
        ########################

        # load the data 
        # read audio features from pickle
        train_feat_dict = util.load_pickle(f'data/folds/train_feats_{fold_i}.pkl')
        test_feat_dict = util.load_pickle(f'data/folds/test_feats_{fold_i}.pkl')
        
        train_loader = dataloader_prep(train_feat_dict, exps, args, False)
        test_loader = dataloader_prep(test_feat_dict, exps, args, True)

        ###########################
        ####    Model param    ####
        ###########################
        audio_dim = 260
        profile_dim = 1    # only training
        ## MODEL
        model = archi(
            audio_dim=audio_dim,
            profile_dim=profile_dim,
            hidden_dim=args.hidden_dim
        ).to(device)
        model.float()
        print(model)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[20,30,40,45],
            gamma=0.5,
        )
        
        model, train_ave_mse, train_ave_r, test_ave_mse, test_ave_r = train(train_loader, model, test_loader, fold_i, args)

        file_name= f'{args.model_name}_{fold_i}'
        save_model(model, savepath, file_name)

        loss_log_folds['train_loss_mse'].append(train_ave_mse)
        loss_log_folds['train_loss_r']. append(train_ave_r)
        loss_log_folds['test_loss_mse'].append(test_ave_mse)
        loss_log_folds['test_loss_r']. append(test_ave_r)
        

        #######################
        ####    Testing    ####
        #######################

        # model = load_model(model, savepath, file_name)
        # test_ave_mse, test_ave_r, sum_test  = test(model, test_loader)

        # plot the predictions and ground truths

    # logging
    args_dict = vars(args)
    # print(type(args_dict))
    # args_dict['num_epochs'] = num_epochs
    args_dict['tr_mse'] = f'{np.mean(loss_log_folds["train_loss_mse"]):.4f}'
    args_dict['tr_r'] = f'{np.mean(loss_log_folds["train_loss_r"]):.4f}'
    # args_dict['tr_loss'] = f'{train_ave_mse+train_ave_r:.4f}'

    args_dict['tt_mse'] = f'{np.mean(loss_log_folds["test_loss_mse"]):.4f}'
    args_dict['tt_r'] = f'{np.mean(loss_log_folds["test_loss_r"]):.4f}'
    # args_dict['tt_loss'] = f'{sum_test:.4f}'

    for e in ['num_workers','dir_path', 'plot']:
        args_dict.pop(e)
    
    # print(args_dict)
    args_series = pd.Series(args_dict)
    args_df = args_series.to_frame().transpose()
    # print(args_df)

    if os.path.exists(exp_log_filepath):
        exp_log = pd.read_pickle(exp_log_filepath)
        exp_log = pd.concat([exp_log, args_df], ignore_index=True)
        pd.to_pickle(exp_log, exp_log_filepath)
        print(exp_log.to_string(index=False))
    else:
        pd.to_pickle(args_df, exp_log_filepath)
        print(args_df.to_string(index=False))