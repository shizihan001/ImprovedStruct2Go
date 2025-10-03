from cProfile import label
from random import shuffle
from re import T
from statistics import mode
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pickle
from model.network import SAGNetworkHierarchical,SAGNetworkGlobal
import torch.nn as nn
import torch.optim as optim
import dgl
from dgl.dataloading import GraphDataLoader
import torch.nn.functional as F
from tkinter import _flatten
from sklearn import metrics
from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_score, recall_score, f1_score, average_precision_score
import argparse
import warnings
import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from data_processing.divide_data import MyDataSet
from model.evaluation import cacul_aupr,calculate_performance
from transformers import get_cosine_schedule_with_warmup
from tqdm import tqdm
import logging
import os

# #损失函数improve
# class FocalLoss(nn.Module):
#     """Focal Loss for handling class imbalance"""
#     def __init__(self, alpha=1, gamma=2):
#         super(FocalLoss, self).__init__()
#         self.alpha = alpha
#         self.gamma = gamma
        
#     def forward(self, inputs, targets):
#         bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
#         pt = torch.exp(-bce_loss)
#         focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
#         return focal_loss.mean()

# def calculate_pos_weights(train_dataloader, device):
#     """计算正样本权重"""
#     all_labels = []
#     for _, _, labels, _ in train_dataloader:
#         labels = labels.squeeze()
#         if len(labels.shape) == 1:
#             labels = labels.unsqueeze(0)
#         all_labels.append(labels)
    
#     all_labels = torch.cat(all_labels, dim=0)
#     pos_counts = all_labels.sum(dim=0)
#     neg_counts = len(all_labels) - pos_counts
#     pos_weights = (neg_counts / (pos_counts + 1e-8)).to(device)
#     pos_weights = torch.clamp(pos_weights, 0.1, 10.0)  # 限制极端值
    
#     return pos_weights









#5.30
#原有内容，上面是损失函数improve
def create_logger(branch_name):
    logger = logging.getLogger(branch_name)
    handler1 = logging.StreamHandler()
    handler2 = logging.FileHandler(filename=os.path.join('log',branch_name+'.log'))
    logger.setLevel(logging.DEBUG)
    handler1.setLevel(logging.ERROR)
    handler2.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s")
    handler1.setFormatter(formatter)
    handler2.setFormatter(formatter)
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger

warnings.filterwarnings('ignore')
Thresholds = [x/100 for x in range(1,100)]

if __name__ == "__main__":
    #参数设置
    device = "cuda:0"
    
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-batch_size', '--batch_size', type=int, default=64,  help="the number of the bach size")
    parser.add_argument('-learningrate', '--learningrate',type=float,default=1e-4)
    parser.add_argument('-dropout', '--dropout',type=float,default=0.5)
    parser.add_argument('-branch', '--branch',type=str,default='mf')
    parser.add_argument('-labels_num', '--labels_num',type=int,default=328)

    #CBAM新加入
    parser.add_argument('--use_cbam', action='store_true', help='Whether to use Graph-CBAM attention for feature enhancement')
    
    args = parser.parse_args()
    # 根据选择的标签大类自动填写训练文件路径
    train_data_path = 'divided_data/'+args.branch+'_train_dataset'
    valid_data_path = 'divided_data/'+args.branch+'_valid_dataset'
    label_network_path = 'processed_data/label_'+args.branch+'_network'
    
    logger = create_logger(args.branch)
    
    with open(train_data_path,'rb')as f:
        train_dataset = pickle.load(f)
    with open(valid_data_path,'rb')as f:
        valid_dataset = pickle.load(f)
    with open(label_network_path,'rb')as f:
        label_network=pickle.load(f)
    label_network = label_network.to(device)

    # 载入/设置参数
    epoch_num = 20
    batch_size = args.batch_size
    learningrate = args.learningrate
    dropout = args.dropout
    labels_num = args.labels_num
    
    # 加载数据
    train_dataloader = GraphDataLoader(dataset=train_dataset, batch_size = batch_size, drop_last = False, shuffle = True)
    valid_dataloader = GraphDataLoader(dataset=valid_dataset, batch_size = batch_size, drop_last = False, shuffle = True)
    
    # TODO: 这里的输入特征应该是one-hot(26)+node2vec(30) = 56
    # 但暂时没做特征拼接，先用node2vec特征试试
    #model = SAGNetworkHierarchical(56, 512, labels_num, num_convs=2, pool_ratio=0.75, dropout=dropout).to(device)
    model = SAGNetworkHierarchical(56, 512, labels_num, num_convs=2, pool_ratio=0.75, dropout=dropout, use_cbam=args.use_cbam).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=learningrate)
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=epoch_num*len(train_dataloader))
    
    
    
    #criterion = nn.CrossEntropyLoss()
    #criterion = nn.BCEWithLogitsLoss()  # 适合多标签分类

    #暂用
    if args.branch == 'mf':
        
        criterion = nn.CrossEntropyLoss()
    else:
        # 其他分支使用标准BCE
        criterion = nn.BCEWithLogitsLoss()


    # # 在train_Struct2GO.py中为MF分支使用特殊处理
    # if args.branch == 'mf':
    #     # 使用Focal Loss处理严重不平衡
    #     criterion = FocalLoss(alpha=2, gamma=2)
    # else:
    #     # 其他分支使用标准BCE
    #     criterion = nn.BCEWithLogitsLoss()
    


    best_fscore = 0
    best_aupr = 0
    best_scores = []
    best_score_dict = {}
    logger.info('#########'+args.branch+'###########')
    logger.info('########start training###########')
    for epoch in range(epoch_num):
        print("epoch:",epoch)
        logger.info("epoch: "+str(epoch))
        model.train()
        train_loss = 0
        print("training")
        logger.info("training")
        for i,(pids, graphs, labels, seq_feats) in tqdm(enumerate(train_dataloader)):
            graphs = graphs.to(device)
            seq_feats = seq_feats.to(device)
            labels = labels.to(device)
            labels = torch.squeeze(labels)
            if len(labels.shape)==1:
                labels = labels.unsqueeze(0)
            
            optimizer.zero_grad()
            logits = model(graphs,seq_feats,label_network)
            
            loss = criterion(logits,labels)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            
            # 累加计算平均loss
            train_loss+=loss.item()
            if i%30 == 29:
                logger.info(f'Epoch: {epoch} / {epoch_num}, Step: {i} / {len(train_dataloader)}, Loss(batch): {loss.item()}')
        
        # 每四轮进行一次验证
        if epoch%4==3:
            model.eval()
            print("validating")
            logger.info("validating")
            valid_loss = 0
            pred = []
            actual = []
            with torch.no_grad():
                for i,(pids, graphs, labels, seq_feats) in tqdm(enumerate(valid_dataloader)):
                    graphs = graphs.to(device)
                    seq_feats = seq_feats.to(device)
                    labels = labels.to(device)
                    labels = torch.squeeze(labels)
                    if len(labels.shape)==1:
                        labels = labels.unsqueeze(0)
                    
                    logits = model(graphs,seq_feats,label_network)
                    logits = F.sigmoid(logits)
                    
                    loss = criterion(logits,labels)
                    
                    valid_loss += loss.item()
                    pred += logits.tolist()
                    actual += labels.tolist()
                    if i%10 == 9:
                        logger.info(f'Valid Step: {i} / {len(valid_dataloader)}, Loss(batch): {loss.item()}')

            fpr, tpr, th = roc_curve(np.array(actual).flatten(), np.array(pred).flatten(), pos_label=1)
            auc_score = auc(fpr, tpr)
            aupr = cacul_aupr(np.array(actual).flatten(), np.array(pred).flatten())
            score_dict = {}
            each_best_fcore = 0
            each_best_scores = []
            for thresh in tqdm(Thresholds):
                f_score, precision, recall = calculate_performance(actual, pred, label_network,threshold=thresh)
                if f_score >= each_best_fcore:
                    each_best_fcore = f_score
                    each_best_scores = [thresh, f_score, recall, precision, auc_score]
                    scores = [f_score, recall, precision, auc_score]
                    score_dict[thresh] = scores
            if each_best_fcore >= best_fscore:
                best_fscore = each_best_fcore
                best_scores = each_best_scores
                best_score_dict = score_dict
                best_aupr = aupr
                #torch.save(model, 'save_models/bestmodel_{}_{}_{}_{}.pkl'.format(args.branch,batch_size,learningrate,dropout))
                # 建议修改保存文件名以区分是否使用CBAM
                cbam_suffix = "_cbam" if args.use_cbam else ""
                torch.save(model, f'save_models/bestmodel_{args.branch}_{batch_size}_{learningrate}_{dropout}{cbam_suffix}.pkl')
            
            thresh, f_score, recall = each_best_scores[0], each_best_scores[1], each_best_scores[2]
            precision, auc_score = each_best_scores[3], each_best_scores[4]
            logger.info('########valid metric###########')
            logger.info('epoch{}, train_loss{}, valid_loss:{}'.format(epoch, train_loss/len(train_dataloader), valid_loss/len(valid_dataloader)))
            logger.info('threshold:{}, f_score{}, auc{}, recall{}, precision{}, aupr{}'.format(thresh, f_score, auc_score, recall, precision, aupr))
        
    logger.info('best_fscore: '+str(best_fscore))
    logger.info('best_scores[thresh,fmax,recall,precision,auc]: '+str(best_scores))
    logger.info('best_aupr: '+str(best_aupr))
    logger.info('best_score_dict: '+str(best_score_dict))

    # 在日志中记录是否使用CBAM
    logger.info(f'Using Graph-CBAM: {args.use_cbam}')




