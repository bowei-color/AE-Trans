# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 12:02:16 2024

@author: Administrator
"""

import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch.nn as nn
import math
import torch.optim as optim
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score
import torch.optim.lr_scheduler as lr_scheduler
from torch.autograd import Variable
import argparse



# 定义位置编码类
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

# 定义多头注意力机制
class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(MultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def scaled_dot_product_attention(self, q, k, v, mask=None):
        d_k = q.size(-1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        output = torch.matmul(attn, v)
        return output, attn

    def split_heads(self, x, batch_size):
        return x.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        q = self.split_heads(self.q_proj(query), batch_size)
        k = self.split_heads(self.k_proj(key), batch_size)
        v = self.split_heads(self.v_proj(value), batch_size)

        attn_output, _ = self.scaled_dot_product_attention(q, k, v, mask)

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
        output = self.out_proj(attn_output)
        return output

# 定义Transformer编码器层
class MyTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(MyTransformerEncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src2 = self.self_attn(src, src, src, mask=src_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

# 定义Transformer编码器
class MyTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super(MyTransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([encoder_layer for _ in range(num_layers)])
        self.norm = norm

    def forward(self, src, mask=None, src_key_padding_mask=None):
        output = src
        for mod in self.layers:
            output = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
        if self.norm is not None:
            output = self.norm(output)
        return output

# 定义Transformer解码器层
class MyTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=256, dropout=0.1):
        super(MyTransformerDecoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None,
                memory_key_padding_mask=None):
        tgt2 = self.self_attn(tgt, tgt, tgt, mask=tgt_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(tgt, memory, memory, mask=memory_mask)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

# 定义Transformer解码器
class MyTransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None):
        super(MyTransformerDecoder, self).__init__()
        self.layers = nn.ModuleList([decoder_layer for _ in range(num_layers)])
        self.norm = norm

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None,
                memory_key_padding_mask=None):
        output = tgt
        for mod in self.layers:
            output = mod(output, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                          tgt_key_padding_mask=tgt_key_padding_mask,
                          memory_key_padding_mask=memory_key_padding_mask)
        if self.norm is not None:
            output = self.norm(output)
        return output



# 定义随机掩码模块
class FeatureMasking(nn.Module):
    def __init__(self, mask_num):
        super(FeatureMasking, self).__init__()
        self.mask_num = mask_num

    def forward(self, x):
        if self.mask_num > 0:
            x_clone = x.clone()  # 克隆张量，避免原地操作
            batch_size, feature_dim = x_clone.size()  # 处理二维张量
            mask_indices = torch.randperm(feature_dim)[:self.mask_num]
            x_clone[:, mask_indices] = 0
            return x_clone
        return x





# 定义自动编码器模型，将掩码操作移到AE编码器之前，并增加两个并行的AE编码器和解码器
class AutoencoderWithTransformer(nn.Module):
    def __init__(self, input_dim, encoding_dim, d_model=128, nhead=8, num_encoder_layers=3,
                 num_decoder_layers=3, dim_feedforward=256, dropout=0.1, mask_num=0):
        super(AutoencoderWithTransformer, self).__init__()

        # 定义随机掩码模块
        self.masking = FeatureMasking(mask_num)  # 掩码操作移到这里

        # 根据 input_dim 计算RNA和甲基化数据的拆分点
        self.rna_split_point = input_dim // 4  # RNA的前一半
        self.methylation_split_point = input_dim // 4  # 甲基化的前一半
        self.first_part_dim = self.rna_split_point + self.methylation_split_point
        self.second_part_dim = input_dim - self.first_part_dim  # 第二部分为剩余的维度

        # 自动编码器的编码器部分（第一部分：RNA前一部分 + 甲基化前一部分）
        self.encoder1 = nn.Sequential(
            nn.Linear(self.first_part_dim, 4096),
            nn.ReLU(),
            nn.Linear(4096, 512),
            nn.ReLU(),
            nn.Linear(512, encoding_dim),
            nn.ReLU()
        )

        # 自动编码器的编码器部分（第二部分：RNA剩余部分 + 甲基化剩余部分）
        self.encoder2 = nn.Sequential(
            nn.Linear(self.second_part_dim, 4096),
            nn.ReLU(),
            nn.Linear(4096, 512),
            nn.ReLU(),
            nn.Linear(512, encoding_dim),
            nn.ReLU()
        )

        # 位置编码
        self.pos_embedding = PositionalEncoding(d_model=d_model, dropout=dropout)

        # Transformer部分（第一部分）
        transformer_encoder_layer1 = MyTransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        transformer_encoder_norm1 = nn.LayerNorm(d_model)
        self.transformer_encoder1 = MyTransformerEncoder(transformer_encoder_layer1, num_encoder_layers, transformer_encoder_norm1)

        # Transformer部分（第二部分）
        transformer_encoder_layer2 = MyTransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        transformer_encoder_norm2 = nn.LayerNorm(d_model)
        self.transformer_encoder2 = MyTransformerEncoder(transformer_encoder_layer2, num_encoder_layers, transformer_encoder_norm2)

        # 线性层用于降维到 d_model
        self.fusion_to_d_model = nn.Linear(encoding_dim * 2, d_model)

        # Transformer解码器部分（第一部分）
        transformer_decoder_layer1 = MyTransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout)
        transformer_decoder_norm1 = nn.LayerNorm(d_model)
        self.transformer_decoder1 = MyTransformerDecoder(transformer_decoder_layer1, num_decoder_layers, transformer_decoder_norm1)

        # Transformer解码器部分（第二部分）
        transformer_decoder_layer2 = MyTransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout)
        transformer_decoder_norm2 = nn.LayerNorm(d_model)
        self.transformer_decoder2 = MyTransformerDecoder(transformer_decoder_layer2, num_decoder_layers, transformer_decoder_norm2)

        # 自动编码器的解码器部分（第一部分）
        self.decoder1 = nn.Sequential(
            nn.Linear(encoding_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 4096),
            nn.ReLU(),
            nn.Linear(4096, self.first_part_dim),
            nn.Sigmoid()
        )

        # 自动编码器的解码器部分（第二部分）
        self.decoder2 = nn.Sequential(
            nn.Linear(encoding_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 4096),
            nn.ReLU(),
            nn.Linear(4096, self.second_part_dim),
            nn.Sigmoid()
        )

        # 分类器部分
        self.classifier = nn.Sequential(
            nn.Linear(encoding_dim*2, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 8),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 掩码操作在AE编码器之前
        input_dim = x.shape[1]
        x = self.masking(x)

        # RNA的前一半和甲基化的前一半
        x1 = torch.cat((x[:, :self.rna_split_point], x[:, input_dim // 2:input_dim // 2 + self.methylation_split_point]), dim=1)

        # RNA的剩余部分和甲基化的剩余部分
        x2 = torch.cat((x[:, self.rna_split_point:input_dim // 2], x[:, input_dim // 2 + self.methylation_split_point:]), dim=1)

        # AE编码器部分
        encoded1 = self.encoder1(x1)
        encoded2 = self.encoder2(x2)

        # Transformer部分
        encoded1 = self.pos_embedding(encoded1.unsqueeze(0))  # 添加伪序列维度
        encoded2 = self.pos_embedding(encoded2.unsqueeze(0))  # 添加伪序列维度

        transformer_encoder1 = self.transformer_encoder1(encoded1).squeeze(0)  # 移除伪序列维度
        transformer_encoder2 = self.transformer_encoder2(encoded2).squeeze(0)  # 移除伪序列维度

        # 融合编码结果
        fused_encoding = torch.cat((transformer_encoder1, transformer_encoder2), dim=1)

        # 通过线性层降维到 d_model
        fused_encoding_reduced = self.fusion_to_d_model(fused_encoding)

        # Transformer解码部分，使用降维后的向量作为输入
        transformer_decoder1 = self.transformer_decoder1(fused_encoding_reduced.unsqueeze(0), encoded1).squeeze(0)
        transformer_decoder2 = self.transformer_decoder2(fused_encoding_reduced.unsqueeze(0), encoded2).squeeze(0)

        # AE解码器部分
        decoded1 = self.decoder1(transformer_decoder1)
        decoded2 = self.decoder2(transformer_decoder2)

        # 分类任务
        classification_output = self.classifier(fused_encoding)

        return classification_output, decoded1, decoded2
    



# 积分梯度函数
def integrated_gradients(model, x, baseline=None, steps=50):
    if baseline is None:
        baseline = torch.zeros_like(x)  # 使用全0向量作为基线
    
    # 梯度累加
    scaled_inputs = [baseline + (float(i) / steps) * (x - baseline) for i in range(steps + 1)]
    classifier_grads = []
    
    for scaled_input in scaled_inputs:
        scaled_input = Variable(scaled_input, requires_grad=True)
        
        # 前向传播计算输出
        logits, _, _ = model(scaled_input)
        model.zero_grad()
        
        # 如果是标量输出（如二分类任务）
        logit_target = logits[0]  # 只取第一个 logit 值
        
        logit_target.backward(retain_graph=True)
        classifier_grads.append(scaled_input.grad.cpu().detach().numpy())  # 确保分离梯度

    # 计算平均梯度
    avg_classifier_grads = np.average(classifier_grads[:-1], axis=0)
    integrated_classifier_grad = (x.cpu().detach().numpy() - baseline.cpu().detach().numpy()) * avg_classifier_grads
    
    return integrated_classifier_grad


# 计算总的平均积分梯度
def get_average_integrated_gradients(model, X_test, device):
    total_grads = None
    num_samples = len(X_test)
    
    for i in range(num_samples):
        x = torch.tensor(X_test[i:i+1].clone().detach(), requires_grad=True, dtype=torch.float32).to(device)
        
        # 计算样本的积分梯度
        classifier_grad = integrated_gradients(model, x)
        
        # 累加梯度
        if total_grads is None:
            total_grads = classifier_grad
        else:
            total_grads += classifier_grad
    
    # 计算平均梯度
    avg_grads = total_grads / num_samples
    
    return avg_grads
    

# 对积分梯度排序，返回三个排序结果
def get_sorted_features(avg_grads, selected_features, num):
    
    # 构建DataFrame，用于排序
    attributions = pd.DataFrame({
        'feature': selected_features,
        'classifier_attribution': np.abs(np.mean(avg_grads, axis=0))  # 计算每个特征的平均积分梯度贡献
    })
    
    # 对特征按贡献值进行降序排序
    sorted_attributions = attributions.sort_values(by='classifier_attribution', ascending=False)
    
    # 获取总的前200个特征
    top_all_features = sorted_attributions.head(num)
    
    # 从前200个特征中筛选出RNA特征（没有Metht_前缀的）
    top_rna_features = top_all_features[top_all_features['feature'].apply(lambda x: not x.startswith('Methy_'))]
    
    # 从前200个特征中筛选出甲基化特征（有Metht_前缀的）
    top_methylation_features = top_all_features[top_all_features['feature'].apply(lambda x: x.startswith('Methy_'))]
    
    return top_all_features, top_rna_features, top_methylation_features



    
# 主函数
def main(epochs, file_path_gene, file_path_methy):
    # 读取RNA和甲基化数据
    rna_file_path = file_path_gene
    methylation_file_path = file_path_methy

    rna_data = pd.read_csv(rna_file_path)
    methylation_data = pd.read_csv(methylation_file_path)

    # 提取特征和标签
    X_rna = rna_data.iloc[:, 2:].values  # RNA数据的特征
    y_rna = rna_data.iloc[:, 1].values   # RNA数据的标签

    X_methylation = methylation_data.iloc[:, 2:].values  # 甲基化数据的特征
    y_methylation = methylation_data.iloc[:, 1].values   # 甲基化数据的标签




    # 1. 划分RNA数据为正负样本
    X_rna_pos = X_rna[y_rna == 1]
    X_rna_neg = X_rna[y_rna == 0]

    # 2. 划分甲基化数据为正负样本
    X_methylation_pos = X_methylation[y_methylation == 1]
    X_methylation_neg = X_methylation[y_methylation == 0]

    # 3. 分别将RNA正样本、RNA负样本、甲基化正样本和甲基化负样本划分为训练集和测试集
    X_rna_pos_train, X_rna_pos_test = train_test_split(X_rna_pos, test_size=0.2, random_state=42)
    X_rna_neg_train, X_rna_neg_test = train_test_split(X_rna_neg, test_size=0.2, random_state=42)

    X_methylation_pos_train, X_methylation_pos_test = train_test_split(X_methylation_pos, test_size=0.2, random_state=42)
    X_methylation_neg_train, X_methylation_neg_test = train_test_split(X_methylation_neg, test_size=0.2, random_state=42)

    # 4. 正样本训练集融合：每个RNA正样本训练集拼接每个甲基化正样本训练集
    positive_train_samples = np.array([np.hstack((rna_sample, meth_sample)) 
                                       for rna_sample in X_rna_pos_train 
                                       for meth_sample in X_methylation_pos_train])

    # 5. 负样本训练集融合：每个RNA负样本训练集拼接每个甲基化负样本训练集
    negative_train_samples = np.array([np.hstack((rna_sample, meth_sample)) 
                                       for rna_sample in X_rna_neg_train 
                                       for meth_sample in X_methylation_neg_train])

    # 6. 正样本测试集融合：每个RNA正样本测试集拼接每个甲基化正样本测试集
    positive_test_samples = np.array([np.hstack((rna_sample, meth_sample)) 
                                      for rna_sample in X_rna_pos_test 
                                      for meth_sample in X_methylation_pos_test])

    # 7. 负样本测试集融合：每个RNA负样本测试集拼接每个甲基化负样本测试集
    negative_test_samples = np.array([np.hstack((rna_sample, meth_sample)) 
                                      for rna_sample in X_rna_neg_test 
                                      for meth_sample in X_methylation_neg_test])

    # 8. 标签：正样本集为1，负样本集为0
    y_positive_train = np.ones(len(positive_train_samples))
    y_negative_train = np.zeros(len(negative_train_samples))

    y_positive_test = np.ones(len(positive_test_samples))
    y_negative_test = np.zeros(len(negative_test_samples))

    # 9. 合并训练集正负样本
    X_train_combined = np.vstack((positive_train_samples, negative_train_samples))
    y_train_combined = np.concatenate((y_positive_train, y_negative_train))

    # 10. 合并测试集正负样本
    X_test_combined = np.vstack((positive_test_samples, negative_test_samples))
    y_test_combined = np.concatenate((y_positive_test, y_negative_test))

    # 11. 打乱训练集
    train_indices = np.random.permutation(len(X_train_combined))
    X_train_combined = X_train_combined[train_indices]
    y_train_combined = y_train_combined[train_indices]
    

    # 对特征进行标准化处理
    scaler = StandardScaler()
    X_train_combined = scaler.fit_transform(X_train_combined)


    # 12. 打乱测试集
    test_indices = np.random.permutation(len(X_test_combined))
    X_test_combined = X_test_combined[test_indices]
    y_test_combined = y_test_combined[test_indices]




    X_test_combined = scaler.fit_transform(X_test_combined)



    # 13. 将数据转换为PyTorch的张量格式
    X_train = torch.tensor(X_train_combined, dtype=torch.float32)
    X_test = torch.tensor(X_test_combined, dtype=torch.float32)
    y_test = torch.tensor(y_test_combined, dtype=torch.float32)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 定义模型参数
    input_dim = X_train.shape[1]
    encoding_dim = 128
    d_model = 128
    nhead = 8
    num_encoder_layers = 3
    num_decoder_layers = 3
    dim_feedforward = 256
    dropout = 0.1
    mask_num = 15000  # 设定掩码特征的数量
    batch_size = 64


    X_train_combined_all = X_train_combined
    y_train_combined_all = y_train_combined

    # 定义损失函数
    classification_loss_fn = nn.BCELoss()
    reconstruction_loss_fn = nn.MSELoss()


    # 存储每一折的评估指标和AUC曲线数据
    fold_accuracies = []
    fold_precisions = []
    fold_recalls = []
    fold_f1s = []
    fold_aucs = []
   



    kf = KFold(n_splits=5, shuffle=True, random_state=42)


    # 五折交叉验证训练
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_rna_pos_train)):
        print(f"Starting Fold {fold + 1}")

        # RNA正样本训练集划分为训练和验证集
        X_rna_pos_train_fold, X_rna_pos_val_fold = X_rna_pos_train[train_idx], X_rna_pos_train[val_idx]

        # 计算验证集比例
        val_ratio = len(val_idx) / len(X_rna_pos_train)

        # RNA负样本训练集按比例划分训练和验证集
        X_rna_neg_train_fold, X_rna_neg_val_fold = train_test_split(X_rna_neg_train, test_size=val_ratio, random_state=fold)

        # 甲基化正样本训练集按比例划分训练和验证集
        X_methylation_pos_train_fold, X_methylation_pos_val_fold = train_test_split(X_methylation_pos_train, test_size=val_ratio, random_state=fold)

        # 甲基化负样本训练集按比例划分训练和验证集
        X_methylation_neg_train_fold, X_methylation_neg_val_fold = train_test_split(X_methylation_neg_train, test_size=val_ratio, random_state=fold)

        # 融合训练集和验证集
        positive_train_samples = np.array([np.hstack((rna_sample, meth_sample)) 
                                            for rna_sample in X_rna_pos_train_fold 
                                            for meth_sample in X_methylation_pos_train_fold])

        negative_train_samples = np.array([np.hstack((rna_sample, meth_sample)) 
                                            for rna_sample in X_rna_neg_train_fold 
                                            for meth_sample in X_methylation_neg_train_fold])

        positive_val_samples = np.array([np.hstack((rna_sample, meth_sample)) 
                                          for rna_sample in X_rna_pos_val_fold 
                                          for meth_sample in X_methylation_pos_val_fold])

        negative_val_samples = np.array([np.hstack((rna_sample, meth_sample)) 
                                          for rna_sample in X_rna_neg_val_fold 
                                          for meth_sample in X_methylation_neg_val_fold])

        # 合并正负样本
        X_train_combined = np.vstack((positive_train_samples, negative_train_samples))
        y_train_combined = np.concatenate((np.ones(len(positive_train_samples)), np.zeros(len(negative_train_samples))))

        X_val_combined = np.vstack((positive_val_samples, negative_val_samples))
        y_val_combined = np.concatenate((np.ones(len(positive_val_samples)), np.zeros(len(negative_val_samples))))

        # 打乱训练集和验证集
        train_indices = np.random.permutation(len(X_train_combined))
        X_train_combined = X_train_combined[train_indices]
        y_train_combined = y_train_combined[train_indices]

        val_indices = np.random.permutation(len(X_val_combined))
        X_val_combined = X_val_combined[val_indices]
        y_val_combined = y_val_combined[val_indices]

        # 将数据转换为 DataLoader 格式
        train_dataset = torch.utils.data.TensorDataset(torch.tensor(X_train_combined, dtype=torch.float32), torch.tensor(y_train_combined, dtype=torch.float32))
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        val_dataset = torch.utils.data.TensorDataset(torch.tensor(X_val_combined, dtype=torch.float32), torch.tensor(y_val_combined, dtype=torch.float32))
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # 实例化模型
        model = AutoencoderWithTransformer(input_dim=input_dim, encoding_dim=encoding_dim, d_model=d_model, nhead=nhead,
                                            num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers,
                                            dim_feedforward=dim_feedforward, dropout=dropout, mask_num=mask_num)
        model.to(device)

        # 定义优化器并添加L2正则化
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

        # 引入学习率调度器
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

        # 训练每一折
        for epoch in range(epochs):
            model.train()
            total_loss = 0.0
            total_classification_loss = 0.0
            total_reconstruction_loss1 = 0.0
            total_reconstruction_loss2 = 0.0

            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)

                optimizer.zero_grad()

                # 前向传播
                logits, reconstructed1, reconstructed2 = model(batch_X)

                # 计算损失
                classification_loss = classification_loss_fn(logits, batch_y.unsqueeze(1))
                reconstruction_loss1 = reconstruction_loss_fn(reconstructed1, batch_X[:, :model.first_part_dim])
                reconstruction_loss2 = reconstruction_loss_fn(reconstructed2, batch_X[:, model.first_part_dim:])

                # 总损失 = 分类损失 + 两个重构损失
                loss = classification_loss + reconstruction_loss1 + reconstruction_loss2

                # 反向传播和优化
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                total_classification_loss += classification_loss.item()
                total_reconstruction_loss1 += reconstruction_loss1.item()
                total_reconstruction_loss2 += reconstruction_loss2.item()

            avg_loss = total_loss / len(train_loader)
            avg_classification_loss = total_classification_loss / len(train_loader)
            avg_reconstruction_loss1 = total_reconstruction_loss1 / len(train_loader)
            avg_reconstruction_loss2 = total_reconstruction_loss2 / len(train_loader)

            print(f'Epoch [{epoch+1}/{epochs}], Fold [{fold+1}], Classification Loss: {avg_classification_loss:.4f}, '
                  f'Reconstruction Loss 1: {avg_reconstruction_loss1:.4f}, Reconstruction Loss 2: {avg_reconstruction_loss2:.4f}, Loss: {avg_loss:.4f}')
            
            # 更新学习率调度器
            scheduler.step(avg_loss)

        # 验证阶段
        model.eval()
        y_true_val = []
        y_pred_val = []
        y_scores_val = []

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)

                logits, _, _ = model(batch_X)
                predicted = (logits > 0.5).float()

                y_true_val.extend(batch_y.cpu().numpy())
                y_pred_val.extend(predicted.cpu().numpy())
                y_scores_val.extend(logits.cpu().numpy())

        # 计算评估指标
        accuracy = accuracy_score(y_true_val, y_pred_val)
        precision = precision_score(y_true_val, y_pred_val)
        recall = recall_score(y_true_val, y_pred_val)
        f1 = f1_score(y_true_val, y_pred_val)
        auc_value = roc_auc_score(y_true_val, y_scores_val)

        fold_accuracies.append(accuracy)
        fold_precisions.append(precision)
        fold_recalls.append(recall)
        fold_f1s.append(f1)
        fold_aucs.append(auc_value)

        print(f'Fold {fold+1} - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, '
              f'F1-score: {f1:.4f}, AUC: {auc_value:.4f}')

    # 计算五折交叉验证的平均评估指标
    avg_accuracy = np.mean(fold_accuracies)
    avg_precision = np.mean(fold_precisions)
    avg_recall = np.mean(fold_recalls)
    avg_f1 = np.mean(fold_f1s)
    avg_auc = np.mean(fold_aucs)

    cv_resultdata = {
        "Evaluation indicators": ["Accuracy", "Precision", "Recall", "F1-Score", "AUC"],
        "Value": [round(avg_accuracy, 4), round(avg_precision, 4), round(avg_recall, 4), round(avg_f1, 4), round(avg_auc, 4)]
        }

    cv_df = pd.DataFrame(cv_resultdata)

    # cv_df.to_csv("../result/cv_resultdata.csv", index=False)

    # 将数据转换为 DataLoader 格式
    train_dataset = torch.utils.data.TensorDataset(torch.tensor(X_train_combined_all, dtype=torch.float32), torch.tensor(y_train_combined_all, dtype=torch.float32))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


    # 实例化模型
    model = AutoencoderWithTransformer(input_dim=input_dim, encoding_dim=encoding_dim, d_model=d_model, nhead=nhead,
                                       num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers,
                                       dim_feedforward=dim_feedforward, dropout=dropout, mask_num=mask_num)
    model.to(device)

    # 定义优化器并添加L2正则化
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

    # 引入学习率调度器
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

    # 训练每一折
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_classification_loss = 0.0
        total_reconstruction_loss1 = 0.0
        total_reconstruction_loss2 = 0.0

        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            optimizer.zero_grad()

            # 前向传播
            logits, reconstructed1, reconstructed2 = model(batch_X)
            
            
            
            # RNA的前一半和甲基化的前一半
            original_part1 = torch.cat((batch_X[:, :model.rna_split_point], 
                                        batch_X[:, input_dim // 2:input_dim // 2 + model.methylation_split_point]), dim=1)

            # RNA的剩余部分和甲基化的剩余部分
            original_part2 = torch.cat((batch_X[:, model.rna_split_point:input_dim // 2], 
                                        batch_X[:, input_dim // 2 + model.methylation_split_point:]), dim=1)

            # 计算损失
            classification_loss = classification_loss_fn(logits, batch_y.unsqueeze(1))
            reconstruction_loss1 = reconstruction_loss_fn(reconstructed1, original_part1)
            reconstruction_loss2 = reconstruction_loss_fn(reconstructed2, original_part2)

            # 总损失 = 分类损失 + 两个重构损失
            loss = classification_loss + reconstruction_loss1 + reconstruction_loss2

            # 反向传播和优化
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_classification_loss += classification_loss.item()
            total_reconstruction_loss1 += reconstruction_loss1.item()
            total_reconstruction_loss2 += reconstruction_loss2.item()

        avg_loss = total_loss / len(train_loader)
        avg_classification_loss = total_classification_loss / len(train_loader)
        avg_reconstruction_loss1 = total_reconstruction_loss1 / len(train_loader)
        avg_reconstruction_loss2 = total_reconstruction_loss2 / len(train_loader)

        print(f'Epoch [{epoch+1}/{epochs}],  Classification Loss: {avg_classification_loss:.4f}, '
              f'Reconstruction Loss 1: {avg_reconstruction_loss1:.4f}, Reconstruction Loss 2: {avg_reconstruction_loss2:.4f}, Loss: {avg_loss:.4f}')





    # 测试集评估并计算AUC
    print("\nTesting on total test set:")
    model.eval()
    y_true = []
    y_pred = []
    y_scores = []

    with torch.no_grad():
        for i in range(len(X_test)):
            batch_X = X_test[i].unsqueeze(0).to(device)  # 添加维度以匹配批次大小
            batch_y = y_test[i].unsqueeze(0).to(device)  # 添加维度以匹配批次大小

            logits, _, _ = model(batch_X)
            predicted = (logits > 0.5).float()

            y_true.append(batch_y.item())
            y_pred.append(predicted.item())
            y_scores.append(logits.item())

    # 计算测试集评估指标
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc_value = roc_auc_score(y_true, y_scores)


    test_resultdata = {
        "Evaluation indicators": ["Accuracy", "Precision", "Recall", "F1-Score", "AUC"],
        "Value": [round(accuracy, 4), round(precision, 4), round(recall, 4), round(f1, 4), round(auc_value, 4)]
        }

    test_df = pd.DataFrame(test_resultdata)


    # test_df.to_csv("../result/test_resultdata.csv", index=False)

    print(f'Test Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}, AUC: {auc_value:.4f}')
    
    
    # # 提取特征名称（从第三列开始）
    # rna_features = rna_data.columns[2:]  # RNA特征名称从第三列开始
    # methylation_features = methylation_data.columns[2:]  # 甲基化特征名称从第三列开始
    
    # # 合并 RNA 和 甲基化数据的特征名称
    # selected_features = list(rna_features) + list(methylation_features)
    
    # # 计算测试集的平均积分梯度
    # avg_grads = get_average_integrated_gradients(model, X_test, device)
    
    # top_all_features, top_rna_features, top_methylation_features = get_sorted_features(avg_grads, selected_features, 1600)
    
    # top_all_features.to_csv("../result/top_features.csv", index=False)
    # top_rna_features.to_csv("../result/rna_top_features.csv", index=False)
    # top_methylation_features.to_csv("../result/methylation_top_features.csv", index=False)
    
    return test_df
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Autoencoder with Transformer")
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs for training')
    parser.add_argument('--file_path_gene', type=str, required=True, help='Path to the gene data file')
    parser.add_argument('--file_path_methy', type=str, required=True, help='Path to the methylation data file')

    args = parser.parse_args()
    result = main(args.epochs, args.file_path_gene, args.file_path_methy)
    print(result)

