import os
import torch
import logging
import torch.nn as nn
from model.model import build_model
from utils import get_logger, get_summary_writer
from efficient_kan.src.efficient_kan import KAN
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import numpy as np
from datetime import datetime
# from kan import *

# class MultiBaseline(nn.Module):
#     def __init__(self, code_length=12, num_models=4, num_classes=200, att_mode=3, backbone_name='resnet50', device='cpu', pretrained=True):
#         super(MultiBaseline, self).__init__()
#         self.models = nn.ModuleList()
#         self.num_bits = code_length // num_models
#         self.num_models = num_models
#         self.iter_num = 0
#         self.step_size = 20000
#         self.init_scale = 1
#         self.att_mode = att_mode
#         self.fc = nn.Linear(code_length, num_classes, bias=False)
#         if att_mode ==3: # The activation for HashNet
#             self.gamma = self.init_scale
#         elif att_mode==2: # Our Adaptanh
#             self.gamma = nn.Parameter(torch.ones(1), requires_grad=True)
#         elif att_mode==1: # Original Tanh
#             self.gamma = 1
        
#         for i in range(self.num_models):
#             if backbone_name=="resnet50":
#                 backbone = resnet50(weights="DEFAULT") #pretrained=pretrained
#                 backbone.fc = nn.Sequential(
#                     nn.Linear(2048, self.num_bits),
#                 )
#             elif backbone_name=="resnet18":
#                 backbone = resnet18(pretrained=pretrained)
#                 backbone.fc = nn.Sequential(
#                     nn.Linear(512, self.num_bits),
#                 )
#             elif backbone_name=="resnet34":
#                 backbone = resnet34(pretrained=pretrained)
#                 backbone.fc = nn.Sequential(
#                     nn.Linear(512, self.num_bits),
#                 )
#             self.models.append(backbone)
#         self.device = device

#     def get_gamma(self):
#         return self.gamma

#     def forward(self, x, targets):
#         out = []
#         if self.training:
#             self.iter_num += 1
#         if self.att_mode == 3: 
#             if self.iter_num % self.step_size == 0:
#                 self.gamma = self.init_scale * (math.pow((1. + 0.00005*self.iter_num), 0.5))
#         for i in range(self.num_models):
#             output = self.models[i](x)

#             out.append(output)
        
#         ret = torch.cat(out, dim=1)
#         ret = self.gamma * ret
#         ret = torch.tanh(ret)
#         y = self.fc(ret)
#         return ret, y

# class LinearHash(nn.Module):
#     def __init__(self, inputDim=2048, outputDim=64):
#         super(LinearHash, self).__init__()
#         self.fc = nn.Linear(inputDim, outputDim)
#         self.drop_out = nn.Dropout(p=0.2)

#     def forward(self, data):
#         # 1. 保存原始输入（降维前）
#         original_input = data  
        
#         # 2. 线性层输出（降维后未激活）
#         linear_output = self.fc(data)  
        
#         # 3. 最终输出（激活后）
#         dropped = self.drop_out(linear_output)
#         activated_output = dropped
        
#         return original_input, linear_output, activated_output

class LinearHash(nn.Module):
    def __init__(self, inputDim=2048, outputDim=64):
        super(LinearHash, self).__init__()
        self.fc1 = nn.Linear(inputDim, 1024)
        
        self.fc2 = nn.Linear(1024, outputDim)
        self.drop_out = nn.Dropout(p=0.2)

    def forward(self, data):
        # 1. 保存原始输入（降维前）
        original_input = data  
        
        # 2. 线性层输出（降维后未激活）
        linear_output = self.fc1(data)  
        data= self.fc2(linear_output)
        
        # 3. 最终输出（激活后）
        dropped = self.drop_out(data)
        activated_output = torch.tanh(dropped)
        
        return original_input, linear_output, activated_output

class kanHash(nn.Module):
    def __init__(self, outputDim=64):
        super(kanHash, self).__init__()
        layers_hidden = [512, 1024, outputDim]
        self.kan = KAN(layers_hidden, base_activation=nn.Identity)
        self.drop_out = nn.Dropout(p=0.2)
    
    def forward(self, data):
        # 返回三个阶段的特征：
        # 1. 输入数据 (降维前)
        # 2. KAN输出 (降维后未激活)
        # 3. 最终输出 (激活后)
        kan_output = self.kan(data)
        activated = torch.tanh(self.drop_out(kan_output))
        return data, kan_output, activated

class DHaPH(nn.Module):

    def __init__(self,
                 outputDim=128,
                 clipPath="/home/iot/space5/shard_pool_data5/DHaPH-main/RN50.pt",
                 writer=None,
                 saveDir="./result/log",
                 logger: logging.Logger = None,
                 is_train=True):
        super(DHaPH, self).__init__()
        os.makedirs(saveDir, exist_ok=True)
        self.logger = logger if logger is not None else get_logger(
            os.path.join(saveDir, "train.log" if is_train else "test.log"))
        self.writer = writer if writer is not None and is_train else get_summary_writer(
            os.path.join(saveDir, "tensorboard"))
        embedDim, self.clip = self.load_clip(clipPath)
        # efficient kan
        # layers_hidden = [64, 512, outputDim]
        # self.textkan = kanHash(outputDim)
        # self.imagekan = kanHash(outputDim)
        # pykan
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.kan = KAN(width=[512,256,64], grid=3, k=3, seed=42, device=device)
        self.imagekan = LinearHash(inputDim=embedDim, outputDim=outputDim)
        self.textkan = LinearHash(inputDim=embedDim, outputDim=outputDim)

    def load_clip(self, clipPath: str) -> tuple:
        try:
            model = torch.jit.load(clipPath, map_location="cpu").eval()
            state_dict = model.state_dict()
        except RuntimeError:
            state_dict = torch.load(clipPath, map_location="cpu")

        return state_dict["text_projection"].shape[1], build_model(state_dict)

    # def encode_image(self, image):
    #     # print(image.shape)
    #     image_embed = self.clip.encode_image(image)  # 512
    #     # image_embed = self.image_hash(image_embed)
    #     # print(image_embed.shape)
    #     image_embed = self.imagekan(image_embed)
    #     return image_embed

    def eval(self):
        # self.image_hash.eval()
        # self.text_hash.eval()
        self.textkan.eval()
        self.imagekan.eval()
        # self.clip.eval()

    def train(self, mode=True):  
        super().train(mode)      
        self.textkan.train(mode)
        self.imagekan.train(mode)
        return self  

    def encode_image(self, image):
        image_embed = self.clip.encode_image(image)
        return self.imagekan(image_embed)  # 返回元组 (原始, KAN输出, 激活后)
    
    def encode_text(self, text):
        text_embed = self.clip.encode_text(text)
        return self.textkan(text_embed)  # 返回元组 (原始, KAN输出, 激活后)
    
    def forward(self, image, text):
        image_intermediate = self.encode_image(image)
        text_intermediate = self.encode_text(text)
        # 返回各阶段特征和最终哈希值
        return image_intermediate, text_intermediate, image_intermediate[2], text_intermediate[2]

    
    def _get_preactivation(self, modality='image'):
        """获取tanh激活前的原始特征"""
        def hook(module, input, output):
            self.preactivation = input[0].detach().cpu()
        return hook
    
    def visualize_manifolds(self, image_data, text_data,epoch,index, method='tsne'):
        # 注册前向钩子
        image_handle = self.imagekan.register_forward_hook(self._get_preactivation('image'))
        text_handle = self.textkan.register_forward_hook(self._get_preactivation('text'))
        
        # 前向传播获取特征
        with torch.no_grad():
            _ = self.encode_image(image_data)
            img_features = self.preactivation
            _ = self.encode_text(text_data)
            txt_features = self.preactivation
        
        # 移除钩子
        image_handle.remove()
        text_handle.remove()
        
        # 构建双瑞士卷流形数据
        img_swiss = self._generate_swiss_roll(img_features.numpy(), label=0)
        txt_swiss = self._generate_swiss_roll(txt_features.numpy(), label=1)
        
        # 联合降维可视化
        combined_data = np.concatenate([img_swiss, txt_swiss])
        labels = np.concatenate([np.zeros(len(img_swiss)), np.ones(len(txt_swiss))])
        
        if method == 'tsne':
            projector = TSNE(n_components=2, perplexity=30)
        else:
            projector = PCA(n_components=2)
            
        proj = projector.fit_transform(combined_data)
        
        # 绘制双模态流形
        plt.figure(figsize=(10,8))
        scatter = plt.scatter(proj[:,0], proj[:,1], c=labels, 
                            cmap='jet', alpha=0.6, 
                            edgecolors='w', linewidth=0.5)
        plt.colorbar(scatter, ticks=[0,1], 
                   label='Modality: 0=Image, 1=Text')
        plt.title(f'KAN Pre-Tanh Manifold ({method.upper()})')
        timestamp = datetime.now().strftime("%m%d_%H%M%S")
        plt.savefig("/home/iot/space5/shard_pool_data5/DHaPH-main/swissroll/{}".format(epoch)+"_"+"{}".format(timestamp)+".png")
        plt.close()

    def _generate_swiss_roll(self, features, label):
        """生成与特征空间对齐的瑞士卷"""
        # 用特征主成分初始化瑞士卷基底
        pca = PCA(n_components=3)
        base = pca.fit_transform(features)
        
        # 添加非线性扭曲
        theta = 1.5 * np.pi * (1 + 2*base[:,0])
        noise = 0.05 * np.random.randn(len(features))
        x = theta * np.cos(theta) + 0.1*base[:,1] + noise
        y = theta * np.sin(theta) + 0.1*base[:,2] + noise
        z = 0.3 * base[:,0] + 0.2*base[:,1]
        
        return np.vstack([x, y, z]).T