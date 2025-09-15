from model.hash_model import DHaPH as DHaPH
import os
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import scipy.io as scio
# from emo import EMOLoss
from .base import TrainBase
from model.optimization import BertAdam
from utils import get_args
from utils.calc_utils import calc_map_k_matrix as calc_map_k
from utils.calc_utils import calc_ndcg
from dataset.dataloader import dataloader
import time
from utils.emoloss import supervised_OT_hard1
from model.hp_model import HPmodel
from utils.HPloss import HPLoss
from utils.MSLoss import MSLoss
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
from thop import profile
from fvcore.nn import FlopCountAnalysis, ActivationCountAnalysis
from functools import wraps
from fvcore.nn.jit_handles import get_shape
from scipy.stats import spearmanr
from sklearn.manifold import Isomap
from sklearn.metrics import mean_squared_error
from sklearn.manifold import trustworthiness
import seaborn as sns
from sklearn.decomposition import PCA
from scipy.spatial import procrustes
from sklearn.neighbors import NearestNeighbors
# from cuml.manifold import TSNE
def scaled_dot_product_attention_flops(input_shapes, output_shapes, *args):
    # input_shapes: [q, k, v] 的形状
    q_shape = input_shapes[0]
    k_shape = input_shapes[1]
    # FLOPs = Q@K^T 的计算量 + softmax + V@Attention_weights
    flops = 2 * q_shape[0] * q_shape[1] * k_shape[1]  # QK^T
    flops += q_shape[0] * q_shape[1]  # softmax
    flops += 2 * q_shape[0] * q_shape[1] * k_shape[1]  # AV
    return flops
CUSTOM_HANDLES = {
    "aten::scaled_dot_product_attention": scaled_dot_product_attention_flops,
    "aten::add": lambda *_: 0,  # 加法操作不计入FLOPs
    "aten::layer_norm": lambda *_: 0,  # 层归一化通常不计入
}
def calc_vit_flops(model, input_size=(3,224,224)):
    model.eval()
    dummy_input = torch.randn(1, *input_size)
    flops = FlopCountAnalysis(model, dummy_input)
    flops = flops.set_op_handles(CUSTOM_HANDLES)
    return flops.total()
# class AdaptiveWeights(nn.Module):
#     def __init__(self, init_a=1.0, init_b=0.2, eps=1e-6):
#         super().__init__()
#         # 使用对数参数保证正数
#         self.log_a = nn.Parameter(torch.log(torch.tensor(init_a)))
#         self.log_b = nn.Parameter(torch.log(torch.tensor(init_b)))
#         self.eps = eps

#     @property
#     def a(self):
#         return torch.exp(self.log_a) + self.eps

#     @property
#     def b(self):
#         return torch.exp(self.log_b) + self.eps

class AdaptiveParams(nn.Module):
    def __init__(self):
        super().__init__()
        # 直接定义原始参数
        self.a = nn.Parameter(torch.tensor(1.0))  # 初始值1.0
        self.b = nn.Parameter(torch.tensor(2.2360))


class Trainer(TrainBase):

    def __init__(self,
                 rank=0):
        args = get_args()
        super(Trainer, self).__init__(args, rank)
        self.logger.info("dataset len: {}".format(
            len(self.train_loader.dataset)))
        self.run()

    def _init_model(self):
        self.logger.info("init model.")
        HashModel = DHaPH
        self.model = HashModel(outputDim=self.args.output_dim, clipPath=self.args.clip_path,
                               writer=self.writer, logger=self.logger, is_train=self.args.is_train).to(self.rank)

        if self.args.pretrained != "" and os.path.exists(self.args.pretrained):
            self.logger.info("load pretrained model.")
            self.model.load_state_dict(torch.load(
                self.args.pretrained, map_location=f"cuda:{self.rank}"))

        self.model.float()
        
        # coco
        # self.optimizer = BertAdam([
        #             {'params': self.model.clip.parameters(), 'lr': self.args.clip_lr},
        #             {'params': self.model.image_hash.parameters(), 'lr': self.args.lr},
        #             {'params': self.model.text_hash.parameters(), 'lr': self.args.lr}
        #             ], lr=self.args.lr, warmup=self.args.warmup_proportion, schedule='warmup_cosine',
        #             b1=0.9, b2=0.98, e=1e-6, t_total=len(self.train_loader) * self.args.epochs,
        #             weight_decay=self.args.weight_decay, max_grad_norm=1.0)
        # filckr25k
        self.optimizer = BertAdam([
            {'params': self.model.clip.parameters(), 'lr': self.args.clip_lr}
            # {'params': self.model.image_hash.parameters(), 'lr': self.args.lr},
            # {'params': self.model.text_hash.parameters(), 'lr': self.args.lr}
            # {'params': self.model.textkan.parameters(), 'lr': self.args.lr},
            # {'params': self.model.imagekan.parameters(), 'lr': self.args.lr}
        ], lr=self.args.lr, warmup=self.args.warmup_proportion, schedule='warmup_cosine',
            b1=0.9, b2=0.98, e=1e-6, t_total=len(self.train_loader) * self.args.epochs,
            weight_decay=self.args.weight_decay, max_grad_norm=1.0)
        # #kan-coco adamw3e-4
        # self.optimizer = BertAdam([
        #             {'params': self.model.clip.parameters(), 'lr': self.args.clip_lr},
        #             # {'params': self.model.image_hash.parameters(), 'lr': self.args.lr},
        #             # {'params': self.model.text_hash.parameters(), 'lr': self.args.lr}
        #             ], lr=self.args.lr, warmup=self.args.warmup_proportion, schedule='warmup_cosine',
        #             b1=0.9, b2=0.98, e=1e-6, t_total=len(self.train_loader) * self.args.epochs,
        #             weight_decay=self.args.weight_decay, max_grad_norm=1.0)

        # self.hpmodel = HPmodel(self.args.output_dim,
        #                        self.args.output_dim).to(self.rank)
        # self.optimizer_hpmodel = torch.optim.AdamW(
        #     params=self.hpmodel.parameters(), lr=3e-4)
        self.hp = HPLoss(nb_proxies=self.args.HM,
                         sz_embed=self.args.output_dim, mrg=self.args.margin).to(self.rank)
        self.optimizer_hploss = torch.optim.AdamW(
            params=self.hp.parameters(), lr=3e-4)
        self.optimizer_textKANadam = torch.optim.AdamW(
            params=self.model.textkan.parameters(),lr=3e-4
            # params=self.model.text_hash.parameters(),lr=3e-4
        )
        self.optimizer_imageKANadam = torch.optim.AdamW(
            params=self.model.imagekan.parameters(),lr=3e-4
        )
        
        self.msloss = MSLoss(batch_norm=self.args.batch_norm,temperature=self.args.tau,
                             totalepoch=self.args.epochs, self_paced=True)

        self.total_time = 0.0
        # self.adaptive_weights = AdaptiveWeights().to(self.rank)
        # 将权重参数添加到优化器
        # self.optimizer.add_param_group({'params': self.adaptive_weights.parameters()})
        # adaptive_params = AdaptiveParams().to(self.rank)
        # self.optimizer.add_param_group({'params': adaptive_params.parameters()})
        self.adaptive_params = AdaptiveParams().to(self.rank)
        self.optimizer.add_param_group({'params': self.adaptive_params.parameters()})
        print(self.model)
        def calc_flops(model, image_shape=(3, 224, 224), text_length=77):
            # 图像输入样例
            dummy_image = torch.randn(1, *image_shape).to(self.rank)
            # 文本输入样例（CLIP默认使用77词元长度）
            dummy_text = torch.randint(0, 1000, (1, text_length)).to(self.rank)  
            
            flops, _ = profile(model, inputs=(dummy_image, dummy_text))
            print(f"FLOPs: {flops / 1e9:.2f}G")
    
        calc_flops(self.model)
        def benchmark_inference(model, num_runs=100, warmup=10):
            model.eval()
            # 生成测试数据
            dummy_image = torch.randn(1, 3, 224, 224).to(self.rank)
            dummy_text = torch.randint(0, 128, (1, 24)).to(self.rank)
            
            # Warmup
            for _ in range(warmup):
                _ = model(dummy_image, dummy_text)
            
            # 正式测试
            torch.cuda.synchronize()
            start_time = time.time()
            for _ in range(num_runs):
                with torch.no_grad():
                    _ = model(dummy_image, dummy_text)
            torch.cuda.synchronize()
            
            avg_time = (time.time() - start_time) * 1000 / num_runs
            print(f"Inference Time: {avg_time:.2f}ms per sample")
        benchmark_inference(self.model)
        def print_model_params(model):
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            print(f"Total Parameters: {total_params / 1e6:.2f}M")
            print(f"Trainable Parameters: {trainable_params / 1e6:.2f}M")
        print_model_params(self.model)
        

        def analyze_complexity(model):
            model.eval()
            device = next(model.parameters()).device  # 关键修复
    
            # === 生成与模型同设备的输入数据 ===
            dummy_image = torch.randn(1, 3, 224, 224).to(device)
            dummy_text = torch.randint(0, 128, (1, 24)).to(device)
            
            # === 分析 FLOPs ===
            flops = FlopCountAnalysis(model, (dummy_image, dummy_text)).total()
                    
            # 激活次数分析
            acts = ActivationCountAnalysis(model, (dummy_image, dummy_text)).total()
            
            print(f"FLOPs: {flops / 1e9:.2f}G | Activations: {acts / 1e6:.2f}M")
        analyze_complexity(self.model)
        # def calc_flops_without_modify_model(model):
        #     # 保存原始train方法
        #     original_train = model.train
            
        #     # 定义兼容性train方法
        #     @wraps(original_train)
        #     def patched_train(mode=True):
        #         return original_train(mode)  # 确保参数兼容
            
        #     try:
        #         # 临时替换train方法
        #         model.train = patched_train
                
        #         # 计算FLOPs
        #         dummy_image = torch.randn(1, 3, 224, 224)
        #         dummy_text = torch.randint(0, 1000, (1, 77))
        #         flops, _ = profile(model, inputs=(dummy_image, dummy_text))
        #         print(f"FLOPs: {flops / 1e9:.2f}G")
                
        #     finally:
        #         # 恢复原始train方法
        #         model.train = original_train

        # # 在模型初始化后调用
        # calc_flops_without_modify_model(self.model)
            
    def _init_dataset(self):
        self.logger.info("init dataset.")
        self.logger.info(f"Using {self.args.dataset} dataset.")
        self.args.index_file = os.path.join(
            "./dataset", self.args.dataset, self.args.index_file)
        self.args.caption_file = os.path.join(
            "./dataset", self.args.dataset, self.args.caption_file)
        self.args.label_file = os.path.join(
            "./dataset", self.args.dataset, self.args.label_file)
        train_data, query_data, retrieval_data = dataloader(captionFile=self.args.caption_file,
                                                            indexFile=self.args.index_file,
                                                            labelFile=self.args.label_file,
                                                            maxWords=self.args.max_words,
                                                            imageResolution=self.args.resolution,
                                                            query_num=self.args.query_num,
                                                            train_num=self.args.train_num,
                                                            seed=self.args.seed)
        self.train_labels = train_data.get_all_label().to(0)
        self.query_labels = query_data.get_all_label()
        self.retrieval_labels = retrieval_data.get_all_label()
        self.args.retrieval_num = len(self.retrieval_labels)
        self.logger.info(f"query shape: {self.query_labels.shape}")
        self.logger.info(f"retrieval shape: {self.retrieval_labels.shape}")
        self.train_loader = DataLoader(
            dataset=train_data,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,

            pin_memory=True,
            shuffle=True
        )
        self.query_loader = DataLoader(
            dataset=query_data,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            pin_memory=True,
            shuffle=True
        )
        self.retrieval_loader = DataLoader(
            dataset=retrieval_data,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            pin_memory=True,
            shuffle=True
        )
   
    def train_epoch(self, epoch):
        self.change_state(mode="train")
        self.logger.info(f">>>>>>> epochs: {epoch}/{self.args.epochs}")
        
        
        def plot_joint_manifold(img_feat, txt_feat, labels, epoch):
            """双模态联合流形可视化"""
            def to_cpu_numpy(tensor):
                if isinstance(tensor, torch.Tensor):
                    return tensor.detach().cpu().numpy()
                return tensor
            if labels.ndim == 2:
                labels = np.argmax(labels, axis=1)  # 假设one-hot编码
            assert img_feat is not None, "Image features cannot be None"
            assert txt_feat is not None, "Text features cannot be None"
            assert labels is not None, "Labels cannot be None"
            img_feat = to_cpu_numpy(img_feat)
            txt_feat = to_cpu_numpy(txt_feat)
            labels = to_cpu_numpy(labels)
                    # 合并特征
            combined_feat = np.concatenate([img_feat, txt_feat], axis=0)
            combined_labels = np.concatenate([labels, labels], axis=0)
            modality = np.array(['Image']*len(img_feat) + ['Text']*len(txt_feat))
            
            # 降维
            tsne = TSNE(n_components=2, perplexity=30, metric='cosine')
            embeddings = tsne.fit_transform(combined_feat)
            
            # 可视化
            plt.figure(figsize=(18,6))
            
            # 模态对比子图
            plt.subplot(121)
            sns.scatterplot(x=embeddings[:,0], y=embeddings[:,1], 
                            hue=modality, palette=['#87CEFA', '#FFD700'], 
                            alpha=0.7, s=24, edgecolor='w')
            plt.title(f'Modality Distribution (Epoch {epoch})')
            
            # 类别对比子图
            plt.subplot(122)
            scatter = plt.scatter(embeddings[:,0], embeddings[:,1], 
                                c=combined_labels, cmap='Spectral', 
                                alpha=0.6, s=24, edgecolor='k')
            plt.colorbar(scatter, label='Class ID')
            plt.title(f'Class Distribution (Epoch {epoch})')
            
            plt.savefig(f'joint_manifold_epoch{epoch}.png', dpi=300, bbox_inches='tight')
            plt.close()
        # def track_manifold_evolution(train_loader):
        #     """流形结构演化追踪"""
        #     # 初始化记录器
        #     evolution_data = {
        #         'epoch': [],
        #         'image_trust': [],
        #         'text_trust': [],
        #         'joint_continuity': []
        #     }
            
        #     for epoch in range(epoch):
        #         # 训练并提取特征
        #         img_feat, txt_feat, labels = train_epoch(epoch)
                
        #         # 计算流形指标
        #         img_trust = trustworthiness(img_feat, TSNE(n_components=2).fit_transform(img_feat))
        #         txt_trust = trustworthiness(txt_feat, TSNE(n_components=2).fit_transform(txt_feat))
        #         joint_cont = continuity_score(img_feat, txt_feat)
                
        #         # 记录指标
        #         evolution_data['epoch'].append(epoch)
        #         evolution_data['image_trust'].append(img_trust)
        #         evolution_data['text_trust'].append(txt_trust)
        #         evolution_data['joint_continuity'].append(joint_cont)

        #     # 绘制演化曲线
        #     plt.figure(figsize=(12,6))
        #     plt.plot(evolution_data['epoch'], evolution_data['image_trust'], label='Image Trustworthiness')
        #     plt.plot(evolution_data['epoch'], evolution_data['text_trust'], label='Text Trustworthiness')
        #     plt.plot(evolution_data['epoch'], evolution_data['joint_continuity'], label='Cross-Modal Continuity')
        #     plt.xlabel('Epoch')
        #     plt.ylabel('Metric Value')
        #     plt.legend()
        #     plt.savefig('manifold_evolution.png', dpi=300)    
        #     plt.close   
        def cross_manifold_alignment(img_feat, txt_feat):
            """计算跨模态流形对齐度"""
            # 降维到相同维度
            pca = PCA(n_components=min(img_feat.shape[1], txt_feat.shape[1]))
            img_pca = pca.fit_transform(img_feat)
            txt_pca = pca.fit_transform(txt_feat)
            
            # 计算Procrustes距离
            _, _, disparity = procrustes(img_pca, txt_pca)
            return 1 / (1 + disparity) 
        # def local_geometry_preservation(original, embedded):
        #     """计算局部邻域保持率"""
        #     nbrs_orig = NearestNeighbors(n_neighbors=10).fit(original)
        #     distances_orig, indices_orig = nbrs_orig.kneighbors(original)
            
        #     nbrs_emb = NearestNeighbors(n_neighbors=10).fit(embedded)
        #     distances_emb, indices_emb = nbrs_emb.kneighbors(embedded)
            
        #     overlap = 0
        #     for i in range(len(original)):
        #         overlap += len(np.intersect1d(indices_orig[i], indices_emb[i]))
        #     return overlap / (len(original) * 10)
        all_loss = 0.0
        total_batches = len(self.train_loader)
        
        # 新增：数据收集容器
        img_features = []
        text_features = []
        all_labels = []
   
        all_features = {
            'img_pre': [], 'img_mid': [], 'img_post': [],
            'txt_pre': [], 'txt_mid': [], 'txt_post': [],
            'labels': []
        }
        img_pre_features = []   # 降维前的图像特征
        txt_pre_features = []   # 降维前的文本特征
        img_post_features = []  # 降维后的图像特征
        txt_post_features = []  # 降维后的文本特征
        progress_bar = tqdm(total=total_batches, desc="Training", unit="batch")
        for image, text, label, index in self.train_loader:
            start_time = time.time()
            image.float()
            image = image.to(self.rank, non_blocking=True)
            text = text.to(self.rank, non_blocking=True)
            label = label.to(self.rank, non_blocking=True)
            label = label.float()
            # print(epoch,index)
            self.model.visualize_manifolds(image, text,epoch,index, method='tsne')
            # hash_img, hash_text = self.model(image, text)
            img_inter, txt_inter, hash_img, hash_text = self.model(image, text)
            
            batch_norm=self.args.batch_norm
            loss1 = self.msloss(batch_norm,hash_img, hash_img, label, epoch + 1)
            loss2 = self.msloss(batch_norm,hash_text, hash_text, label, epoch + 1)
            loss3 = self.msloss(batch_norm,hash_img, hash_text, label, epoch + 1)
            cout= self.args.count
            reg = self.args.reg
            supervised_loss = supervised_OT_hard1(
                hash_img, hash_text, cout,reg,
                labels=label, 
                 tau_plus=0, kappa=1,new_cost=False
            )
            loss = loss1 + loss2 + loss3 + supervised_loss * 0.2
            # loss = loss1 + loss2 + loss3
            all_loss += loss
            
            #tsne特征
            img_pre_kan, img_kan_out, img_activated = img_inter
            txt_pre_kan, txt_kan_out, txt_activated = txt_inter
            all_features['img_pre'].append(img_pre_kan.detach().cpu())
            all_features['img_mid'].append(img_kan_out.detach().cpu())
            all_features['img_post'].append(img_activated.detach().cpu())
            all_features['txt_pre'].append(txt_pre_kan.detach().cpu())
            all_features['txt_mid'].append(txt_kan_out.detach().cpu())
            all_features['txt_post'].append(txt_activated.detach().cpu())
            all_features['labels'].append(label.cpu())
            img_pre_features.append(img_pre_kan.detach().cpu())
            txt_pre_features.append(txt_pre_kan.detach().cpu())
            img_post_features.append(img_activated.detach().cpu())
            txt_post_features.append(txt_activated.detach().cpu())
            
            # 收集数据用于tsne可视化
            with torch.no_grad():
                img_features.append(hash_img.cpu())
                text_features.append(hash_text.cpu())
                all_labels.append(label.cpu())
            
            self.optimizer.zero_grad()
            # self.optimizer_hpmodel.zero_grad()
            self.optimizer_hploss.zero_grad()
            self.optimizer_textKANadam.zero_grad()
            self.optimizer_imageKANadam.zero_grad()
            loss.backward()
            self.optimizer.step()
            # self.optimizer_hpmodel.step()
            self.optimizer_hploss.step()
            self.optimizer_textKANadam.step()
            self.optimizer_imageKANadam.step()
            self.total_time += time.time() - start_time
            progress_bar.update(1)
            progress_bar.set_postfix({
                "Loss": loss.item(),
                "a": self.adaptive_params.a.item(),
                "b": self.adaptive_params.b.item()
            })
        img_feat = torch.cat(img_features, dim=0)
        txt_feat = torch.cat(text_features, dim=0)
        lbl = torch.cat(all_labels, dim=0)
        if epoch % 5 == 0:  # 每5个epoch执行一次深度分析
            # 流形可视化
            plot_joint_manifold(img_feat, txt_feat,lbl, epoch)
            
            # 计算量化指标
            alignment_score = cross_manifold_alignment(img_feat, txt_feat)
            # img_trust = trustworthiness(img_feat, TSNE().fit_transform(img_feat))
            
            # 记录到TensorBoard
            self.writer.add_scalar('Manifold/Alignment', alignment_score, epoch)
            # self.writer.add_scalar('Manifold/ImageTrust', img_trust, epoch)
        # 新增：TSNE可视化（每个epoch执行）
        # import matplotlib
        # matplotlib.use('Agg')
        # import matplotlib.pyplot as plt
        # from sklearn.manifold import TSNE
        # import numpy as np
        # import os

        # # 合并数据
        # img_features = torch.cat(img_features).numpy()
        # text_features = torch.cat(text_features).numpy()
        # labels = torch.cat(all_labels).numpy().argmax(1)

        # # 只采样固定的几个label（例如前5类）
        # # selected_labels = [ 4,9,15,16,22,17,23,14,21,5,6,7]  # 可调整
        # selected_labels = [ 4,9,15,16,22,17,23,28,21]  # 可调整
        # mask = np.isin(labels, selected_labels)
        # img_features = img_features[mask]
        # text_features = text_features[mask]
        # labels = labels[mask]

        # # 配置TSNE
        # tsne = TSNE(
        #     n_components=2,
        #     perplexity=min(30, len(labels) // 3),  # 动态调整
        #     n_iter=1000,
        #     learning_rate=200,
        #     init='pca',
        #     random_state=42,
        #     metric='cosine'
        # )

        # 可视化函数（使用高分辨率颜色）
        # def plot_tsne(features, labels, title, save_name):
        #     embeddings = tsne.fit_transform(features)
            
        #     plt.figure(figsize=(16, 12))  # 增大画布尺寸
        #     scatter = plt.scatter(
        #         embeddings[:, 0], embeddings[:, 1],
        #         c=labels, 
        #         cmap='gist_rainbow',  # 改用彩虹色系
        #         alpha=0.85,           # 适当降低透明度
        #         s=150,                 # 增大点尺寸
        #         # edgecolor='w',        # 白色描边
        #         # linewidth=0.4,        # 描边粗细
        #         marker='o'            # 使用实心圆
        #     )
            
        #     # 增强颜色条显示
        #     cbar = plt.colorbar(scatter, ticks=selected_labels, spacing='proportional')
        #     cbar.set_label('Class Label', rotation=270, labelpad=20, fontsize=12)
                    
        #     plt.title(title, fontsize=14, pad=20)
        #     plt.xlabel('TSNE Dimension 1', fontsize=12)
        #     plt.ylabel('TSNE Dimension 2', fontsize=12)
        #     plt.grid(True, linestyle='--', alpha=0.3)
            
        #     save_path = os.path.join(self.args.save_dir, f"epoch_{epoch}_{save_name}")
        #     plt.savefig(save_path, dpi=350, bbox_inches='tight')  # 提高DPI至350
        #     plt.close()
        # #     self.logger.info(f"TSNE可视化已保存至: {save_path}")
        # def plot_tsne(features, labels, title, save_name):
        #     embeddings = tsne.fit_transform(features)
            
        #     plt.figure(figsize=(16, 12))
        #     # 修改颜色映射部分
        #     fixed_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
        #                 '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22']  # 9种固定颜色
        #     color_indices = np.array([np.where(selected_labels == l)[0][0] for l in labels])
            
        #     scatter = plt.scatter(
        #         embeddings[:, 0], embeddings[:, 1],
        #         c=color_indices,  # 使用索引映射颜色
        #         cmap=matplotlib.colors.ListedColormap(fixed_colors),
        #         alpha=0.85,
        #         s=200,
        #         marker='o'
        #     )
            
        #     # 修改图例显示方式
        #     legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label=f'Class {l}',
        #                         markerfacecolor=fixed_colors[i], markersize=10) 
        #                         for i, l in enumerate(selected_labels)]
        #     plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')
            
        #     plt.title(title, fontsize=14, pad=20)
        #     plt.xlabel('TSNE Dimension 1', fontsize=12)
        #     plt.ylabel('TSNE Dimension 2', fontsize=12)
        #     plt.grid(True, linestyle='--', alpha=0.3)
    
        #     save_path = os.path.join(self.args.save_dir, f"epoch_{epoch}_{save_name}")
        #     plt.savefig(save_path, dpi=350, bbox_inches='tight')
        #     plt.close()

        # # 绘制图像模态和文本模态的TSNE
        # plot_tsne(
        #     img_features, labels,
        #     f"Image Modality TSNE (Epoch {epoch}) - Selected Classes",
        #     "image_tsne_highres.png"
        # )
        # plot_tsne(
        #     text_features, labels,
        #     f"Text Modality TSNE (Epoch {epoch}) - Selected Classes",
        #     "text_tsne_highres.png"
        # )

        for key in all_features:
            if key != 'labels':
                all_features[key] = torch.cat(all_features[key])
            else:
                all_features[key] = torch.cat(all_features['labels'])
    #     self.generate_3d_tsne(all_features, epoch)
    #     # self.plot_joint_3d_tsne(img_feat, txt_feat, epoch)
    #     img_pre = torch.cat(img_pre_features, dim=0)
    #     txt_pre = torch.cat(txt_pre_features, dim=0)
    #     img_post = torch.cat(img_post_features, dim=0)
    #     txt_post = torch.cat(txt_post_features, dim=0)
    #     self.plot_2d_tsne_comparison(
    #     img_pre, txt_pre, 
    #     img_post, txt_post, 
    #     epoch
    # )
    #     self.plot_joint_3d_tsne_comparison(
    #     img_pre, txt_pre, 
    #     img_post, txt_post, 
    #     epoch
    # )
        self.logger.info(
            f">>>>>> [{epoch}/{self.args.epochs}] Avg Loss: {all_loss.data / total_batches:.4f}, Time: {self.total_time:.2f}s"
        )
        progress_bar.close()
    #全部可视化
    # def train_epoch(self, epoch):
    #     self.change_state(mode="train")
    #     self.logger.info(f">>>>>>> epochs: {epoch}/{self.args.epochs}")
        
    #     all_loss = 0.0
    #     total_batches = len(self.train_loader)
        
    #     # 新增：数据收集容器
    #     img_features = []
    #     text_features = []
    #     all_labels = []
        
    #     progress_bar = tqdm(total=total_batches, desc="Training", unit="batch")
    #     for image, text, label, index in self.train_loader:
    #         start_time = time.time()
    #         image.float()

    #         image = image.to(self.rank, non_blocking=True)
    #         text = text.to(self.rank, non_blocking=True)
    #         label = label.to(self.rank, non_blocking=True)
    #         label = label.float()

    #         hash_img, hash_text = self.model(image, text)

    #         loss1 = self.msloss(hash_img, hash_img, label, epoch + 1)
    #         loss2 = self.msloss(hash_text, hash_text, label, epoch + 1)
    #         loss3 = self.msloss(hash_img, hash_text, label, epoch + 1)
    #         supervised_loss = supervised_OT_hard1(
    #                 hash_img, hash_text, 
    #                 labels=label,
    #                 reg=10000, tau_plus=0, kappa=1, new_cost=False
    #             )
    #         loss = loss1 + loss2 + loss3 + supervised_loss*0.2
            
    #         all_loss += loss
            
    #         # 收集数据用于tsne可视化（每个epoch都收集）
    #         with torch.no_grad():
    #             img_features.append(hash_img.cpu())
    #             text_features.append(hash_text.cpu())
    #             all_labels.append(label.cpu())
            
    #         self.optimizer.zero_grad()
    #         self.optimizer_hpmodel.zero_grad()
    #         self.optimizer_hploss.zero_grad()
    #         self.optimizer_textKANadam.zero_grad()
    #         self.optimizer_imageKANadam.zero_grad()
    #         loss.backward()
    #         self.optimizer.step()
    #         self.optimizer_hpmodel.step()
    #         self.optimizer_hploss.step()
    #         self.optimizer_textKANadam.step()
    #         self.optimizer_imageKANadam.step()
    #         self.total_time += time.time() - start_time
    #         progress_bar.update(1)
    #         progress_bar.set_postfix({
    #             "Loss": loss.item(),
    #             "a": self.adaptive_params.a.item(),
    #             "b": self.adaptive_params.b.item()
    #         })

    #     # 新增：每个epoch都进行t-SNE可视化
    #     import matplotlib
    #     matplotlib.use('Agg')  # 不需要GUI后端
    #     import matplotlib.pyplot as plt
    #     from sklearn.manifold import TSNE
    #     import numpy as np
    #     import os
        
    #     # 准备数据
    #     img_features = torch.cat(img_features).numpy()
    #     text_features = torch.cat(text_features).numpy()
    #     labels = torch.cat(all_labels).numpy().argmax(1)
        
    #     # 配置TSNE
    #     tsne = TSNE(
    #         n_components=2,
    #         perplexity=min(30, len(labels)//2),
    #         n_iter=1000,
    #         learning_rate=200,
    #         init='pca',
    #         random_state=42,  # 固定随机种子以确保可复现
    #         metric='cosine'
    #     )
        
    #     # 可视化函数
    #     def plot_tsne(features, labels, title, save_name):
    #         embeddings = tsne.fit_transform(features)
            
    #         plt.figure(figsize=(12, 10))
    #         scatter = plt.scatter(embeddings[:,0], embeddings[:,1], 
    #                             c=labels, cmap='tab20', alpha=0.8,
    #                             s=15, edgecolor='none')
            
    #         # 添加类别标签
    #         unique_labels = np.unique(labels)
    #         handles = [plt.Line2D([0], [0], marker='o', color=scatter.cmap(scatter.norm(l)), 
    #                             markersize=8, linestyle='', label=str(l)) for l in unique_labels]
    #         plt.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left')
    #         plt.title(title)
            
    #         save_path = os.path.join(self.args.save_dir, f"epoch_{epoch}_{save_name}")
    #         plt.savefig(save_path, dpi=300, bbox_inches='tight')
    #         plt.close()
    #         self.logger.info(f"TSNE可视化已保存至: {save_path}")
        
    #     # 绘制图像模态的TSNE
    #     plot_tsne(img_features, labels, 
    #              f"Image Modality Feature Space Visualization (Epoch {epoch})", 
    #              "image_modality_tsne.png")
        
    #     # 绘制文本模态的TSNE
    #     plot_tsne(text_features, labels, 
    #              f"Text Modality Feature Space Visualization (Epoch {epoch})", 
    #              "text_modality_tsne.png")

    #     self.logger.info(
    #         f">>>>>> [{epoch}/{self.args.epochs}] Avg Loss: {all_loss.data / total_batches:.4f}, Time: {self.total_time:.2f}s"
    #     )
    #     progress_bar.close()
    
    def plot_2d_tsne_comparison(self, img_pre, txt_pre, img_post, txt_post, epoch):
        """
        生成二维 t-SNE 对比可视化，包含降维前和降维后的特征
        蓝色点表示图像特征，黄色点表示文本特征
        
        参数:
            img_pre: 降维前的图像特征 (batch_size, feature_dim)
            txt_pre: 降维前的文本特征 (batch_size, feature_dim)
            img_post: 降维后的图像特征 (batch_size, output_dim)
            txt_post: 降维后的文本特征 (batch_size, output_dim)
            epoch: 当前epoch
        """
        # 创建保存目录
        save_dir = os.path.join(self.args.save_dir, "NODSOT_2D_TSNE_Comparison")
        os.makedirs(save_dir, exist_ok=True)
        
        # 采样1000个点以避免内存问题
        sample_size = min(1000, len(img_pre))
        indices = torch.randperm(len(img_pre))[:sample_size]
        
        # 准备数据
        pre_features = {
            'img': img_pre[indices].cpu().numpy(),
            'txt': txt_pre[indices].cpu().numpy()
        }
        
        post_features = {
            'img': img_post[indices].cpu().numpy(),
            'txt': txt_post[indices].cpu().numpy()
        }
        
        # 创建两个子图：降维前 vs 降维后
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 10))
        
        # 绘制降维前的特征
        self._plot_single_tsne(
            pre_features, 
            ax1, 
            title=f'Pre-Dimensionality Reduction (Epoch {epoch})',
            perplexity=min(30, sample_size//4)  # 动态调整困惑度
        )
        
        # 绘制降维后的特征
        self._plot_single_tsne(
            post_features, 
            ax2, 
            title=f'Post-Dimensionality Reduction (Epoch {epoch})',
            perplexity=min(15, sample_size//8)  # 降维后困惑度可以更小
        )
        
        # 添加整体标题
        fig.suptitle(f'Feature Space Evolution (Epoch {epoch})', fontsize=16)
        
        # 保存图像
        plt.savefig(
            os.path.join(save_dir, f'tsne_comparison_epoch_{epoch}.png'),
            bbox_inches='tight',
            dpi=200
        )
        plt.close()
        self.logger.info(f"Saved 2D t-SNE comparison visualization for epoch {epoch}")

    def _plot_single_tsne(self, features, ax, title, perplexity=30):
        """
        在单个轴上绘制图像和文本特征的联合t-SNE
        
        参数:
            features: 包含 'img' 和 'txt' 特征的字典
            ax: matplotlib轴对象
            title: 图标题
            perplexity: t-SNE困惑度参数
        """
        # 合并特征和创建标签
        img_feat = features['img']
        txt_feat = features['txt']
        
        combined_features = np.vstack([img_feat, txt_feat])
        modality_labels = np.array(
            [0] * len(img_feat) +  # 0 = 图像特征
            [1] * len(txt_feat)    # 1 = 文本特征
        )
        
        # 执行t-SNE降维
        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            n_iter=1000,
            learning_rate=200,
            random_state=42,
            metric='cosine'
        )
        embeddings = tsne.fit_transform(combined_features)
        
        # 分离图像和文本的嵌入点
        img_emb = embeddings[modality_labels == 0]
        txt_emb = embeddings[modality_labels == 1]
        
        # 绘制散点图
        ax.scatter(
            img_emb[:, 0], img_emb[:, 1],
            c='#3399FF', alpha=0.9, s=100, 
            label='Image Features'
        )
        ax.scatter(
            txt_emb[:, 0], txt_emb[:, 1],
            c="#E4C515", alpha=0.9, s=100, 
            label='Text Features'
        )
        
        # 添加标题和图例
        ax.set_title(title, fontsize=30)
        ax.set_xlabel('t-SNE Dimension 1')
        ax.set_ylabel('t-SNE Dimension 2')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.3)

    def plot_joint_3d_tsne_comparison(self, img_pre, txt_pre, img_post, txt_post, epoch):
        """
        生成包含降维前和降维后的联合3D t-SNE对比可视化
        蓝色点表示图像特征，黄色点表示文本特征
        
        参数:
            img_pre: 降维前的图像特征 (n_samples, feature_dim)
            txt_pre: 降维前的文本特征 (n_samples, feature_dim)
            img_post: 降维后的图像特征 (n_samples, output_dim)
            txt_post: 降维后的文本特征 (n_samples, output_dim)
            epoch: 当前epoch
        """
        # 创建保存目录
        save_dir = os.path.join(self.args.save_dir, "NODSOT_3D_TSNE_Comparison")
        os.makedirs(save_dir, exist_ok=True)
        
        # 采样1000个点以避免内存问题
        sample_size = min(1000, len(img_pre))
        indices = torch.randperm(len(img_pre))[:sample_size]
        
        # 准备降维前数据
        pre_features = {
            'img': img_pre[indices].cpu().numpy(),
            'txt': txt_pre[indices].cpu().numpy()
        }
        
        # 准备降维后数据
        post_features = {
            'img': img_post[indices].cpu().numpy(),
            'txt': txt_post[indices].cpu().numpy()
        }
        
        # 创建两个3D子图：降维前 vs 降维后
        fig = plt.figure(figsize=(24, 12))
        
        # 降维前特征可视化
        ax1 = fig.add_subplot(121, projection='3d')
        self._plot_3d_tsne(
            pre_features, 
            ax1, 
            title=f'Pre-Dimensionality Reduction (Epoch {epoch})',
            perplexity=min(30, sample_size//4)  # 动态调整困惑度
        )
        
        # 降维后特征可视化
        ax2 = fig.add_subplot(122, projection='3d')
        self._plot_3d_tsne(
            post_features, 
            ax2, 
            title=f'Post-Dimensionality Reduction (Epoch {epoch})',
            perplexity=min(15, sample_size//8)  # 降维后困惑度可以更小
        )
        
        # 添加整体标题
        fig.suptitle(f'3D Feature Space Evolution (Epoch {epoch})', fontsize=16)
        
        # 保存图像
        plt.savefig(
            os.path.join(save_dir, f'3d_tsne_comparison_epoch_{epoch}.png'),
            bbox_inches='tight',
            dpi=200
        )
        plt.close()
        self.logger.info(f"Saved 3D t-SNE comparison visualization for epoch {epoch}")

    def _plot_3d_tsne(self, features, ax, title, perplexity=30):
        """
        在单个3D轴上绘制图像和文本特征的联合t-SNE
        
        参数:
            features: 包含 'img' 和 'txt' 特征的字典
            ax: matplotlib 3D轴对象
            title: 图标题
            perplexity: t-SNE困惑度参数
        """
        # 合并特征和创建标签
        img_feat = features['img']
        txt_feat = features['txt']
        
        combined_features = np.vstack([img_feat, txt_feat])
        modality_labels = np.array(
            [0] * len(img_feat) +  # 0 = 图像特征
            [1] * len(txt_feat)    # 1 = 文本特征
        )
        
        # 执行t-SNE降维到3维
        tsne = TSNE(
            n_components=3,
            perplexity=perplexity,
            n_iter=1000,
            learning_rate=200,
            random_state=42,
            metric='cosine'
        )
        embeddings = tsne.fit_transform(combined_features)
        
        # 分离图像和文本的嵌入点
        img_emb = embeddings[modality_labels == 0]
        txt_emb = embeddings[modality_labels == 1]
        
        # 绘制图像特征点 (蓝色)
        ax.scatter(
            img_emb[:, 0], img_emb[:, 1], img_emb[:, 2],
            c='#87CEFA',  # 蓝色表示图像特征'#87CEFA', '#FFD700'
            alpha=0.8,
            s=20,      # 点的大小
            label='Image Features'
        )
        
        # 绘制文本特征点 (黄色)
        ax.scatter(
            txt_emb[:, 0], txt_emb[:, 1], txt_emb[:, 2],
            c='#FFD700',  # 黄色表示文本特征
            alpha=0.8,
            s=40,        # 点的大小
            label='Text Features'
        )
        
        # 添加图例和标签
        ax.legend()
        ax.set_xlabel('TSNE Dimension 1')
        ax.set_ylabel('TSNE Dimension 2')
        ax.set_zlabel('TSNE Dimension 3')
        ax.set_title(title, fontsize=14)
        
        # 设置视角
        ax.view_init(elev=30, azim=45)

    def generate_3d_tsne(self, features, epoch):
        """生成三维t-SNE可视化"""
        # 随机采样1000个点 (避免内存问题)
        sample_size = min(1000, len(features['labels']))
        indices = torch.randperm(len(features['labels']))[:sample_size]
        
        # 准备可视化目录
        tsne_dir = os.path.join(self.args.save_dir, "NODSOT_3D_TSNE", f"epoch_{epoch}")
        os.makedirs(tsne_dir, exist_ok=True)
        
        # 对每个特征阶段生成可视化
        stages = {
            'pre': 'Before Hashing',
            'mid': 'After Hashing (Pre-Activation)',
            'post': 'After Hashing (Post-Activation)'
        }
        
        for modality in ['img', 'txt']:
            for stage, desc in stages.items():
                key = f"{modality}_{stage}"
                feat = features[key][indices].numpy()
                labels = features['labels'][indices].numpy().argmax(1)
                
                # 三维t-SNE降维
                tsne = TSNE(n_components=3, perplexity=30, random_state=42)
                embeddings = tsne.fit_transform(feat)
                
                # 创建3D图
                fig = plt.figure(figsize=(12, 10))
                ax = fig.add_subplot(111, projection='3d')
                
                # 按类别着色
                scatter = ax.scatter(
                    embeddings[:, 0], embeddings[:, 1], embeddings[:, 2],
                    c=labels, cmap='viridis', alpha=0.7, s=30
                )
                
                # 添加标签和图例
                plt.colorbar(scatter, label='Class')
                ax.set_title(f'{modality.capitalize()} Features: {desc}\n(Epoch {epoch})')
                ax.set_xlabel('TSNE-1')
                ax.set_ylabel('TSNE-2')
                ax.set_zlabel('TSNE-3')
                
                # 保存图像
                plt.savefig(os.path.join(tsne_dir, f'NODSOT_{key}_3d_tsne.png'), dpi=150)
                plt.close()
        
        self.logger.info(f"Saved 3D t-SNE visualizations for epoch {epoch}")
        
    def cross_manifold_alignment(img_feat, txt_feat):
        """计算跨模态流形对齐度"""
        # 降维到相同维度
        pca = PCA(n_components=min(img_feat.shape[1], txt_feat.shape[1]))
        img_pca = pca.fit_transform(img_feat)
        txt_pca = pca.fit_transform(txt_feat)
        
        # 计算Procrustes距离
        _, _, disparity = procrustes(img_pca, txt_pca)
        return 1 / (1 + disparity)
    def local_geometry_preservation(original, embedded):
        """计算局部邻域保持率"""
        nbrs_orig = NearestNeighbors(n_neighbors=10).fit(original)
        distances_orig, indices_orig = nbrs_orig.kneighbors(original)
        
        nbrs_emb = NearestNeighbors(n_neighbors=10).fit(embedded)
        distances_emb, indices_emb = nbrs_emb.kneighbors(embedded)
        
        overlap = 0
        for i in range(len(original)):
            overlap += len(np.intersect1d(indices_orig[i], indices_emb[i]))
        return overlap / (len(original) * 10)

    def train(self):
        
        self.logger.info("Start train.")

        for epoch in range(self.args.epochs):
            self.train_epoch(epoch)
            self.valid(epoch)
            self.save_model(epoch)

        self.logger.info(
            f">>>>>>> FINISHED >>>>>> Best epoch, I-T: {self.best_epoch_i}, mAP: {self.max_mapi2t}, T-I: {self.best_epoch_t}, mAP: {self.max_mapt2i}")
    
    # original code
    # def get_code(self, data_loader, length: int):
        
        
        
        
    #     img_buffer = torch.empty(
    #         length, self.args.output_dim, dtype=torch.float).to(self.rank)
    #     text_buffer = torch.empty(
    #         length, self.args.output_dim, dtype=torch.float).to(self.rank)
    #     encoder_time = 0
    #     for image, text, label, index in tqdm(data_loader):
    #         start_encoder_time = time.time()
    #         image = image.to(self.rank, non_blocking=True)
    #         text = text.to(self.rank, non_blocking=True)
    #         index = index.numpy()
    #         image_hash = self.model.encode_image(image)
    #         image_hash = torch.sign(image_hash)
    #         text_hash = self.model.encode_text(text)
    #         text_hash = torch.sign(text_hash)
    #         encoder_time = time.time() - start_encoder_time

    #         img_buffer[index, :] = image_hash.data
    #         text_buffer[index, :] = text_hash.data

    #     return img_buffer, text_buffer, encoder_time
    def get_code(self, data_loader, length: int):
        img_buffer = torch.empty(
            length, self.args.output_dim, dtype=torch.float).to(self.rank)
        text_buffer = torch.empty(
            length, self.args.output_dim, dtype=torch.float).to(self.rank)
        encoder_time = 0
        
        # 添加：确保模型在评估模式
        self.model.eval()
        
        for image, text, label, index in tqdm(data_loader):
            start_encoder_time = time.time()
            image = image.to(self.rank, non_blocking=True)
            text = text.to(self.rank, non_blocking=True)
            index = index.numpy()
            
            # 修改：只取最终的哈希值（元组的最后两个元素）
            with torch.no_grad():
                _, _, image_hash, text_hash = self.model(image, text)
            
            image_hash = torch.sign(image_hash)
            text_hash = torch.sign(text_hash)
            encoder_time = time.time() - start_encoder_time

            img_buffer[index, :] = image_hash.data
            text_buffer[index, :] = text_hash.data

        return img_buffer, text_buffer, encoder_time
        
        
    
    
    def test(self, mode_name="i2t"):
        if self.args.pretrained == "":
            raise RuntimeError(
                "test step must load a model! please set the --pretrained argument.")
        self.change_state(mode="valid")
        tsne_save_dir = os.path.join(self.args.save_dir, "TSNE_plots_test")
        os.makedirs(tsne_save_dir, exist_ok=True)
        save_dir = os.path.join(self.args.save_dir, "PR_cruve")
        os.makedirs(save_dir, exist_ok=True)
        query_img, query_txt, q_encoder_time = self.get_code(
            self.query_loader, self.args.query_num)
        retrieval_img, retrieval_txt, r_encoder_time = self.get_code(
            self.retrieval_loader, self.args.retrieval_num)
        
        query_img, query_txt, _ = self.get_code(self.query_loader, self.args.query_num)
        retrieval_img, retrieval_txt, _ = self.get_code(self.retrieval_loader, self.args.retrieval_num)
        
        def plot_tsne1(features, labels, title_suffix=""):
            """
            嵌入式t-SNE可视化函数
            :param features: 特征矩阵 [N, D] (需为CPU numpy数组)
            :param labels: 标签矩阵 [N, C] (需为CPU numpy数组)
            :param title_suffix: 图像标题后缀
            """
            
            # 转换标签为类别索引（取首个非零标签）
            class_indices = np.argmax(labels, axis=1)
            
            # 采样1000个点（若数据量较大）
            sample_size = min(1000, len(features))
            indices = np.random.choice(len(features), sample_size, replace=False)
            sampled_feat = features[indices]
            sampled_labels = class_indices[indices]
            
            # t-SNE降维
            tsne = TSNE(n_components=2, perplexity=30, n_iter=300)
            embeddings = tsne.fit_transform(sampled_feat)
            
            # 可视化
            plt.figure(figsize=(10, 8))
            plt.scatter(embeddings[:, 0], embeddings[:, 1], 
                        c=sampled_labels, cmap='tab20', alpha=0.6, 
                        edgecolors='w', linewidths=0.5)
            plt.title(f"Test t-SNE {title_suffix}")
            plt.colorbar(label='Class')
            plt.axis('off')
            
            # 保存图像
            filename = f"test_tsne_{title_suffix.replace(' ', '_')}.png"
            plt.savefig(os.path.join(tsne_save_dir, filename), 
                    bbox_inches='tight', dpi=150)
            plt.close()
            self.logger.info(f"Saved test t-SNE plot: {filename}")
        
        
        # 执行可视化（确保数据在CPU并转换为numpy）
        with torch.no_grad():
            # 可视化查询集图像特征
            plot_tsne1(query_img.cpu().numpy(), 
                    self.query_labels.numpy(), 
                    "Query Image Features")
            
            # 可视化查询集文本特征
            plot_tsne1(query_txt.cpu().numpy(),
                    self.query_labels.numpy(),
                    "Query Text Features")
            
            # 可视化检索集图像特征（可选）
            plot_tsne1(retrieval_img.cpu().numpy(),
                    self.retrieval_labels.numpy(),
                    "Retrieval Image Features")
        mAPi2t = calc_map_k(query_img, retrieval_txt, self.query_labels,
                            self.retrieval_labels, None, self.rank)
        mAPt2i = calc_map_k(query_txt, retrieval_img, self.query_labels,
                            self.retrieval_labels, None, self.rank)
        mAPi2i = calc_map_k(query_img, retrieval_img, self.query_labels,
                            self.retrieval_labels, None, self.rank)
        mAPt2t = calc_map_k(query_txt, retrieval_txt, self.query_labels,
                            self.retrieval_labels, None, self.rank)
        
        self.max_mapt2i = max(self.max_mapt2i, mAPt2i)
        self.logger.info(
            f">>>>>> MAP(i->t): {mAPi2t}, MAP(t->i): {mAPt2i}, MAP(t->t): {mAPt2t}, MAP(i->i): {mAPi2i}")

        query_img = query_img.cpu().detach().numpy()
        query_txt = query_txt.cpu().detach().numpy()
        retrieval_img = retrieval_img.cpu().detach().numpy()
        retrieval_txt = retrieval_txt.cpu().detach().numpy()
        query_labels = self.query_labels.numpy()
        retrieval_labels = self.retrieval_labels.numpy()
        
        result_dict = {
            'q_img': query_img,
            'q_txt': query_txt,
            'r_img': retrieval_img,
            'r_txt': retrieval_txt,
            'q_l': query_labels,
            'r_l': retrieval_labels
        }
        scio.savemat(os.path.join(save_dir, str(self.args.output_dim) +
                     "-ours-" + self.args.dataset + "-" + mode_name + ".mat"), result_dict)
        self.logger.info(">>>>>> save all data!")

    def plot_tsne(self, features, labels, mode='image', epoch=0):
        # 数据预处理
        feat_np = features.cpu().numpy()
        label_np = labels.cpu().numpy()
        
        # ===== 关键改进1：多标签编码 =====
        # 使用标签的二进制编码作为颜色标识（适用于多标签场景）
        unique_labels = np.unique(label_np, axis=0)
        color_labels = np.array([np.where((unique_labels == row).all(1))[0][0] 
                            for row in label_np])
        
        # ===== 关键改进2：优化采样策略 =====
        # 分层采样确保各类别均衡
        n_samples = min(1000, len(feat_np))  # 增加到1000样本
        indices = []
        for lbl in unique_labels:
            class_mask = (label_np == lbl).all(axis=1)
            class_indices = np.where(class_mask)[0]
            if len(class_indices) > 0:
                n_class = max(1, int(n_samples * len(class_indices)/len(feat_np)))
                indices.extend(np.random.choice(class_indices, n_class, replace=False))
        indices = np.unique(indices)[:n_samples]
        
        # ===== 关键改进3：数据标准化 =====
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        sampled_feat = scaler.fit_transform(feat_np[indices])
        
        # ===== 关键改进4：TSNE参数优化 =====
        tsne_params = {
            'n_components': 2,
            'perplexity': 500,           # 增大以适应数据复杂度
            'early_exaggeration': 30,   # 提升类间分离度
            'learning_rate': 'auto',    # 根据数据量自动调整
            'n_iter': 250,             # 增加迭代次数
            'metric': 'cosine',         # 更适合哈希任务
            'init': 'random',              # 稳定初始化
            'random_state': 42
        }
        tsne = TSNE(**tsne_params)
        embeddings = tsne.fit_transform(sampled_feat)
        
        # ===== 关键改进5：可视化增强 =====
        plt.figure(figsize=(12, 10))
        scatter = plt.scatter(
            embeddings[:, 0], embeddings[:, 1],
            c=color_labels[indices], 
            cmap='gist_ncar',           # 改用高对比度colormap
            alpha=0.7,
            edgecolors='k',             # 添加黑色边框增强区分
            linewidths=0.3,
            s=40                        # 增大点尺寸
        )
        
        # 添加类别标注（随机选取部分类展示）
        unique_colors = np.unique(color_labels[indices])
        if len(unique_colors) <= 20:  # 类别较少时显示图例
            handles = [plt.Line2D([0], [0], marker='o', color='w', 
                                markerfacecolor=scatter.cmap(scatter.norm(c)),
                                markersize=8) for c in unique_colors]
            plt.legend(handles, [f'Class {c}' for c in unique_colors], 
                    loc="upper left", bbox_to_anchor=(1, 1))
        
        plt.title(f'Flickr30K {mode} Features (Epoch {epoch})\nperplexity={tsne_params["perplexity"]}, n_iter={tsne_params["n_iter"]}')
        plt.tight_layout()
        
        # 保存高清图像
        save_dir = os.path.join(self.args.save_dir, "Enhanced_TSNE")
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f'flickr30k_tsne_{mode}_epoch{epoch}.png'), 
                bbox_inches='tight', dpi=300)
        plt.close()

    def plot_tsne_3d(self, features, labels, mode='image', epoch=0):
        # 数据预处理
        feat_np = features.cpu().numpy()
        label_np = labels.cpu().numpy()
        
        # 多标签编码（保持原逻辑）
        unique_labels = np.unique(label_np, axis=0)
        color_labels = np.array([np.where((unique_labels == row).all(1))[0][0] 
                            for row in label_np])
        
        # 分层采样策略（保持原逻辑）
        n_samples = min(1000, len(feat_np))
        indices = []
        for lbl in unique_labels:
            class_mask = (label_np == lbl).all(axis=1)
            class_indices = np.where(class_mask)[0]
            if len(class_indices) > 0:
                n_class = max(1, int(n_samples * len(class_indices)/len(feat_np)))
                indices.extend(np.random.choice(class_indices, n_class, replace=False))
        indices = np.unique(indices)[:n_samples]
        
        # 数据标准化（保持原逻辑）
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        sampled_feat = scaler.fit_transform(feat_np[indices])
        
        # ===== 修改点1：TSNE三维投影 =====
        tsne_params = {
            'n_components': 3,  # 关键修改：输出三维嵌入
            'perplexity': 50,   # 适当减小perplexity
            'early_exaggeration': 24,
            'learning_rate': 200,
            'n_iter': 1000,
            'metric': 'cosine',
            'init': 'pca',      # 使用PCA初始化更稳定
            'random_state': 42
        }
        tsne = TSNE(**tsne_params)
        embeddings = tsne.fit_transform(sampled_feat)
        
        # ===== 修改点2：三维可视化 =====
        fig = plt.figure(figsize=(16, 12))
        ax = fig.add_subplot(111, projection='3d')
        
        # 三维散点图
        scatter = ax.scatter3D(
            embeddings[:, 0], 
            embeddings[:, 1],
            embeddings[:, 2],
            c=color_labels[indices],
            cmap='gist_rainbow',    # 使用更鲜艳的colormap
            alpha=0.7,
            edgecolors='k',
            linewidths=0.2,
            s=40,
            depthshade=True         # 启用深度阴影
        )
        
        # 设置视角参数
        ax.view_init(elev=20, azim=-45)  # 初始视角
        
        # 添加颜色条
        cbar = fig.colorbar(scatter, ax=ax, pad=0.1)
        cbar.set_label('Class ID', rotation=270, labelpad=20)
        
        # 坐标轴设置
        ax.set_xlabel('TSNE-1', labelpad=12)
        ax.set_ylabel('TSNE-2', labelpad=12)
        ax.set_zlabel('TSNE-3', labelpad=12)
        ax.xaxis.pane.fill = False  # 透明背景
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        
        # 添加标题
        plt.title(f'3D T-SNE Visualization of {mode} Features (Epoch {epoch})\n'
                f'Perplexity={tsne_params["perplexity"]}, Iter={tsne_params["n_iter"]}',
                pad=20)
        
        # 保存图像
        save_dir = os.path.join(self.args.save_dir, "NODSOT_3D_TSNE")
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(
            os.path.join(save_dir, f'3d_tsne_{mode}_epoch{epoch}.png'),
            bbox_inches='tight',
            dpi=300,
            transparent=True  # 透明背景
        )
        plt.close()

    # 在valid方法中替换原plot_tsne调用

        
    def valid(self, epoch):
        self.logger.info("Valid.")
        self.change_state(mode="valid")
        query_img, query_txt, q_encoder_time = self.get_code(
            self.query_loader, self.args.query_num)
        # self.plot_tsne_3d(query_img, self.query_labels, mode='image', epoch=epoch)
        # self.plot_tsne_3d(query_txt, self.query_labels, mode='text', epoch=epoch)
        retrieval_img, retrieval_txt, r_encoder_time = self.get_code(
            self.retrieval_loader, self.args.retrieval_num)
        # self.plot_tsne_3d(retrieval_img, self.retrieval_labels, mode='retrieval_image', epoch=epoch)
        # self.plot_tsne_3d(retrieval_txt, self.retrieval_labels, mode='retrieval_text', epoch=epoch)
        mAPi2t = calc_map_k(query_img, retrieval_txt, self.query_labels,
                            self.retrieval_labels, None, self.rank)
        mAPt2i = calc_map_k(query_txt, retrieval_img, self.query_labels,
                            self.retrieval_labels, None, self.rank)
        mAPi2i = calc_map_k(query_img, retrieval_img, self.query_labels,
                            self.retrieval_labels, None, self.rank)
        mAPt2t = calc_map_k(query_txt, retrieval_txt, self.query_labels,
                            self.retrieval_labels, None, self.rank)
    
        if self.max_mapi2t < mAPi2t:
            self.best_epoch_i = epoch
            self.save_mat(query_img, query_txt, retrieval_img,
                          retrieval_txt, mode_name="i2t")
        self.max_mapi2t = max(self.max_mapi2t, mAPi2t)
        if self.max_mapt2i < mAPt2i:
            self.best_epoch_t = epoch
            self.save_mat(query_img, query_txt, retrieval_img,
                          retrieval_txt, mode_name="t2i")
        self.max_mapt2i = max(self.max_mapt2i, mAPt2i)
        self.logger.info(f">>>>>> [{epoch}/{self.args.epochs}], MAP(i->t): {mAPi2t}, MAP(t->i): {mAPt2i}, MAP(t->t): {mAPt2t}, MAP(i->i): {mAPi2i}, \
                    MAX MAP(i->t): {self.max_mapi2t}, MAX MAP(t->i): {self.max_mapt2i}, query_encoder_time: {q_encoder_time}, retrieval_encoder_time: {r_encoder_time}")

    def save_mat(self, query_img, query_txt, retrieval_img, retrieval_txt, mode_name="i2t"):

        save_dir = os.path.join(self.args.save_dir, "PR_cruve")
        os.makedirs(save_dir, exist_ok=True)

        query_img = query_img.cpu().detach().numpy()
        query_txt = query_txt.cpu().detach().numpy()
        retrieval_img = retrieval_img.cpu().detach().numpy()
        retrieval_txt = retrieval_txt.cpu().detach().numpy()
        query_labels = self.query_labels.numpy()
        retrieval_labels = self.retrieval_labels.numpy()

        result_dict = {
            'q_img': query_img,
            'q_txt': query_txt,
            'r_img': retrieval_img,
            'r_txt': retrieval_txt,
            'q_l': query_labels,
            'r_l': retrieval_labels
        }
        scio.savemat(os.path.join(save_dir, str(self.args.output_dim) +
                     "-ours-" + self.args.dataset + "-" + mode_name + ".mat"), result_dict)
        self.logger.info(f">>>>>> save best {mode_name} data!")

    