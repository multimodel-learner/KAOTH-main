import scipy.io as scio
import numpy as np
import torch
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt

# ------------------------- 核心指标计算函数 -------------------------
def calc_hammingDist(B1, B2):
    q = B2.shape[1]
    if len(B1.shape) < 2:
        B1 = B1.unsqueeze(0)
    distH = 0.5 * (q - B1.mm(B2.transpose(0, 1)))
    return distH

def calc_map_k_matrix(qB, rB, query_L, retrieval_L, k=None, rank=0):
    num_query = query_L.shape[0]
    map = 0.0
    k = k if k else retrieval_L.shape[0]
    
    # 矩阵化计算加速
    sim_matrix = (query_L @ retrieval_L.T) > 0
    hamm_matrix = calc_hammingDist(qB, rB)
    
    for i in range(num_query):
        gnd = sim_matrix[i].float()
        tsum = torch.sum(gnd)
        if tsum == 0:
            continue
            
        _, ind = torch.sort(hamm_matrix[i])
        gnd_sorted = gnd[ind]
        
        relevant = torch.nonzero(gnd_sorted)[:k]
        if relevant.size(0) == 0:
            continue
            
        pos = torch.arange(1, relevant.size(0)+1, dtype=torch.float)
        rank_pos = relevant.squeeze().float() + 1.0
        map += torch.mean(pos / rank_pos)
        
    return map / num_query

def calc_map_k(qB, rB, query_L, retrieval_L, k=None, rank=0):
    num_query = query_L.shape[0]
    map = 0.0
    k = k if k else retrieval_L.shape[0]
    
    for i in tqdm(range(num_query), desc="Calculating mAP"):
        gnd = (query_L[i].unsqueeze(0) @ retrieval_L.T > 0).squeeze().float()
        hamm = calc_hammingDist(qB[i], rB).squeeze()
        
        _, ind = torch.sort(hamm)
        gnd = gnd[ind]
        
        relevant = torch.nonzero(gnd)[:k]
        if relevant.size(0) == 0:
            continue
            
        pos = torch.arange(1, relevant.size(0)+1, dtype=torch.float)
        rank_pos = relevant.squeeze().float() + 1.0
        map += torch.mean(pos / rank_pos)
        
    return map / num_query

def calc_precisions_topn_matrix(qB, rB, query_L, retrieval_L, topn_list=[100, 500, 1000]):
    # 矩阵化快速计算
    sim_matrix = (query_L @ retrieval_L.T) > 0
    hamm_matrix = calc_hammingDist(qB, rB)
    
    prec_dict = {n: 0.0 for n in topn_list}
    for n in topn_list:
        _, topn_indices = torch.topk(hamm_matrix, k=n, dim=1, largest=False)
        hits = torch.gather(sim_matrix.float(), 1, topn_indices).sum(dim=1)
        prec_dict[n] = (hits / n).mean().item()
        
    return prec_dict

def calc_precisions_topn(qB, rB, query_L, retrieval_L, num_retrieval=1000):
    num_query = query_L.shape[0]
    precision = 0.0
    
    for i in tqdm(range(num_query), desc="Calculating Precision@k"):
        gnd = (query_L[i].unsqueeze(0) @ retrieval_L.T > 0).squeeze().float()
        hamm = calc_hammingDist(qB[i], rB).squeeze()
        
        _, ind = torch.sort(hamm)
        gnd = gnd[ind][:num_retrieval]
        
        precision += gnd.sum() / num_retrieval
        
    return precision / num_query

def calc_precisions_hash(qB, rB, query_L, retrieval_L, max_radius=32):
    # 汉明距离分布统计
    hamm_matrix = calc_hammingDist(qB, rB).byte()
    sim_matrix = (query_L @ retrieval_L.T > 0).float()
    
    precision_curve = []
    recall_curve = []
    total_relevant = sim_matrix.sum().item()
    
    for radius in range(0, max_radius+1):
        mask = hamm_matrix <= radius
        retrieved = mask.sum().float()
        relevant_retrieved = (sim_matrix * mask).sum().float()
        
        precision = relevant_retrieved / retrieved if retrieved > 0 else 0
        recall = relevant_retrieved / total_relevant if total_relevant > 0 else 0
        
        precision_curve.append(precision.item())
        recall_curve.append(recall.item())
        
    return precision_curve, recall_curve

def calc_ndcg(queries, retrievals, query_L, retrieval_L, top_k=1000):
    # 基于余弦相似度的NDCG
    similarity = torch.mm(queries, retrievals.T)
    ndcg_scores = []
    
    for i in tqdm(range(queries.shape[0]), desc="Calculating NDCG"):
        sim = similarity[i]
        rel = (query_L[i].unsqueeze(0) @ retrieval_L.T).squeeze().float()
        
        # 获取top-k结果
        _, top_indices = torch.topk(sim, k=top_k)
        rel_sorted = rel[top_indices]
        
        # 计算DCG
        dcg = torch.sum(rel_sorted / torch.log2(torch.arange(2, top_k+2, dtype=torch.float)))
        
        # 计算IDCG
        ideal_rel = torch.sort(rel, descending=True)[0][:top_k]
        idcg = torch.sum(ideal_rel / torch.log2(torch.arange(2, top_k+2, dtype=torch.float)))
        
        ndcg_scores.append((dcg / idcg).item() if idcg > 0 else 0)
        
    return torch.tensor(ndcg_scores).mean().item()

def calc_precisions_hamming_radius(qB, rB, query_L, retrieval_L, radius_list=[2,4,8,16]):
    hamm_matrix = calc_hammingDist(qB, rB)
    sim_matrix = (query_L @ retrieval_L.T > 0).float()
    
    prec_dict = {}
    for radius in radius_list:
        mask = hamm_matrix <= radius
        valid_queries = mask.any(dim=1)
        
        if valid_queries.sum() == 0:
            prec_dict[radius] = 0.0
            continue
            
        correct = (sim_matrix * mask.float()).sum(dim=1)[valid_queries]
        total = mask.sum(dim=1)[valid_queries].float()
        prec_dict[radius] = (correct / total).mean().item()
        
    return prec_dict

# ------------------------- 主程序 -------------------------
def load_data(mat_path):
    """加载.mat文件并转换为PyTorch Tensor"""
    data = scio.loadmat(mat_path)
    return {
        'q_img': torch.from_numpy(data['q_img']),
        'q_txt': torch.from_numpy(data['q_txt']),
        'r_img': torch.from_numpy(data['r_img']),
        'r_txt': torch.from_numpy(data['r_txt']),
        'q_l': torch.from_numpy(data['q_l']),
        'r_l': torch.from_numpy(data['r_l'])
    }

def calculate_all_metrics(data, top_k=1000, hamming_radius=[2,4,8]):
    """计算所有指标"""
    metrics = {}
    
    # 定义模态组合
    modalities = {
        'i2t': (data['q_img'], data['r_txt']),
        't2i': (data['q_txt'], data['r_img']),
        'i2i': (data['q_img'], data['r_img']),
        't2t': (data['q_txt'], data['r_txt'])
    }
    
    for mode, (q, r) in modalities.items():
        print(f"\n========== 计算 {mode.upper()} 模式 ==========")
        
        # 二值化处理
        q_binary = torch.sign(q)
        r_binary = torch.sign(r)
        
        metrics[f'mAP_{mode}'] = calc_map_k_matrix(q_binary, r_binary, data['q_l'], data['r_l'])
        metrics[f'Precision@{top_k}_{mode}'] = calc_precisions_topn_matrix(q_binary, r_binary, data['q_l'], data['r_l'], [top_k])[top_k]
        metrics[f'NDCG@{top_k}_{mode}'] = calc_ndcg(q, r, data['q_l'], data['r_l'], top_k)
        metrics[f'Hamming_Precision_{mode}'], metrics[f'Hamming_Recall_{mode}'] = calc_precisions_hash(q_binary, r_binary, data['q_l'], data['r_l'])
        metrics[f'Radius_Precision_{mode}'] = calc_precisions_hamming_radius(q_binary, r_binary, data['q_l'], data['r_l'], hamming_radius)
        
    return metrics

def plot_curves(metrics):
    """可视化曲线"""
    # 汉明距离精度召回曲线
    plt.figure(figsize=(10, 6))
    for mode in ['i2t', 't2i']:
        plt.plot(metrics[f'Hamming_Recall_{mode}'], metrics[f'Hamming_Precision_{mode}'], label=mode.upper())
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.savefig('precision_recall_curve.png')
    
    # 半径精度柱状图
    plt.figure(figsize=(10, 6))
    modes = ['i2t', 't2i']
    radius_list = [2,4,8]
    width = 0.3
    for i, mode in enumerate(modes):
        values = [metrics[f'Radius_Precision_{mode}'][r] for r in radius_list]
        plt.bar(np.arange(len(radius_list)) + i*width, values, width, label=mode.upper())
    plt.xticks(np.arange(len(radius_list)) + width/2, radius_list)
    plt.xlabel('Hamming Radius')
    plt.ylabel('Precision')
    plt.title('Precision at Different Hamming Radii')
    plt.legend()
    plt.savefig('radius_precision.png')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='跨模态检索指标计算脚本')
    parser.add_argument('--mat-path', type=str, required=True, help='.mat文件路径')
    parser.add_argument('--top-k', type=int, default=1000, help='Top K值，默认1000')
    parser.add_argument('--hamming-radius', nargs='+', type=int, default=[2,4,8], help='汉明半径列表')
    args = parser.parse_args()
    
    # 1. 加载数据
    data = load_data(args.mat_path)
    
    # 2. 计算指标
    metrics = calculate_all_metrics(data, args.top_k, args.hamming_radius)
    
    # 3. 打印结果
    print("\n========== 最终指标 ==========")
    for k, v in metrics.items():
        if isinstance(v, dict):
            print(f"{k}:")
            for sub_k, sub_v in v.items():
                print(f"  {sub_k}: {sub_v:.4f}")
        elif isinstance(v, list):
            print(f"{k}: {np.round(v, 4)}")
        else:
            print(f"{k}: {v:.4f}")
    
    # 4. 可视化
    plot_curves(metrics)



# python retrieval_metrics.py \
#     --mat-path /path/to/your_data.mat \
#     --top-k 1000 \
#     --hamming-radius 2 4 8