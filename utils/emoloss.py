import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# import ot
from scipy.linalg import block_diag

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")            
temperature=0.2
def get_negative_mask(batch_size):
    negative_mask = torch.ones((batch_size, 2 * batch_size), dtype=bool)
    for i in range(batch_size):
        negative_mask[i, i] = 0
        negative_mask[i, i + batch_size] = 0

    negative_mask = torch.cat((negative_mask, negative_mask), 0)
    return negative_mask

def get_neg_mask(batch_size):
    postive_mask = torch.zeros((2*batch_size, 2*batch_size))
    for i in range(batch_size):
        postive_mask[i, i] = 1
        postive_mask[i, i + batch_size] = 1
        postive_mask[i + batch_size, i] = 1
        postive_mask[i + batch_size, i + batch_size] = 1
    negative_mask = 1-postive_mask
    return negative_mask.to(device)


def cost_fun(out_1, out_2):
    x = out_1[0].unsqueeze(0)
    y = out_2[0].unsqueeze(1)
    cost = torch.sum(torch.abs(x-y)**2,2)
    batch_size = out_1[0].shape[0]
    postive_mask = torch.zeros((batch_size, batch_size)).to(device)
    half_batch_size = int(batch_size/2)
    for i in range(half_batch_size):
        postive_mask[i, i] = float("Inf")
        postive_mask[i, i + half_batch_size] = float("Inf")
        postive_mask[i + half_batch_size, i] = float("Inf")
        postive_mask[i + half_batch_size, i + half_batch_size] = float("Inf")
    cost = cost + postive_mask
    return cost.reshape((1, cost.shape[0], cost.shape[1]))
    
def new_cost_fun(out_1, out_2, kappa):
    x = out_1[0].unsqueeze(0)
    y = out_2[0].unsqueeze(1)
    
    # 增加数值稳定处理
    squared_dist = torch.sum(torch.abs(x-y)**2, 2)
    max_val = torch.max(squared_dist).detach()
    clamped_dist = torch.clamp(squared_dist - kappa, min=-50, max=50)  # 防止指数爆炸
    
    # 使用更稳定的指数计算
    cost = torch.exp(clamped_dist - max_val) * torch.exp(max_val)
    
    batch_size = out_1[0].shape[0]
    half_batch_size = batch_size // 2
    
    # 使用大数值替代Inf
    LARGE_VALUE = 1e10
    postive_mask = torch.zeros_like(cost)
    for i in range(half_batch_size):
        postive_mask[i, i] = LARGE_VALUE
        postive_mask[i, i + half_batch_size] = LARGE_VALUE
        postive_mask[i + half_batch_size, i] = LARGE_VALUE
        postive_mask[i + half_batch_size, i + half_batch_size] = LARGE_VALUE
    
    return (cost + postive_mask).reshape((1, *cost.shape))
   
def pad_to_match(tensor, target_shape):
    """
    将 tensor 填充到 target_shape 的形状
    """
    while tensor.shape != target_shape:
        # 在最后一个维度上填充 0
        tensor = torch.cat([tensor, torch.zeros_like(tensor[..., :1])], dim=-1)
    return tensor



def sinkhorn_stabilized(a, b, M, reg, numItermax=200, stopThr=1e-6, warn=True):
    """数值稳定的Sinkhorn实现"""
    u = np.ones_like(a)
    v = np.ones_like(b)
    K = np.exp(-M / reg)
    
    for i in range(numItermax):
        u_prev = u.copy()
        v = b / (K.T @ u + 1e-16)
        u = a / (K @ v + 1e-16)
        
        if np.linalg.norm(u - u_prev) < stopThr:
            break
            
    return u.reshape(-1, 1) * K * v.reshape(1, -1)



def supervised_OT_hard1(out_1, out_2, count,  reg, labels,   tau_plus=0.1, kappa=1.0, new_cost=True):
    temperature = 0.2
    current_mask = out_1.size(0)  # 动态获取batch_size
    
    # ==== 与OT_hard完全相同的预处理 ====
    out_1 = F.normalize(out_1, p=2, dim=1)
    out_2 = F.normalize(out_2, p=2, dim=1)
    out = torch.cat([out_1, out_2], dim=0)
    
    # ==== 原始相似度计算 ====
    sim_matrix = torch.mm(out, out.t().contiguous())
    logits = sim_matrix / temperature
    log_max = torch.max(logits, dim=1, keepdim=True)[0]
    stable_logits = logits - log_max.detach()
    neg = torch.exp(stable_logits)
    neg_mask = get_neg_mask(current_mask)
    neg_masked = neg * neg_mask

    if reg > 0:
        # ==== 监督增强的成本矩阵 ====
        with torch.no_grad():
            if new_cost:
                M = new_cost_fun([out], [out], kappa)
            else:
                M = cost_fun([out], [out])
            
            # 标签注入 (核心修改点)
            label_sim = torch.mm(labels.float(), labels.float().T) > 0
            extended_mask = torch.block_diag(label_sim, label_sim).float()
            M[0] -= count * extended_mask  # 同类样本成本降低

            # ==== 保持原始传输矩阵计算 ====
            a = torch.ones(2*current_mask, device=out.device) / (2*current_mask)
            
            Trans_global = sinkhorn_stabilized(
                a.cpu().numpy(),
                a.cpu().numpy(),
                M[0].cpu().numpy(),
                reg=reg,
                numItermax=200,
                stopThr=1e-6,
                warn=False
            )
            
            # ==== 局部OT计算 ====
            part1 = out[:current_mask]
            part2 = out[current_mask:]
            M1 = cost_fun([part1], [part1])[0].cpu().numpy()
            M2 = cost_fun([part2], [part2])[0].cpu().numpy()
            
            Trans1 = sinkhorn_stabilized(
                np.ones(current_mask)/current_mask,
                np.ones(current_mask)/current_mask,
                M1, reg=reg, numItermax=200
            )
            Trans2 = sinkhorn_stabilized(
                np.ones(current_mask)/current_mask,
                np.ones(current_mask)/current_mask,
                M2, reg=reg, numItermax=200
            )

            # ==== 混合传输矩阵 ====
            Trans_combined = 0.5 * Trans_global + 0.5 * block_diag(Trans1, Trans2)
            # Trans_combined = Trans_global
            Trans_combined = torch.tensor(Trans_combined, device=out.device)

        # ==== 保持原始损失计算流程 ====
        N = current_mask * 2 - 2
        pos_sim = torch.sum(out_1 * out_2, dim=-1)
        log_pos = pos_sim / temperature - log_max[:current_mask].squeeze()
        pos = torch.exp(log_pos)
        pos = torch.cat([pos, pos], dim=0)
        
        neg = neg_masked * Trans_combined
        neg_2 = torch.sum(neg, 1) * current_mask * 2 * N
        Ng = (-tau_plus * N * pos + neg_2) / (1 - tau_plus)
        Ng = torch.clamp(Ng, min=N * torch.exp(torch.tensor(-1/temperature, device=out.device)))
        
        # ==== 最终损失计算 ====
        with torch.cuda.amp.autocast(enabled=False):
            pos = pos.float()
            Ng = Ng.float()
            loss = (- torch.log(pos / (pos + Ng + 1e-12))).mean()
    else:
        Ng = neg.sum(dim=-1)
        with torch.cuda.amp.autocast(enabled=False):
            loss = (- torch.log(pos / (pos + Ng + 1e-12))).mean()

    return loss



