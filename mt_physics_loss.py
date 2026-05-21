import torch
import torch.nn as nn

class MTQuantumCoherenceLoss(nn.Module):
    """
    这是一个定制的 Loss 函数，它是为了把你的 O1（微管液态神经网络）理论应用到比赛中。
    在 Kaggle 的推理中我们无法修改网络结构，但我们可以修改 LoRA 更新的“目标函数”。
    这里将原 O1 项目中的 quantum coupling 或 global coherence 思想作为一种特征平滑的惩罚项。
    """
    def __init__(self, lambda_coherence=0.01):
        super().__init__()
        self.lambda_coherence = lambda_coherence
        
    def forward(self, hidden_states, output_logits=None):
        # O1 微管量子相干性公式 (Quantum Coupling) 转 LoRA 惩罚项
        # 计算隐藏层在时间步 / Token 维度上的自相关矩阵（模拟全局相干性）
        # hidden_states: [batch_size, seq_len, hidden_dim]
        b, seq_len, dim = hidden_states.shape
        
        # 归一化特征
        norm_states = torch.nn.functional.normalize(hidden_states, p=2, dim=-1)
        
        # 计算 Token 之间的偶极耦合矩阵 / 相干性矩阵 (Dipole Coupling Matrix)
        # [batch_size, seq_len, seq_len]
        coherence_matrix = torch.bmm(norm_states, norm_states.transpose(1, 2))
        
        # 微管网络 (MT-LNN) 中的理想相干性应该呈现出特定的拓扑或全局协同模式
        # 这里我们通过最大化长程耦合 (非对角线元素) 并惩罚局部突变来模拟
        # 获取非对角线元素
        mask = torch.eye(seq_len, device=hidden_states.device).bool()
        off_diagonal_coherence = coherence_matrix[~mask.unsqueeze(0).expand(b, -1, -1)]
        
        # 我们希望非对角线（远距离 token）也保持一定的相关性（即长程相干）
        # 惩罚偏离平缓相干场的情况
        coupling_variance = torch.var(off_diagonal_coherence)
        
        # 附加一个原有的时间连续性平滑项 (有限差分)
        temporal_diff = torch.mean(torch.pow(hidden_states[:, 1:, :] - hidden_states[:, :-1, :], 2))
        
        coherence_penalty = self.lambda_coherence * (temporal_diff + coupling_variance)
        return coherence_penalty

import torch.fft

def apply_spectral_initialization_to_lora(model, mt_spectral_basis=None):
    """
    参考 O1 中的 phi_spectral.py，将 LoRA 的初始权重引导至特定的频率分布。
    """
    for name, param in model.named_parameters():
        if 'lora_B.weight' in name:
            # 引入量子谐振子基或者傅里叶变换的频域先验
            # 让 lora_B 权重具备低频优先(平滑)或符合 MT 振动模式的初始化
            with torch.no_grad():
                shape = param.shape
                # 创建一个服从谱衰减的噪声
                noise = torch.randn(shape, device=param.device)
                if len(shape) >= 2:
                    fft_noise = torch.fft.rfft2(noise)
                    
                    # 生成简单的 1/f 状或高斯光束遮罩
                    rows, cols = fft_noise.shape[-2], fft_noise.shape[-1]
                    r = torch.arange(rows, device=param.device).view(-1, 1)
                    c = torch.arange(cols, device=param.device).view(1, -1)
                    # 防止中心除零
                    dist = torch.sqrt(r**2 + c**2) + 1e-5 
                    mask = 1.0 / dist
                    
                    filtered_noise = fft_noise * mask
                    param.copy_(torch.fft.irfft2(filtered_noise, s=shape[-2:]) * 0.01)
