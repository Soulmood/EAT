import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import DropPath, to_2tuple

"""
输入 (B, 1, F, T)
        ↓
TFDDCModule
    ├─ stem (Conv)
    ├─ TFDDC × N
    ├─ patch_proj (Patchify)
    ├─ TFSA (时频注意力)
        ↓
输出 (B, N_patches, C)
        ↓
Transformer (EAT_pretraining.py)
        ↓
Decoder / Loss（含 PGEnergy）
"""

"""
结构依赖关系
CDilated
   ↑
TFDDC
   ↑
TFDDCModule
   ↓
TFSA
   ↓
Transformer (EAT)
   ↓
Decoder
   ↓
PGEnergyModule（loss）

数据流
输入频谱 (B,1,F,T)
        ↓
TFDDCModule
    ↓
(B, N, C)
        ↓
Transformer Encoder（EAT）
        ↓
Decoder
        ↓
recon (B,N,D)
        ↓
PGEnergyModule
        ↓
loss
"""
class CDilated(nn.Module):# 膨胀卷积封装
    def __init__(self, n_in, n_out, k_size, stride=1, dilation=1, groups=1):# 构造函数
        super().__init__()
        k_h, k_w = to_2tuple(k_size)
        padding_h = ((k_h - 1) // 2) * dilation
        padding_w = ((k_w - 1) // 2) * dilation
        self.padding = nn.ConstantPad2d((padding_w, padding_w, padding_h, padding_h), 0.0)# 参数顺序(left, right, top, bottom)
        self.conv = nn.Conv2d(
            n_in,
            n_out,
            (k_h, k_w),
            stride=stride,
            bias=False,
            dilation=dilation,
            groups=groups,
        )

    def forward(self, x):
        return self.conv(self.padding(x))


class TFSA(nn.Module): #Time-Frequency Self-Attention
    """
    全局时频 attention（类似 Transformer）

    时间因果 attention（类似序列模型）  
    """
    def __init__(self, c=768, causal=True):
        super().__init__()
        d_c = max(1, c // 4)
        self.d_c = d_c
        self.f_qkv = nn.Sequential(
            nn.Conv2d(c, d_c * 3, kernel_size=1, bias=False),
            nn.BatchNorm2d(d_c * 3),
            nn.PReLU(d_c * 3),
        )
        self.t_qk = nn.Sequential(
            nn.Conv2d(c, d_c * 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(d_c * 2),
            nn.PReLU(d_c * 2),
        )
        self.proj = nn.Sequential(
            nn.Conv2d(d_c, c, kernel_size=1, bias=False),
            nn.BatchNorm2d(c),
            nn.PReLU(c),
        )
        self.causal = causal

    def forward(self, inp):
        b, _, f, t = inp.shape
        f_qkv = self.f_qkv(inp)
        qf, kf, v = torch.chunk(f_qkv, 3, dim=1)
        qf = qf.reshape(b, self.d_c, f * t).transpose(1, 2)
        kf = kf.reshape(b, self.d_c, f * t)
        score = torch.bmm(qf, kf) / (self.d_c**0.5)
        attn = score.softmax(dim=-1)
        v_ = v.reshape(b, self.d_c, f * t).transpose(1, 2)
        out = torch.bmm(attn, v_).transpose(1, 2).reshape(b, self.d_c, f, t)
        out = self.proj(out)

        t_qk = self.t_qk(inp)
        qt, kt = torch.chunk(t_qk, 2, dim=1)
        qt = qt.mean(dim=2)
        kt = kt.mean(dim=2)
        t_score = torch.bmm(qt.transpose(1, 2), kt) / (self.d_c**0.5)
        if self.causal:
            mask = torch.ones(t, t, device=t_score.device, dtype=torch.bool).triu_(1)# 构造上三角mask
            # `-1e9` overflows in fp16; use dtype-safe minimum instead.
            t_score = t_score.masked_fill(mask, torch.finfo(t_score.dtype).min)
        t_attn = t_score.softmax(dim=-1)
        out_t = torch.bmm(t_attn, out.mean(dim=2).transpose(1, 2)).transpose(1, 2)
        out = out + out_t.unsqueeze(2)
        return out + inp


class TFDDC(nn.Module):#时频解耦 + 膨胀卷积 + 深度可分离卷积 + 可学习融合权重
    """
    这一模块的本质总结（模型设计角度）
    这是一个：
    “CNN版 Transformer block 替代结构”
    它替代了什么？
    Transformer组件	TFDDC 对应
    Self-Attention	Dilated Conv
    FFN	1x1 Conv expand
    Residual	skip_path
    Dropout	DropPath
    内部结构
    x
    ↓
    1×1 Conv（扩展通道）
    ↓
    分支1：时间卷积 (3×5, dilation)
    分支2：频率卷积 (5×3, dilation)
    ↓
    加权融合（可学习 α）
    ↓
    1×1 Conv（压缩）
    ↓
    Residual + DropPath
    """
    def __init__(self, in_chs, expan_ratio=4, stride=1, dilation=1, drop_path=0.0):
        super().__init__()
        self.skip_path = (
            nn.Identity()
            if stride == 1
            else nn.Sequential(
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                nn.Conv2d(in_chs, in_chs, kernel_size=1),
            )
        )
        expan_chs = in_chs * expan_ratio

        self.pconv1 = nn.Conv2d(in_chs, expan_chs, kernel_size=1)
        self.norm_act1 = nn.Sequential(nn.BatchNorm2d(expan_chs), nn.GELU())

        self.conv2_t = CDilated(
            expan_chs,
            expan_chs,
            k_size=(3, 5),
            stride=stride,
            dilation=dilation,
            groups=expan_chs,
        )
        self.conv2_f = CDilated(
            expan_chs,
            expan_chs,
            k_size=(5, 3),
            stride=stride,
            dilation=dilation,
            groups=expan_chs,
        )
        self.t_f_weight = nn.Parameter(torch.tensor(0.5))

        self.pconv2 = nn.Conv2d(expan_chs, in_chs, kernel_size=1)
        self.norm_act2 = nn.Sequential(nn.BatchNorm2d(in_chs), nn.GELU())
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x):
        skip = self.skip_path(x)
        x = self.pconv1(x)
        x = self.norm_act1(x)
        x_t = self.conv2_t(x)
        x_f = self.conv2_f(x)
        x = self.t_f_weight * x_t + (1 - self.t_f_weight) * x_f
        x = self.pconv2(x)
        x = self.drop_path(x) + skip
        x = self.norm_act2(x)
        return x


class TFDDCModule(nn.Module):#桥接 CNN → Transformer
    """
    这个模块决定：
    CNN 输出如何变成 token
    为什么可以接 Transformer（EAT 主体）
    同时它里面包含：
    👉 TFSA 的调用（attention + CNN 融合）
    """
    def __init__(self, img_size, patch_size=16, in_chans=1, embed_dim=768, num_tfddc=2):
        super().__init__()
        self.img_size = to_2tuple(img_size)
        self.patch_size = to_2tuple(patch_size)
        self.embed_dim = embed_dim
        self.stem = nn.Conv2d(in_chans, embed_dim, kernel_size=3, stride=1, padding=1)
        self.tfddc_layers = nn.ModuleList(
            [TFDDC(embed_dim, stride=1, dilation=1 + i) for i in range(num_tfddc)]
        )
        self.patch_proj = nn.Conv2d(
            embed_dim,
            embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )
        self.tffi = TFSA(embed_dim, causal=True)

        w = self.patch_proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

    def forward(self, x):
        x = self.stem(x)
        for blk in self.tfddc_layers:
            x = blk(x)
        x = self.patch_proj(x)
        x = self.tffi(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class PGEnergyModule(nn.Module):#物理约束 loss
    def forward(self, recon_patches, target_patches):
        recon_energy = (recon_patches.float() ** 2).mean(dim=-1)
        target_energy = (target_patches.float() ** 2).mean(dim=-1)
        return F.mse_loss(recon_energy, target_energy)

