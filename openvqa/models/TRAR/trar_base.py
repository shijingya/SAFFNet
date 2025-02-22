from openvqa.ops.fc import FC, MLP
from openvqa.ops.layer_norm import LayerNorm

import torch.nn as nn
import torch.nn.functional as F
import torch
import math
import numpy as np


from torch.nn.parameter import Parameter
from torch.nn.init import xavier_uniform_

# ---------------------------
# ---- Attention Pooling ----
# ---------------------------
class AttFlat(nn.Module):
    def __init__(self, in_channel, glimpses=1, dropout_r=0.1):
        super(AttFlat, self).__init__()
        self.glimpses = glimpses

        self.mlp = MLP(
            in_size=in_channel,
            mid_size=in_channel,
            out_size=glimpses,
            dropout_r=dropout_r,
            use_relu=True
        )

        self.linear_merge = nn.Linear(
            in_channel * glimpses,
            in_channel
        )
        self.norm = LayerNorm(in_channel)

    def forward(self, x, x_mask):
        att = self.mlp(x)

        att = att.masked_fill(
            x_mask.squeeze(1).squeeze(1).unsqueeze(2),
            -1e9
        )
        att = F.softmax(att, dim=1)

        att_list = []
        for i in range(self.glimpses):
            att_list.append(
                torch.sum(att[:, :, i: i + 1] * x, dim=1)
            )

        x_atted = torch.cat(att_list, dim=1)
        x_atted = self.linear_merge(x_atted)
        x_atted = self.norm(x_atted)
        return x_atted

# ---------------------------------------
# ---- Multi-Head Attention question ----
# ---------------------------------------
class MHAtt(nn.Module):
    def __init__(self, __C):
        super(MHAtt, self).__init__()
        self.__C = __C

        self.linear_v = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.linear_k = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.linear_q = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.linear_merge = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)  # 4,14,512

        self.dropout = nn.Dropout(__C.DROPOUT_R)

    def forward(self, v, k, q, mask):
        n_batches = q.size(0)

        v = self.linear_v(v).view(
            n_batches,   # 64
            -1,
            self.__C.MULTI_HEAD,  # 8
            #int(self.__C.HIDDEN_SIZE / self.__C.MULTI_HEAD)
            self.__C.HIDDEN_SIZE_HEAD  # 64
        ).transpose(1, 2)  # 64,8,14,64
        
        k = self.linear_k(k).view(
            n_batches,
            -1,
            self.__C.MULTI_HEAD,
            #int(self.__C.HIDDEN_SIZE / self.__C.MULTI_HEAD)
            self.__C.HIDDEN_SIZE_HEAD
        ).transpose(1, 2)
        
        #k=v

        q = self.linear_q(q).view(
            n_batches,
            -1,
            self.__C.MULTI_HEAD,
            #int(self.__C.HIDDEN_SIZE / self.__C.MULTI_HEAD)
            self.__C.HIDDEN_SIZE_HEAD
        ).transpose(1, 2)                              # 64,8,14,64

        atted = self.att(v, k, q, mask)   # 64,8,14,64
        atted = atted.transpose(1, 2).contiguous().view(
            n_batches,
            -1,
            self.__C.HIDDEN_SIZE   # 64,14,512
        )

        atted = self.linear_merge(atted)

        return atted

    def att(self, value, key, query, mask):
        d_k = query.size(-1)  # 64

        scores = torch.matmul(
            query, key.transpose(-2, -1)
        ) / math.sqrt(d_k)                        # 64,8,14,14

        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)

        att_map = F.softmax(scores, dim=-1)
        att_map = self.dropout(att_map)

        return torch.matmul(att_map, value)   # 64,8,14,64


# ---------------------------------------
# ---- Fourier-Multi-Head Attention question ----
# ---------------------------------------
class Fourier_MHAtt(nn.Module):
    def __init__(self, __C):
        super(Fourier_MHAtt, self).__init__()
        self.__C = __C

        self.linear_v = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.linear_k = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.linear_q = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.linear_merge = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)  # 64,64,512
        self.linear_s = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)

        self.avgpool_s = nn.AdaptiveAvgPool2d((1, None))
        self.linear_ws = nn.Linear(__C.HIDDEN_SIZE_HEAD, __C.HIDDEN_SIZE_HEAD)

        self.dropout = nn.Dropout(__C.DROPOUT_R)

    def forward(self, v, k, q, mask, s):  # s:[64,14,512]
        n_batches = q.size(0)

        s = self.linear_s(s).view(  # 64,8,14,64
            n_batches,  # 64
            -1,
            self.__C.MULTI_HEAD,  # 8
            # int(self.__C.HIDDEN_SIZE / self.__C.MULTI_HEAD)
            self.__C.HIDDEN_SIZE_HEAD  # 64
        ).transpose(1, 2)

        w_s = self.avgpool_s(s)  # 64,8,1,64
        w_s = w_s.expand(n_batches, self.__C.MULTI_HEAD, 64, self.__C.HIDDEN_SIZE_HEAD)  # 64,8,64,64 加在文本上是 14 64
        ww_s = self.linear_ws(w_s)  # 64,8,64,64 #加在文本上是 14 64
        weight = torch.sigmoid(ww_s)

        # complex_ws = we_s.unsqueeze(-1).repeat(1, 1, 1, 1, 2).cuda()  # 64,8,64,64,2
        # weight = torch.view_as_complex(complex_ws).cuda()  # 64,8,64,64

        v = self.linear_v(v).view(
            n_batches,  # 64
            -1,
            self.__C.MULTI_HEAD,  # 8
            # int(self.__C.HIDDEN_SIZE / self.__C.MULTI_HEAD)
            self.__C.HIDDEN_SIZE_HEAD  # 64
        ).transpose(1, 2)  # 64,8,64,64

        k = self.linear_k(k).view(
            n_batches,
            -1,
            self.__C.MULTI_HEAD,
            # int(self.__C.HIDDEN_SIZE / self.__C.MULTI_HEAD)
            self.__C.HIDDEN_SIZE_HEAD
        ).transpose(1, 2)

        # k=v

        q = self.linear_q(q).view(
            n_batches,
            -1,
            self.__C.MULTI_HEAD,
            # int(self.__C.HIDDEN_SIZE / self.__C.MULTI_HEAD)
            self.__C.HIDDEN_SIZE_HEAD
        ).transpose(1, 2)  # 64,8,14,64

        atted = self.att(v, k, q, mask, weight)  # 64,8,14,64
        atted = atted.transpose(1, 2).contiguous().view(
            n_batches,
            -1,
            self.__C.HIDDEN_SIZE  # 64,14,512
        )

        atted = self.linear_merge(atted)

        return atted

# 不用fft。这里使用上下文引导的weight * sa
    def att1(self, value, key, query, mask, weight):
        d_k = query.size(-1)  # 64

        scores = torch.matmul(
            query, key.transpose(-2, -1)
        ) / math.sqrt(d_k)

        scores = scores * weight

        if mask is not None:  # 64,1,1,64
            scores = scores.masked_fill(mask, -1e9)

        att_map = F.softmax(scores, dim=-1)
        att_map = self.dropout(att_map)

        return torch.matmul(att_map, value)  # 64,8,64,64

    def att(self, value, key, query, mask, weight):
        d_k = query.size(-1)  # 64

        q_fft = torch.fft.fft2(query, dim=(-2, -1), norm='ortho')
        key_fft = torch.fft.fft2(key,  dim=(-2, -1), norm='ortho')  # 64,8,64,64
        # w = nn.Parameter(torch.randn(n_batches, 8, 64, 64, dtype=torch.float32) * 0.02).cuda()

        scores = torch.matmul(
            q_fft, key_fft.transpose(-2, -1)  # 在时域内
        ) / math.sqrt(d_k)  # 64,8,64,64
        scores_shift = torch.fft.fftshift(scores, dim=(-2, -1))  # 去中心化

        scores_fft = (scores_shift * weight).cuda()
        scores_fft = torch.fft.ifftshift(scores_fft, dim=(-2, -1))
        scores_fft = torch.fft.ifft2(scores_fft, dim=(-2, -1), norm='ortho')
        scores = torch.real(scores_fft)

        scores_s = torch.matmul(
            query, key.transpose(-2, -1)
        ) / math.sqrt(d_k)
        scores = scores + scores_s * weight

        if mask is not None:  # 64,1,1,64
            # mask_fft = torch.fft.fft(mask, dim=-1)
            # mask_fft = torch.unsqueeze(mask_fft, dim=-2)
            # scores = scores.masked_fill(mask_fft, -1e9)
            scores = scores.masked_fill(mask, -1e9)

        # scores = torch.fft.ifft(scores, dim=-2)
        # scores = torch.real(scores)
        att_map = F.softmax(scores, dim=-1)
        att_map = self.dropout(att_map)
        # att_map = torch.fft.fft(att_map, dim=-2)
        # s = torch.matmul(att_map, v_fft)
        # s = torch.fft.ifft(s, dim=-2)
        # s = torch.real(s)
        # return s
        return torch.matmul(att_map, value)  # 64,8,64,64


# ---------------------------
# ---- Feed Forward Nets ----
# ---------------------------

class FFN(nn.Module):
    def __init__(self, __C):
        super(FFN, self).__init__()

        self.mlp = MLP(
            in_size=__C.HIDDEN_SIZE,
            mid_size=__C.FF_SIZE,
            out_size=__C.HIDDEN_SIZE,
            dropout_r=__C.DROPOUT_R,
            use_relu=True
        )

    def forward(self, x):
        return self.mlp(x)


# -----------------------------
# ---- Transformer Encoder ----
# -----------------------------

class Encoder(nn.Module):
    def __init__(self, __C):
        super(Encoder, self).__init__()

        self.mhattE = MHAtt(__C)
        # self.mhattE = Fourier_MHAtt(__C)
        self.ffn = FFN(__C)

        self.dropout1 = nn.Dropout(__C.DROPOUT_R)
        self.norm1 = LayerNorm(__C.HIDDEN_SIZE)

        self.dropout2 = nn.Dropout(__C.DROPOUT_R)
        self.norm2 = LayerNorm(__C.HIDDEN_SIZE)

    def forward(self, y, y_mask):
        
        y = self.norm1(y + self.dropout1(
            self.mhattE(y, y, y, y_mask)
        ))
       
        y = self.norm2(y + self.dropout2(
            self.ffn(y)
        ))
        return y

# ---------------------------------
# ---- Multimodal TRAR Decoder ----
# ---------------------------------
class TRAR(nn.Module):
    def __init__(self, __C):
        super(TRAR, self).__init__()

        self.mhatt1 = Fourier_MHAtt(__C)
        # self.mhatt1 = MHAtt(__C)
        self.mhatt2 = MHAtt(__C)
        self.ffn = FFN(__C)

        self.dropout1 = nn.Dropout(__C.DROPOUT_R)
        self.norm1 = LayerNorm(__C.HIDDEN_SIZE)

        self.dropout2 = nn.Dropout(__C.DROPOUT_R)
        self.norm2 = LayerNorm(__C.HIDDEN_SIZE)

        self.dropout3 = nn.Dropout(__C.DROPOUT_R)
        self.norm3 = LayerNorm(__C.HIDDEN_SIZE)

    def forward(self, x, y, x_mask, y_mask):

        x = self.norm1(x + self.dropout1(
            self.mhatt1(v=x, k=x, q=x, mask=x_mask, s=y)
            # self.mhatt1(v=x, k=x, q=x, mask=x_mask)
        ))
      
        x = self.norm2(x + self.dropout2(
            self.mhatt2(v=y, k=y, q=x, mask=y_mask)
        ))
        
        x = self.norm3(x + self.dropout3(
            self.ffn(x)
        ))
       
        return x

# ----------------------------------------
# ---- Encoder-Decoder with TRAR Block----
# ----------------------------------------
class TRAR_ED(nn.Module):
    def __init__(self, __C):
        super(TRAR_ED, self).__init__()
        self.__C = __C

        self.enc_list = nn.ModuleList([Encoder(__C) for _ in range(__C.LAYER)])
        self.dec_list = nn.ModuleList([TRAR(__C) for _ in range(__C.LAYER)])

    def forward(self, y, x, y_mask, x_mask):
        # Get encoder last hidden vector
        for enc in self.enc_list:
            y = enc(y, y_mask)

        for dec in self.dec_list:
            x = dec(x, y, x_mask, y_mask)

        return y, x

    def set_tau(self, tau):
        self.tau = tau

    def set_training_status(self, training):
        self.training = training

