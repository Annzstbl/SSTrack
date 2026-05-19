import math
import os
from typing import List, NamedTuple

import torch
from torch import nn
from torch.nn.modules.transformer import _get_clones

from lib.models.layers.head import build_box_head
from lib.models.sstrack.vit import vit_base_patch16_224
from lib.models.sstrack.vit_ce import VisionTransformerCE, vit_base_patch16_224_ce
from lib.utils.box_ops import box_xyxy_to_cxcywh

from timm.models.layers import Mlp


class _CVTPRuntime(NamedTuple):
    """一次 forward 内 CVTP（模板 token drop）的配置快照。"""
    enabled: bool  # cfg 打开且 self.training
    drop_rate: float  # 传入的丢弃率；未启用时为 0
    strategy: str
    hard_ratio: float


class ATTFu(nn.Module):
    def __init__(self, channels, ratio=0.25):
        super(ATTFu, self).__init__()
        self.channels = channels
        self.fc = nn.Sequential(
            nn.Linear(2*channels, int(ratio*channels), bias=False),
            nn.ReLU(),
            nn.Linear(int(ratio*channels), 2*channels, bias=False),
            nn.Sigmoid()
        )
        self.mlp = Mlp(in_features=2*channels, hidden_features=4*2*channels)
        
    def forward(self, l_pro, l_tem):
        l_tem = torch.mean(l_tem, dim=1, keepdim=True)      # l_tem是模板，先做全局池化
        l_fu = torch.cat([l_pro, l_tem], dim=2)             # 模板和提示cat在一起
        att = self.fc(l_fu)                                 # 利用fc层实现自注意力
        out = self.mlp(att) + att

        return out[:,:,:self.channels]                      # 只保留提示部分

class ATTFu_CrossAttention(nn.Module):
    """
    方案一：使用交叉注意力机制
    特点：l_pro 作为 Query，l_tem 作为 Key 和 Value，实现自适应特征融合
    """
    def __init__(self, channels, ratio=0.25, num_heads=8):
        super(ATTFu_CrossAttention, self).__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        
        assert channels % num_heads == 0, "channels must be divisible by num_heads"
        
        # 交叉注意力：Query 来自 l_pro，Key 和 Value 来自 l_tem
        self.q_proj = nn.Linear(channels, channels, bias=False)
        self.k_proj = nn.Linear(channels, channels, bias=False)
        self.v_proj = nn.Linear(channels, channels, bias=False)
        self.out_proj = nn.Linear(channels, channels, bias=False)
        
        # 层归一化
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)
        
        # FFN（前馈网络）
        self.ffn = nn.Sequential(
            nn.Linear(channels, int(channels * 4)),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(int(channels * 4), channels),
            nn.Dropout(0.1)
        )
        
        self.scale = (self.head_dim) ** -0.5
        
    def forward(self, l_pro, l_tem):
        """
        args:
            l_pro: (B, 1, 768) - 光谱提示特征
            l_tem: (B, 432, 768) - 模板特征
        returns:
            out: (B, 1, 768) - 更新后的光谱提示特征
        """
        B, N_pro, C = l_pro.shape
        _, N_tem, _ = l_tem.shape
        
        # 1. 交叉注意力：l_pro 查询 l_tem 的信息
        q = self.q_proj(l_pro).reshape(B, N_pro, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # (B, H, 1, D)
        k = self.k_proj(l_tem).reshape(B, N_tem, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # (B, H, 432, D)
        v = self.v_proj(l_tem).reshape(B, N_tem, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # (B, H, 432, D)
        
        # 2. 计算注意力分数
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, H, 1, 432)
        attn = attn.softmax(dim=-1)
        
        # 3. 加权聚合模板特征
        out = (attn @ v).permute(0, 2, 1, 3).reshape(B, N_pro, C)  # (B, 1, 768)
        out = self.out_proj(out)
        
        # 4. 残差连接 + 层归一化
        out = self.norm1(l_pro + out)
        
        # 5. FFN 增强
        ffn_out = self.ffn(out)
        out = self.norm2(out + ffn_out)
        
        return out

# 特点和优势：
# ✅ 优势：
#   1. 自适应权重：通过注意力机制自动学习哪些模板特征对当前跟踪最重要
#   2. 保留空间信息：不需要全局池化，可以保留模板的空间结构信息
#   3. 可解释性强：注意力权重可以可视化，了解模型关注哪些区域
#   4. 多尺度融合：可以同时关注多个模板区域
# 
# ⚠️ 特点：
#   1. 计算复杂度：O(N_tem)，比原始方案稍高
#   2. 参数量：相对较多（约 3*768*768 + FFN）
#   3. 需要调优：num_heads 需要根据任务调整



class ATTFu_GatedFusion(nn.Module):
    """
    方案二：使用门控融合机制
    特点：通过门控单元控制信息流，实现选择性特征融合
    """
    def __init__(self, channels, ratio=0.25):
        super(ATTFu_GatedFusion, self).__init__()
        self.channels = channels
        
        # 模板特征压缩（可选：使用注意力池化代替全局平均池化）
        self.tem_pool = nn.Sequential(
            nn.Linear(channels, channels // 4),
            nn.ReLU(),
            nn.Linear(channels // 4, 1),
            nn.Softmax(dim=1)  # 加权池化而非平均池化
        )
        
        # 门控单元：控制历史信息和模板信息的融合比例
        self.gate = nn.Sequential(
            nn.Linear(channels * 2, channels),
            nn.ReLU(),
            nn.Linear(channels, channels * 2),
            nn.Sigmoid()  # 输出两个门控值
        )
        
        # 特征变换
        self.pro_transform = nn.Sequential(
            nn.Linear(channels, channels),
            nn.LayerNorm(channels),
            nn.GELU()
        )
        
        self.tem_transform = nn.Sequential(
            nn.Linear(channels, channels),
            nn.LayerNorm(channels),
            nn.GELU()
        )
        
        # 融合后的增强网络
        self.fusion_mlp = nn.Sequential(
            nn.Linear(channels, int(channels * 2)),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(int(channels * 2), channels),
            nn.Dropout(0.1)
        )
        
        self.norm = nn.LayerNorm(channels)
        
    def forward(self, l_pro, l_tem):
        """
        args:
            l_pro: (B, 1, 768) - 光谱提示特征
            l_tem: (B, 432, 768) - 模板特征
        returns:
            out: (B, 1, 768) - 更新后的光谱提示特征
        """
        B = l_pro.shape[0]
        
        # 1. 模板特征池化（使用注意力加权池化）
        tem_weights = self.tem_pool(l_tem)  # (B, 432, 1)
        l_tem_pooled = (l_tem * tem_weights).sum(dim=1, keepdim=True)  # (B, 1, 768)
        
        # 2. 特征变换
        l_pro_transformed = self.pro_transform(l_pro)  # (B, 1, 768)
        l_tem_transformed = self.tem_transform(l_tem_pooled)  # (B, 1, 768)
        
        # 3. 拼接并生成门控值
        concat_feat = torch.cat([l_pro_transformed, l_tem_transformed], dim=2)  # (B, 1, 1536)
        gates = self.gate(concat_feat)  # (B, 1, 1536)
        
        # 4. 分离两个门控值
        gate_pro, gate_tem = gates.chunk(2, dim=2)  # 各 (B, 1, 768)
        
        # 5. 门控融合
        fused = gate_pro * l_pro_transformed + gate_tem * l_tem_transformed  # (B, 1, 768)
        
        # 6. MLP 增强 + 残差
        enhanced = self.fusion_mlp(fused)
        out = self.norm(fused + enhanced)
        
        return out

# 特点和优势：
# ✅ 优势：
#   1. 选择性融合：门控机制可以动态决定保留多少历史信息和模板信息
#   2. 计算高效：复杂度 O(N_tem)，但实现简单，参数量适中
#   3. 稳定性好：门控值在 [0,1] 之间，训练稳定
#   4. 可解释性：门控值可以分析信息融合的比例
#   5. 注意力池化：比全局平均池化更灵活，可以关注重要模板区域
# 
# ⚠️ 特点：
#   1. 仍需要池化：虽然使用注意力池化，但仍会丢失部分空间信息
#   2. 门控设计：需要合理设计门控网络结构



class ATTFu_TransformerEncoder(nn.Module):
    """
    方案三：使用 Transformer 编码器
    特点：使用完整的 Transformer 编码器层，实现深度特征交互
    """
    def __init__(self, channels, ratio=0.25, num_layers=2, num_heads=8, final_residual_and_norm=False):
        super(ATTFu_TransformerEncoder, self).__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.final_residual_and_norm = final_residual_and_norm
        assert channels % num_heads == 0, "channels must be divisible by num_heads"
        
        # 位置编码（可选，用于区分 l_pro 和 l_tem）
        self.pos_embed = nn.Parameter(torch.randn(1, 433, channels))  # 1个l_pro + 432个l_tem
        
        # Transformer 编码器层
        # 注意：某些较低版本的 PyTorch 不支持 `norm_first` 参数，这里仅使用通用参数
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=channels,
            nhead=num_heads,
            dim_feedforward=int(channels * 4),
            dropout=0.1,
            activation='gelu',
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 输出投影（只提取 l_pro 对应的输出）
        self.output_proj = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        if final_residual_and_norm:
            self.final_norm = nn.LayerNorm(channels)
        else:
            self.final_norm = None
        
        
    def forward(self, l_pro, l_tem):
        """
        args:
            l_pro: (B, 1, 768) - 光谱提示特征
            l_tem: (B, 432, 768) - 模板特征
        returns:
            out: (B, 1, 768) - 更新后的光谱提示特征
        """
        B = l_pro.shape[0]
        
        # # 1. 拼接 l_pro 和 l_tem
        # # l_pro 放在最前面，这样输出时可以直接取第一个 token
        # seq = torch.cat([l_pro, l_tem], dim=1)  # (B, 433, 768)
        
        # # 2. 添加位置编码
        # seq = seq + self.pos_embed

        

        # 1. 拼接 l_pro 和 l_tem
        # l_pro 放在最前面，这样输出时可以直接取第一个 token
        seq = torch.cat([l_pro, l_tem], dim=1)  # (B, 1+N, 768)
        seq_len = seq.shape[1]
        

        assert seq_len == self.pos_embed.shape[1], "seq_len must be equal to pos_embed.shape[1]"

        # 2. 添加位置编码（动态适配序列长度）
        if seq_len == self.pos_embed.shape[1]:
            # 长度匹配，直接使用
            pos_embed = self.pos_embed
        # elif seq_len < self.pos_embed.shape[1]:
        #     # 测试时序列更短，截取前seq_len个位置编码
        #     pos_embed = self.pos_embed[:, :seq_len, :]
        #     # # 测试时序列更短，截取后seq_len个位置编码
        #     # pos_embed = self.pos_embed[:, -seq_len:, :]
        # else:
        #     # 序列更长，使用插值扩展（虽然理论上不应该发生）
        #     pos_embed = torch.nn.functional.interpolate(
        #         self.pos_embed.transpose(1, 2), 
        #         size=seq_len, 
        #         mode='linear', 
        #         align_corners=False
        #     ).transpose(1, 2)
        
        seq = seq + pos_embed
        
        # 3. Transformer 编码（自注意力 + FFN）
        # 所有 token 之间可以相互交互
        encoded = self.transformer(seq)  # (B, 433, 768)
        
        # 4. 提取 l_pro 对应的输出（第一个 token）
        l_pro_updated = encoded[:, 0:1, :]  # (B, 1, 768)
        
        # 5. 输出投影
        out = self.output_proj(l_pro_updated)

        # 6. 残差连接 + 层归一化
        if self.final_residual_and_norm:
            out = self.final_norm(out + l_pro_updated)
        
        return out

# 特点和优势：
# ✅ 优势：
#   1. 深度交互：多层 Transformer 可以实现深度的特征交互
#   2. 双向信息流：l_pro 和 l_tem 可以双向交互，不仅仅是 l_pro 查询 l_tem
#   3. 保留完整信息：不需要池化，保留所有模板 patch 的信息
#   4. 强大的表达能力：Transformer 架构在特征融合任务上表现优异
#   5. 可扩展性强：可以轻松增加层数或调整结构
# 
# ⚠️ 特点：
#   1. 计算复杂度：O(N_tem^2)，计算量最大
#   2. 参数量：最多（多层 Transformer）
#   3. 内存占用：需要存储所有 token 的中间结果
#   4. 训练难度：可能需要更多的训练技巧（如 warmup、梯度裁剪等）
    

class SSTrack(nn.Module):
    """ This is the base class for SSTrack """

    def __init__(self, transformer, box_head, aux_loss=False, head_type="CORNER",
                 token_len=1, num_searches=2, cfg=None):
        """ Initializes the model.
        Parameters:
            transformer: torch module of the transformer architecture.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()

        if cfg is None:
            raise ValueError("cfg must be provided to SSTrack")

        self.cfg = cfg
        self.backbone = transformer
        self.box_head = box_head

        self.aux_loss = aux_loss
        self.head_type = head_type
        if head_type == "CORNER" or head_type == "CENTER":
            self.feat_sz_s = int(box_head.feat_sz)
            self.feat_len_s = int(box_head.feat_sz ** 2)

        if self.aux_loss:
            self.box_head = _get_clones(self.box_head, 6)
        
        # track query: temporal state for test/tracking (see reset_track_query)
        self.track_query = None
        self.token_len = token_len
        self.num_searches = num_searches
        self.token_type = cfg.MODEL.BACKBONE.ATTN_TYPE
        
        # self.prompt = ATTFu(768)
        # # 方案1：使用交叉注意力机制
        # self.prompt = ATTFu_CrossAttention(768, num_heads=8)
        # # 方案2：使用门控融合机制
        # self.prompt = ATTFu_GatedFusion(768)
        # # 方案3：使用 Transformer 编码器
        # self.prompt = ATTFu_TransformerEncoder(768, num_layers=2, num_heads=8)
        

        # 原始方案：FC注意力 + MLP
        if self.cfg.MODEL.PROMPT_TYPE == "original":
            self.prompt = ATTFu(768)
        # 方案1：使用交叉注意力机制
        elif self.cfg.MODEL.PROMPT_TYPE == "cross_attn":
            self.prompt = ATTFu_CrossAttention(768, num_heads=8)
        # 方案2：使用门控融合机制
        elif self.cfg.MODEL.PROMPT_TYPE == "gate":
            self.prompt = ATTFu_GatedFusion(768)
        # 方案3：使用 Transformer 编码器
        elif self.cfg.MODEL.PROMPT_TYPE == "trans_enc":
            self.prompt = ATTFu_TransformerEncoder(768, num_layers=2, num_heads=8)
        else:
            raise ValueError("Unknown PROMPT_TYPE")

    def _backbone_template_drop_kw(
            self,
            template_drop_rate,
            template_drop_mode="null",
            template_drop_strategy="random",
            template_drop_query=None,
            template_hard_ratio=0.5,
    ):
        if isinstance(self.backbone, VisionTransformerCE):
            return {
                "template_drop_rate": template_drop_rate,
                "template_drop_mode": template_drop_mode,
                "template_drop_strategy": template_drop_strategy,
                "template_drop_query": template_drop_query,
                "template_hard_ratio": template_hard_ratio,
            }
        return {}

    def reset_track_query(self):
        """Clear temporal track query (new sequence / new training-val batch)."""
        self.track_query = None

    def _cvtp_runtime(self, cvtp_template_drop_rate=None) -> _CVTPRuntime:
        """解析 MODEL.CVTP：仅在训练且 ENABLE 时启用；推理或未配置时返回 enabled=False、drop_rate=0。"""
        cvtp_cfg = getattr(self.cfg.MODEL, "CVTP", None)
        enabled = self.training and cvtp_cfg is not None and bool(getattr(cvtp_cfg, "ENABLE", False))
        rate = float(cvtp_template_drop_rate) if cvtp_template_drop_rate is not None else 0.0
        return _CVTPRuntime(
            enabled=enabled,
            drop_rate=rate if enabled else 0.0,
            strategy=str(getattr(cvtp_cfg, "DROP_STRATEGY", "random")) if cvtp_cfg is not None else "random",
            hard_ratio=float(getattr(cvtp_cfg, "HARD_RATIO", 0.5)) if cvtp_cfg is not None else 0.5,
        )

    def forward(self, template: torch.Tensor,
                search: torch.Tensor,
                ce_template_mask=None,
                ce_keep_rate=None,
                return_last_attn=False,
                cvtp_template_drop_rate=None,
                ):
        assert isinstance(search, list), "The type of search is not List"

        track_query = self.track_query

        cvtp = self._cvtp_runtime(cvtp_template_drop_rate)

        out_dict = []
        for i in range(self.num_searches-1, len(search)): # self.num_searches = 2 = DATA.SEARCH.LENGTH  len(search) = DATA.SEARCH.NUMBER。这里表示dataloader取了3帧，这里可以滑动两次，每次取2帧。 i表示的是当前最后帧的索引。
            use_cvtp_drop = cvtp.enabled and (track_query is not None)
            cur_drop = cvtp.drop_rate if use_cvtp_drop else 0.0
            drop_query = track_query if use_cvtp_drop else None
            # search的提取做了修改，以获取list形式的search
            # 返回的x_大致为 cls_token + 模板tokens + 搜索tokens
            # self.feat_len_s = 576
            x_, aux_dict, top_k_indices = self.backbone(
                z=template.copy(),
                x=[search[idx] for idx in range(i - self.num_searches + 1, i + 1)],
                ce_template_mask=ce_template_mask,
                ce_keep_rate=ce_keep_rate,
                return_last_attn=return_last_attn,
                track_query=track_query,
                token_len=self.token_len,
                **self._backbone_template_drop_kw(
                    cur_drop,
                    "null",
                    template_drop_strategy=cvtp.strategy,
                    template_drop_query=drop_query,
                    template_hard_ratio=cvtp.hard_ratio,
                ),
                token_type=self.token_type,
            )
            # search部分只保留最后一个search的特征图
            x = torch.cat((x_[:, :-1 * self.num_searches * self.feat_len_s, :], x_[:, -self.feat_len_s:, :]), dim=1)

            feat_last = x   # x.shape torch.Size([Bs, 1009, 768]); 1009 = 1 + 3*144 + 576
            if isinstance(x, list):
                feat_last = x[-1]

            enc_opt = feat_last[:, -self.feat_len_s:]  # encoder output for the search region (B, HW, 576)    # enc_opt.shape torch.Size([Bs, 576, 768])
            if self.backbone.add_cls_token:
                t_query = x[:, :self.token_len]
                z_query = x[:, self.token_len:-self.feat_len_s]
                track_query = self.prompt(t_query.detach(), z_query.detach()) #只训练self.prompt

            att = torch.matmul(enc_opt, x[:, :1].transpose(1, 2))  # (B, HW, N)
            opt = (enc_opt.unsqueeze(-1) * att.unsqueeze(-2)).permute((0, 3, 2, 1)).contiguous()  # (B, HW, C, N) --> (B, N, C, HW)

            # print((int((enc_opt.abs().amax(dim=-1) < 1e-6).sum().item()), int(enc_opt.shape[1])))
            
            # Forward head
            out = self.forward_head(opt, None)

            out.update(aux_dict)
            out['backbone_feat'] = x

            out_dict.append(out)

        if not self.training:
            self.track_query = track_query

        return out_dict

    def forward_head(self, opt, gt_score_map=None):
        """
        enc_opt: output embeddings of the backbone, it can be (HW1+HW2, B, C) or (HW2, B, C)
        """
        # opt = (enc_opt.unsqueeze(-1)).permute((0, 3, 2, 1)).contiguous()
        bs, Nq, C, HW = opt.size()
        opt_feat = opt.view(-1, C, self.feat_sz_s, self.feat_sz_s)

        if self.head_type == "CORNER":
            # run the corner head
            pred_box, score_map = self.box_head(opt_feat, True)
            outputs_coord = box_xyxy_to_cxcywh(pred_box)
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map,
                   }
            return out

        elif self.head_type == "CENTER":
            # run the center head
            score_map_ctr, bbox, size_map, offset_map = self.box_head(opt_feat, gt_score_map)
            
            # outputs_coord = box_xyxy_to_cxcywh(bbox)
            outputs_coord = bbox
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            
            out = {'pred_boxes': outputs_coord_new,
                    'score_map': score_map_ctr,
                    'size_map': size_map,
                    'offset_map': offset_map}
            
            return out
        else:
            raise NotImplementedError


def build_sstrack(cfg, training=True):
    current_dir = os.path.dirname(os.path.abspath(__file__))  # This is your Project Root
    pretrained_path = os.path.join(current_dir, '../../../pretrained_networks')
    if cfg.MODEL.PRETRAIN_FILE and ('SSTrack' not in cfg.MODEL.PRETRAIN_FILE) and training:
        pretrained = os.path.join(pretrained_path, cfg.MODEL.PRETRAIN_FILE)
    else:
        pretrained = ''

    if cfg.MODEL.BACKBONE.TYPE == 'vit_base_patch16_224':
        backbone = vit_base_patch16_224(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                                        add_cls_token=cfg.MODEL.BACKBONE.ADD_CLS_TOKEN,
                                        attn_type=cfg.MODEL.BACKBONE.ATTN_TYPE,)
        
    elif cfg.MODEL.BACKBONE.TYPE == 'vit_base_patch16_224_ce':
        backbone = vit_base_patch16_224_ce(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                                           ce_loc=cfg.MODEL.BACKBONE.CE_LOC,
                                           ce_keep_ratio=cfg.MODEL.BACKBONE.CE_KEEP_RATIO,
                                           add_cls_token=cfg.MODEL.BACKBONE.ADD_CLS_TOKEN,
                                           )

    else:
        raise NotImplementedError
    hidden_dim = backbone.embed_dim
    patch_start_index = 1
    
    backbone.finetune_track(cfg=cfg, patch_start_index=patch_start_index)

    box_head = build_box_head(cfg, hidden_dim)

    model = SSTrack(
        backbone,
        box_head,
        aux_loss=False,
        head_type=cfg.MODEL.HEAD.TYPE,
        token_len=cfg.MODEL.BACKBONE.TOKEN_LEN,
        num_searches=cfg.DATA.SEARCH.LENGTH,
        cfg=cfg,
    )

    return model
