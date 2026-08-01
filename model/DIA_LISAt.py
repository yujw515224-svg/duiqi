import torch
import torch.nn as nn
import torch.nn.functional as F




class ContextEvidenceAdapter(nn.Module):
    """
    DIA-LISA 的 Context-to-Evidence Adapter。

    输入:
        con_embeddings: [N, C]
            N 是当前图像里 [CON] token 的数量，通常等于 mask 数量。
            C 是 SAM prompt embedding 维度，LISAt 里通常是 256。

        image_embeddings: [1, C, H, W]
            SAM image encoder 输出的 dense image feature。
            这里 batch=1，因为 LISAt 当前是逐图送进 SAM mask decoder。

        image_pe: [1, C, H, W] or None
            SAM prompt encoder 的 dense positional encoding。

    输出:
        evidence_tokens: [N, K, C]
            每个 [CON] 从图像里检索出来的 K 个视觉证据 token。

        attn_maps: [N, K, H, W]
            每个 evidence token 对 SAM feature map 的注意力分布。
            后续用 GT mask 对这个图做 attention alignment loss。
    """
    def __init__(self, dim=256, num_heads=8, num_evidence_tokens=4, dropout=0.1):
        super().__init__()
        self.num_evidence_tokens = num_evidence_tokens
        # 把 [CON] embedding 投影成 cross-attention 的 query 空间。
        self.query_proj = nn.Linear(dim, dim)
        # 给一个 [CON] 派生出 K 个不同的 evidence queries。
        # 直觉上：同一个目标可能需要多个视觉证据，例如主体、边界、上下文。
        self.query_offsets = nn.Parameter(torch.zeros(num_evidence_tokens, dim))
        # query 来自 [CON]，key/value 来自 SAM image feature。
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        # 稳定 evidence token 的尺度，防止后面 fusion 训练不稳。
        self.out_norm = nn.LayerNorm(dim)
    
    def forward(self, con_embeddings, image_embeddings, image_pe=None):

        b, c, h, w = image_embeddings.shape
        assert b == 1, "LISAt decodes one image at a time before SAM mask decoding."

        # [1, C, H, W] -> [1, H*W, C]
        # 每个空间位置变成一个 image token。
        image_tokens = image_embeddings.flatten(2).transpose(1, 2)

        # 加位置编码，让 adapter 知道 evidence 来自图像的哪个空间位置。
        if image_pe is not None:
            image_tokens = image_tokens + image_pe.flatten(2).transpose(1, 2).to(image_tokens)
        
        # [N, C] -> [N, 1, C] -> [N, K, C]
        # 每个 [CON] 变成 K 个 evidence query。
        query = self.query_proj(con_embeddings).unsqueeze(1) + self.query_offsets.unsqueeze(0)
        key_value = image_tokens.expand(query.shape[0], -1, -1)


        # evidence_tokens: [N, K, C]
        # attn_probs: [N, K, H*W]，因为 average_attn_weights 默认会平均多个 head。
        evidence_tokens, attn_probs = self.cross_attn(
            query=query,
            key=key_value,
            value=key_value,
            need_weights=True,
        )
        evidence_tokens = self.out_norm(evidence_tokens)

        attn_maps = attn_probs.view(query.shape[0], query.shape[1], h, w)

        return evidence_tokens, attn_maps
    


class EvidenceGuideFusion(nn.Module):
    """
    把 [SEG]、[CON]、视觉 evidence 融合成 SAM 可用的 sparse prompt tokens。

    输出 prompt_tokens 的形状是:
        [N, 1 + K, C]

    第 1 个 token 是 fused [SEG] prompt。
    后 K 个 token 是 evidence tokens。
    """
    def __init__(self, dim=256, dropout=0.1):
        super().__init__()

        # gate 控制 evidence 对 [SEG] 的影响强度。
        # 如果 evidence 暂时不可靠，模型可以学到较小 gate，保留原始 [SEG]。
        self.gate = nn.Sequential(
            nn.Linear(dim * 3, dim),
            nn.Sigmoid(),
        )

        # 真正执行融合的 MLP
        self.fuse = nn.Sequential(
            nn.LayerNorm(dim * 3),
            nn.Linear(dim*3, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )

        self.out_norm = nn.LayerNorm(dim)
    
    def forward(self, seg_embeddings, con_embeddings, evidence_tokens):
        # evidence_tokens: [N, K, C]
        # 先平均成 [N, C]，作为当前目标的视觉证据摘要。
        evidence_summary = evidence_tokens.mean(dim=1)

        # 拼接三种信息:
        # [SEG]：mask prompt 语义
        # [CON]：概念/上下文 query 语义
        # evidence：从图像里检索到的视觉证据
        fused_input = torch.cat(
            [seg_embeddings, con_embeddings, evidence_summary],
            dim = -1,
        )

        gate = self.gate(fused_input)

        # residual 设计:
        # fused_seg = 原始 [SEG] + gate * 新证据
        # 这样即使 evidence 早期没学好，也不会完全破坏原 LISA 路径。
        fused_seg = seg_embeddings + gate * self.fuse(fused_input)
        fused_seg = self.out_norm(fused_seg)

        # SAM prompt_encoder 支持多个 text prompt token。
        # 所以这里把 fused [SEG] 和 evidence tokens 一起送进去。
        prompt_tokens = torch.cat(
            [fused_seg.unsqueeze(1), evidence_tokens],
            dim = 1,
        )

        return prompt_tokens


def attention_alignment_loss(attn_maps, gt_masks, eps=1e-6):
    """
    用 GT mask 监督 [CON] 的 attention map。

    attn_maps:
        [N, K, H, W]
        N 是 mask 数量，K 是 evidence token 数量。

    gt_masks:
        [N, H_gt, W_gt]
        原始 GT segmentation mask。

    思路:
        把 GT mask resize 到 attention map 大小；
        把 GT mask 归一化为空间概率分布；
        让 attention 分布靠近 GT 分布。
    """

    # 多个 evidence token 的 attention 先求平均。
    # [N, K, H, W] -> [N, H, W]
    attn = attn_maps.mean(dim=1)

    # GT mask resize 到 attention map 的空间分辨率。
    target = F.interpolate(
        gt_masks.unsqueeze(1).float(),
        size=attn.shape[-2:],
        mode="nearest",
    ).squeeze(1)
    # [N, H, W] -> [N, H*W]

    target = target.flatten(1)

    # 把 GT mask 归一化成概率分布。
    # clamp_min 防止空 mask 或极小目标导致除 0。
    target = target / target.sum(dim=-1, keepdim=True).clamp_min(eps)
    # attention 本身已经是 softmax 出来的概率，但这里仍然 clamp，防止 log(0)。
    attn = attn.flatten(1).clamp_min(eps)
    # Cross entropy: - sum p_gt * log p_attn
    loss = -(target * attn.log()).sum(dim=-1).mean()

    return loss

