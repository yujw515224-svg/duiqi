# DIA-LISAt：面向遥感分割的图文解耦对齐

在 **LISAt** 基座上实现结构图中的方法：用 `[CON]` 显式承载「目标概念」，通过
concept→evidence 的 cross-attention 与视觉证据做局部对齐，再把对齐后的证据反馈给
`[SEG]` 做 mask 解码。**主干不动**（RemoteCLIP、Vicuna-7B+LoRA、SAM 编码器/解码器、
`text_hidden_fcs` 全部保持原样），只在 token→mask 这条路径上加了两个轻量模块（约 2M 参数）。

---

## 1. 方法与结构图的对应

```
                      SAM Image Encoder (frozen)
 Remote Sensing I ───────────────────────────────► F ∈ R^{C×H×W}  (C=256, H=W=64)
                                                    │ Key/Value
 Text T ──► MLLM (LISAt) ──► hidden states          │
                 ├── h_CON ──Query──► ┌─────────────▼──────────────┐
                 │                    │ Concept-to-Evidence Adapter │
                 │                    │  A = softmax(qKᵀ/√d)        │──► A (注意力图)
                 │                    │  e = A·V                    │──► e (证据 token)
                 │                    └─────────────┬──────────────┘
                 │                                  │
                 └── h_SEG ──text_hidden_fcs──► p ──┴──► Evidence-guided Fusion
                                                          z = p + clip(g ⊙ MLP([p;e]))
                                                                  │
                              SAM Mask Decoder (image F + prompt z) ──► M_pred
```

| 结构图里的模块 | 代码位置 |
| --- | --- |
| `<CON>` concept query | `model/dia_lisat_model.py` 的 `con_token_idx` + `dataloaders/dia_conversation.py` |
| Concept-to-Evidence Adapter（Cross-Attention / Softmax / Attention Map） | `ConceptToEvidenceAdapter`（`model/dia_modules.py`） |
| Evidence-guided Fusion（Fuse → Prompt Embedding z） | `EvidenceGuidedFusion`（`model/dia_modules.py`） |
| Attention Alignment Loss L_attn | `attention_alignment_loss`（`model/dia_modules.py`） |
| SAM Mask Decoder / Mask Loss | 完全复用 LISAt（`decode_masks_with_sam`） |

### 公式

设 `h_CON, h_SEG ∈ R^{4096}` 为 MLLM 末层在两个特殊 token 位置的隐状态，
`F ∈ R^{C×HW}` 为 SAM 的 dense image features：

```
q  = W_q · LN(h_CON)                      # 概念查询
K  = W_k · LN(F + PE),  V = W_v · LN(F + PE)
A  = softmax(qKᵀ / √d_h)  ∈ Δ^{HW}        # 概念→视觉证据的注意力图
e  = W_o (A · V)                          # 证据 token

p  = text_hidden_fcs(h_SEG)               # LISAt 原本的 SAM prompt
g  = σ(MLP_g([LN(p); LN(e)]))
z  = p + clip( g ⊙ MLP_Δ([LN(p); LN(e)]) )   s.t. ‖z−p‖ ≤ ρ‖p‖

M_pred = SAM_dec(F, PromptEnc(z))
L = L_txt + λ_bce·BCE + λ_dice·DICE + λ_attn·L_attn
L_attn = −log( Σ_{i∈M_gt} A_i )           # 落在目标区域内的注意力质量
```

三个刻意的设计选择：

1. **`MLP_Δ` 最后一层零初始化** ⇒ 第 0 步 `z ≡ p`，训练从 LISAt 基线**精确**出发，
   适配器只能靠 mask loss 的梯度"挣"到影响力，不会一上来就破坏已训练好的 prompt 分布。
2. **残差上限 `ρ`（`--dia_max_delta_ratio`，默认 0.5）**：SAM decoder 对 prompt 分布很敏感，
   限幅可避免证据把 prompt 推出分布外。
3. **`L_attn` 用 "mass" 形式而非逐像素 BCE/KL**：遥感目标常常只占 64×64 特征图里的 1~2 格，
   逐像素损失会被背景淹没；`−log(mask 内注意力质量)` 与目标尺度无关。GT 下采样用
   `adaptive_max_pool2d`，保证 4×4 像素的小目标不会在 64×64 上消失
   （对应 `tests/test_dia_modules.py::test_alignment_loss_keeps_tiny_objects`）。

### `[CON]` 怎么进对话

答案模板里的每个 `[SEG]` 被重写为（默认 `clause` 风格）：

```
"Sure, it is [SEG]."  ->  "Sure, it is [CON], so the segmentation result is [SEG]."
```

**为什么不用相邻的 `[CON][SEG]`**：LISA/LISAt 取的是「产生该 token 的那个隐状态」
（`input_ids[:, 1:]` 的位移约定）。如果两个 token 紧挨着，`[SEG]` 的 prompt 就直接读到
`[CON]` 位置的隐状态，两个角色又缠在一起了——正是本方法要拆开的东西。
`clause` 风格在两者之间留了若干普通 token。若仍想做对照实验，可用 `--con_style adjacent`。

多目标场景下每个 `[SEG]` 都会配一个自己的 `[CON]`；配对规则是「同一轮对话中最近的前置
`[CON]`」。若某个 `[SEG]` 前面没有可用的 `[CON]`（例如推理时模型漏生成了），
自动退化为用 `h_SEG` 自己做 query —— 即退回 LISAt 行为，不会崩。

---

## 2. 仓库里的文件

DIA 已经接进主训练流程，**不需要再打补丁**。

| 文件 | 说明 |
| --- | --- |
| `model/dia_modules.py` | 纯 torch 实现：适配器 / 融合 / 对齐损失 / 批次索引 / SAM 解码 |
| `model/dia_lisat_model.py` | `DIALISATForCausalLM`、`init_dia_lisat_model`、`load_pretrained_model_DIA_LISAT` |
| `dataloaders/dia_conversation.py` | `[CON]` 注入与校验 |
| `dia_integration.py` | 训练脚本胶水：`add_dia_args` / `build_dia_model_args` / `DIA_LOG_KEYS` |
| `tests/test_dia_modules.py` | 23 个 CPU 单测（无需权重/GPU） |
| `tests/test_dia_batching.py` | 9 个批次索引与解码单测 |
| `train_dia_lisat.sh` | DIA 训练启动示例 |
| `train_lisat.sh` | 上游 LISAt 基线启动脚本（做对照实验用） |

`model/LISAT.py` 是**上游原版**，一行没改；DIA 全部以子类形式叠在它上面。

### 已接入的位置

| 文件 | 改了什么 |
| --- | --- |
| `train_lisat.py` | `init_dia_lisat_model` 替代 `init_LISAT_model`；`add_dia_args(parser)`；`model_args.update(build_dia_model_args(args))`；日志键 `= 基础键 + DIA_LOG_KEYS` |
| `dataloaders/trainval_dataset.py` | `collate_fn_train/val` 增加 `con_style` 参数，在 `tokenize_and_pad` **之前**注入 `[CON]`（这样 label 掩码不会错位） |
| `eval_lisat.py` | 改用 `load_pretrained_model_DIA_LISAT` 加载 DIA checkpoint |

`evaluate()` 的签名与 LISAt 完全一致（只多一个可选的 `return_attention=True`，
用来导出注意力图做可视化），所以其余评估代码不受影响。
`merge_lora_weights_and_save_hf_model.py` 若要合并 DIA checkpoint，需把
`LISATForCausalLM` 换成 `DIALISATForCausalLM` 并同样 `add_tokens("[CON]")`。

---

## 3. 跑起来

```bash
python tests/test_dia_modules.py && python tests/test_dia_batching.py
```

```bash
bash train_dia_lisat.sh
```

关键超参（全部有默认值，可直接不填）：

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--dia_attn_loss_weight` | 0.1 | λ_attn。先用 0.1；若 mask loss 被带偏就降到 0.02~0.05 |
| `--dia_attn_loss_mode` | `mass` | `mass`（推荐）或 `kl` |
| `--dia_max_delta_ratio` | 0.5 | prompt 残差上限；调到 0.1 更保守，≤0 关闭 |
| `--dia_num_heads` / `--dia_embed_dim` | 8 / 256 | 适配器容量 |
| `--con_style` | `clause` | `clause` / `adjacent` |
| `--no_dia_use_dense_pe` | – | 关掉给 K/V 加 SAM 位置编码（消融用） |

### 日志怎么读

| 字段 | 期望行为 |
| --- | --- |
| `dia_con_hit_rate` | **应当≈1.0**。若为 0，说明 `[CON]` 没进对话（3.1 步没生效），此时模型已退化成 LISAt |
| `attn_mass` | `[CON]` 注意力落在 GT 内的比例，应从 ~0.0x 稳步上升；这是"对齐是否学到"的直接证据 |
| `attn_loss` | 随 `attn_mass` 上升而下降 |
| `dia_delta_ratio` | 从 0 开始增长（零初始化）。若长期贴着 `--dia_max_delta_ratio` 说明限幅太紧 |
| `dia_gate` | 融合门均值，0.5 附近属正常 |

---

## 4. 建议的消融

| 设置 | 命令 | 想说明的问题 |
| --- | --- | --- |
| baseline | `bash train_lisat.sh` | LISAt |
| + 适配器，无对齐监督 | `--dia_attn_loss_weight 0` | 光加参数有没有用 |
| **完整 DIA** | 默认 | 主结果 |
| 概念查询用 `h_SEG` | `--con_style none` | **`[CON]` 解耦本身的贡献**（无 `[CON]` 时自动退化为用 `h_SEG` 当 query） |
| `--con_style adjacent` | | 相邻放置导致的角色再纠缠 |
| λ_attn ∈ {0.02,0.05,0.1,0.2} | | 对齐监督强度 |
| `--no_dia_use_dense_pe` | | 位置编码对遥感小目标定位的作用 |

指标：GRES 测试集 gIoU/cIoU（All / Small / Large）+ `attn_mass`。
`attn_mass` 是本方法特有的可报告指标，能直接支撑「显式缓解视觉过载」的论述；
配合 `evaluate(..., return_attention=True)` 导出的注意力图可做定性可视化（结构图右下角那种热力图）。

---

## 5. 常见问题

- **`dia_con_hit_rate` 一直是 0** → `collate_fn` 没注入 `[CON]`（检查 `--con_style`），
  或 tokenizer 没加 `[CON]`（必须用 `init_dia_lisat_model` 建模型）。
- **显存**：适配器对 64×64=4096 个 key 做单 query attention，开销可忽略；
  每个 `[SEG]` 会 gather 一份 image feature（bf16 下约 2MB），batch 很大时注意。
- **resume**：DIA 超参会写进 `config`，从 DIA checkpoint 续训时不必重复传参；
  但从**基线 LISAt** checkpoint 起步时，`[CON]` 行会用 `[SEG]` 行热启动
  （`--no_init_con_from_seg` 可关闭）。
- **训练不稳定**：先把 `--dia_attn_loss_weight` 降到 0.02、`--dia_max_delta_ratio` 降到 0.1；
  由于零初始化，DIA 在最坏情况下只会退化成 LISAt，不应该出现"比基线差很多"的情况——
  如果出现了，先查 `dia_delta_ratio` 是不是瞬间冲顶。
