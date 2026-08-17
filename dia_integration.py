# -*- coding: utf-8 -*-
"""Glue between ``train_lisat.py`` and DIA-LISAt.

Keeps the training script edits down to three lines (see docs/DIA_README.md):

    from dia_integration import add_dia_args, build_dia_model_args, DIA_LOG_KEYS
    ...
    add_dia_args(parser)                       # inside parse_args()
    model_args.update(build_dia_model_args(args))
    tokenizer, model, vision_tower = init_dia_lisat_model(args, model_args)
"""

from model.dia_lisat_model import DIA_DEFAULTS

__all__ = ["DIA_LOG_KEYS", "add_dia_args", "build_dia_model_args"]

# Extra scalars returned by DIALISATForCausalLM.model_forward; add them to the
# AverageMeter list of the training loop to watch the alignment come up.
DIA_LOG_KEYS = [
    "attn_loss",       # the alignment loss itself
    "attn_mass",       # share of [CON] attention that lands inside the GT mask
    "dia_gate",        # mean fusion gate
    "dia_delta_ratio", # ||z - p|| / ||p||   (how much evidence moved the prompt)
    "dia_con_hit_rate",# fraction of [SEG] that found a preceding [CON]
]


def add_dia_args(parser):
    """Register the DIA flags on an existing ``argparse`` parser."""
    group = parser.add_argument_group("DIA-LISAt")
    group.add_argument(
        "--dia_attn_loss_weight",
        type=float,
        default=DIA_DEFAULTS["dia_attn_loss_weight"],
        help="lambda_attn of L = L_txt + L_mask + lambda_attn * L_attn.",
    )
    group.add_argument(
        "--dia_attn_loss_mode",
        choices=["mass", "kl"],
        default=DIA_DEFAULTS["dia_attn_loss_mode"],
        help="mass: -log(attention mass inside GT); kl: match the full map.",
    )
    group.add_argument("--dia_embed_dim", type=int, default=DIA_DEFAULTS["dia_embed_dim"])
    group.add_argument("--dia_num_heads", type=int, default=DIA_DEFAULTS["dia_num_heads"])
    group.add_argument("--dia_dropout", type=float, default=DIA_DEFAULTS["dia_dropout"])
    group.add_argument(
        "--dia_fusion_hidden_dim",
        type=int,
        default=DIA_DEFAULTS["dia_fusion_hidden_dim"],
    )
    group.add_argument(
        "--dia_max_delta_ratio",
        type=float,
        default=DIA_DEFAULTS["dia_max_delta_ratio"],
        help="Cap on ||z - p|| / ||p||. <=0 disables the cap.",
    )
    group.add_argument(
        "--dia_use_dense_pe",
        dest="dia_use_dense_pe",
        action="store_true",
        default=DIA_DEFAULTS["dia_use_dense_pe"],
        help="Add SAM's dense positional encoding to the adapter keys.",
    )
    group.add_argument(
        "--no_dia_use_dense_pe", dest="dia_use_dense_pe", action="store_false"
    )
    group.add_argument(
        "--dia_lr_multiplier",
        type=float,
        default=1.0,
        help=(
            "LR multiplier for the freshly-initialised DIA modules. 1.0 keeps a "
            "single optimizer group (recommended default)."
        ),
    )
    group.add_argument(
        "--con_style",
        choices=["clause", "adjacent", "none"],
        default="clause",
        help=(
            "How [CON] is injected into the answer templates. 'none' disables "
            "the rewrite: the model then falls back to using the [SEG] state as "
            "its own concept query (ablation of the decoupling itself)."
        ),
    )
    group.add_argument(
        "--init_con_from_seg",
        dest="init_con_from_seg",
        action="store_true",
        default=True,
        help="Warm-start the [CON] embedding row from [SEG].",
    )
    group.add_argument(
        "--no_init_con_from_seg", dest="init_con_from_seg", action="store_false"
    )
    return parser


def build_dia_model_args(args) -> dict:
    """Collect the DIA hyper-parameters that must reach the model constructor."""
    return {name: getattr(args, name) for name in DIA_DEFAULTS}
