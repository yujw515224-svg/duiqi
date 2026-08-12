# DIA-LISA `[SEG]` Decoupling Design

## Objective

The explicit DIA path should give each token one testable responsibility:

- `[CON]` contains the requested concept and retrieves local visual evidence.
- `[SEG]` is a mask-decoding command, not a second concept representation.
- SAM receives a shared decoding anchor plus evidence retrieved by `[CON]`.

The current `faithful_evidence_fusion` path is a safe migration baseline, but it
still sends the instance-specific projected `[SEG]` hidden state to SAM. Because
that hidden state has attended to the whole instruction and to `[CON]`, semantic
information can bypass the evidence adapter. An attention loss alone cannot
prevent this shortcut.

## Proposed token-to-mask path

For prompt `p` and SAM feature map `F`:

1. Project the `[CON]` hidden state only: `q_p = P_con(h_con,p)`.
2. Retrieve evidence: `(e_p, A_p) = CrossAttention(q_p, F + PE, F)`.
3. Start mask decoding from a prompt-shared learned anchor `s0`, rather than
   `P_seg(h_seg,p)`.
4. Form the decoder prompt with an evidence-only residual:
   `s_p = s0 + alpha * G(e_p, s0)`.
5. Pass `s_p` to the existing SAM prompt encoder and mask decoder unchanged.

`[SEG]` remains in the language sequence to declare that a mask is requested.
Its occurrence determines prompt count and routing, but its instance-specific
hidden state does not carry target semantics into SAM. This removes the direct
`instruction -> h_SEG -> mask` shortcut.

## Required code changes

### 1. Add a strictly decoupled prompt adapter

Add `DecoupledMaskPrompt` to `model/DIA_LISAt.py` with:

- `decoder_anchor: nn.Parameter` shaped `[1, out_dim]`;
- an evidence-only, bias-free MLP from `evidence_tokens.mean(1)` to a residual;
- zero initialization on the last projection for checkpoint-safe startup;
- a bounded residual ratio, as in `FaithfulEvidenceFusion`;
- no `seg_embeddings` values in its computation (only `num_prompts` and dtype /
  device may be taken from the `[SEG]` routing tensor).

Its API should make leakage difficult to reintroduce:

```python
prompt_tokens = decoupled_mask_prompt(
    evidence_tokens=evidence_tokens,
    num_prompts=seg_i.shape[0],
)
```

Do not accept `seg_embeddings` as an argument. A unit test should perturb the
`[SEG]` hidden state and prove that the generated sparse prompt is unchanged.

### 2. Add a dedicated fusion mode

Add `decoupled_evidence_prompt` to `--dia_fusion_mode` in `train_lisat.py` and
to the model construction, validation, checkpoint loading, trainable-parameter,
and structure-validation branches in `model/LISAT.py`.

In `generate_pred_masks`, keep the existing `[CON] -> context_adapter` retrieval,
but select `DecoupledMaskPrompt` before the unchanged SAM prompt encoder. Do not
add evidence directly to the full SAM image embedding in this mode; doing so
makes it harder to attribute gains to the proposed token-to-mask adapter.

### 3. Supervise evidence, not only the final mask

Keep the positive attention alignment loss, but add an evidence-presence head
computed from the retrieved evidence or pre-softmax localization logits:

- positive edited view: presence target `1`, attention target is the GT mask;
- target-removed negative view: presence target `0`, no mask decode and no KL
  normalization of an empty mask;
- optional hard negative: an unedited image containing a visually similar,
  wrong concept.

Use separate loss terms and logs:

```text
L = L_lm + lambda_mask * L_mask
         + lambda_align * L_attn_positive
         + lambda_pres * L_presence
```

A spatial softmax always sums to one, so it cannot express “no evidence”. The
presence head (or sigmoid localization map used by `evidence_feedback`) is
therefore necessary for meaningful negative supervision.

### 4. Allow concept-only negative routing

The current explicit protocol assumes every `[CON]` is immediately followed by
`[SEG]`. Extend `dataloaders/trainval_dataset.py` to support two valid forms:

- positive segmentation: `[CON][SEG]`;
- absent-target negative: `[CON]` without `[SEG]`.

Collation should return explicit concept-to-image and segment-to-concept indices.
Do not infer pairing only from equal token counts. During training, run the
context adapter for both forms, compute presence loss for both, and invoke SAM
only for entries with `[SEG]` and a mask target.

### 5. Prevent text-label leakage in edited data

In `dataloaders/dia_align_aug_dataset.py`, build descriptions from the edited
view. A target-removed sample must not reuse the removed GT mask's exact
location, extent, or shape in its text. Positive and negative prompts should use
the same template family so wording does not reveal the label.

Prefer offline image editing before SAM / CLIP preprocessing. Editing normalized
padded tensors can create preprocessing artifacts that become an easier negative
cue than semantic absence.

## Training schedule

1. **Evidence stage:** freeze SAM and the `[SEG]` decoder anchor; train the
   `[CON]` projector, cross-attention, localization/presence heads with positive
   and negative edited pairs.
2. **Fusion stage:** enable `DecoupledMaskPrompt` and the SAM mask decoder while
   retaining evidence losses at a lower weight.
3. **Joint stabilization:** optionally unfreeze small LoRA blocks, but keep the
   direct projected `[SEG]`-hidden-to-SAM path disabled.

Initialize the learned anchor from the mean projected `[SEG]` embedding of a
small training subset or distill it from the original LISAt prompt. Zero-init the
evidence residual output so the new adapter starts from this stable anchor.

## Acceptance tests

The implementation is decoupled only if all of these hold:

1. Changing `h_SEG` while keeping `[CON]`, image features, and prompt count fixed
   does not change the SAM sparse prompt or predicted mask.
2. Changing `h_CON` changes the attention map and can change the mask.
3. Zero evidence produces exactly the shared anchor prompt.
4. Removed-target negatives produce a low presence score and never enter mask
   loss or SAM decoding.
5. Gradients from attention/presence losses reach the `[CON]` projector and
   cross-attention; final mask gradients reach the evidence prompt adapter.
6. A text-only ablation with shuffled SAM features performs near chance,
   demonstrating that the old `[SEG]` semantic shortcut is unavailable.

## Recommended first experiment

Use `K=1`, eight attention heads, zero attention dropout, and the existing
attention loss weight as the initial baseline. Compare:

- original LISAt;
- current `faithful_evidence_fusion`;
- `decoupled_evidence_prompt` without negative presence loss;
- the full decoupled mode with positive/negative presence supervision.

Report mask IoU, presence accuracy, positive attention IoU, negative maximum
presence, and the shuffled-feature ablation. The last two metrics are important:
mask IoU alone cannot prove that `[SEG]` was actually decoupled.
