import torch

from model.DIA_LISAt import DecoupledMaskPrompt


def test_decoupled_prompt_zero_init_is_shared_anchor():
    module = DecoupledMaskPrompt(dim=8, hidden_dim=12, max_delta_ratio=0.2)
    evidence = torch.randn(3, 1, 8)
    output = module(evidence, num_prompts=3)
    expected = module.decoder_anchor.expand(3, -1)
    torch.testing.assert_close(output[:, 0], expected)


def test_decoupled_prompt_has_no_seg_embedding_input_and_uses_evidence():
    module = DecoupledMaskPrompt(dim=8, hidden_dim=12, max_delta_ratio=0.2)
    torch.nn.init.normal_(module.evidence_out.weight, std=0.1)
    evidence_a = torch.randn(2, 1, 8)
    evidence_b = evidence_a + 2.0 * torch.randn_like(evidence_a)
    output_a = module(evidence_a, num_prompts=2)
    output_b = module(evidence_b, num_prompts=2)
    assert not torch.allclose(output_a, output_b)
    assert "seg_embeddings" not in module.forward.__code__.co_varnames


def test_decoupled_prompt_residual_is_bounded_and_backpropagates():
    module = DecoupledMaskPrompt(dim=8, hidden_dim=12, max_delta_ratio=0.15)
    torch.nn.init.normal_(module.evidence_out.weight, std=10.0)
    evidence = torch.randn(4, 2, 8, requires_grad=True)
    output = module(evidence, num_prompts=4)[:, 0]
    anchor = module.decoder_anchor.expand_as(output)
    ratio = (output - anchor).norm(dim=-1) / anchor.norm(dim=-1)
    assert torch.all(ratio <= 0.15001)
    output.square().mean().backward()
    assert evidence.grad is not None
    assert torch.isfinite(evidence.grad).all()
