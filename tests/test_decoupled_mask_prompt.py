import torch

from model.DIA_LISAt import DecoupledMaskPrompt, EvidencePresenceHead


def test_decoupled_prompt_zero_init_is_shared_anchor():
    module = DecoupledMaskPrompt(dim=8, hidden_dim=12, max_delta_ratio=0.2)
    evidence = torch.randn(3, 1, 8)
    output = module(evidence, num_prompts=3)
    expected = module.decoder_anchor.expand(3, -1)
    torch.testing.assert_close(output[:, 0], expected)


def test_anchor_initializes_once_from_teacher_mean():
    module = DecoupledMaskPrompt(dim=4, hidden_dim=8, max_delta_ratio=0.2)
    teacher = torch.tensor([[1.0, 2.0, 3.0, 4.0], [3.0, 4.0, 5.0, 6.0]])
    module.initialize_anchor_from_teacher(teacher.sum(0), torch.tensor(2.0))
    torch.testing.assert_close(module.decoder_anchor[0], teacher.mean(0))
    module.initialize_anchor_from_teacher(torch.zeros(4), torch.tensor(1.0))
    torch.testing.assert_close(module.decoder_anchor[0], teacher.mean(0))


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
    output = module(evidence, num_prompts=4)
    anchor = module.decoder_anchor.view(1, 1, 8).expand_as(output)
    ratio = (output - anchor).flatten(1).norm(dim=-1) / anchor.flatten(1).norm(dim=-1)
    assert torch.all(ratio <= 0.15001)
    output.square().mean().backward()
    assert evidence.grad is not None
    assert torch.isfinite(evidence.grad).all()


def test_decoupled_prompt_preserves_multiple_evidence_tokens():
    module = DecoupledMaskPrompt(dim=8, hidden_dim=12, max_delta_ratio=0.2)
    torch.nn.init.normal_(module.evidence_out.weight, std=0.1)
    evidence = torch.randn(2, 4, 8)
    output = module(evidence, num_prompts=2)
    assert output.shape == (2, 4, 8)
    assert not torch.allclose(output[:, 0], output[:, 1])


def test_presence_head_trains_on_positive_and_concept_only_negative():
    head = EvidencePresenceHead(dim=8, hidden_dim=12)
    concepts = torch.randn(3, 8, requires_grad=True)
    evidence = torch.randn(3, 4, 8, requires_grad=True)
    targets = torch.tensor([1.0, 0.0, 0.0])
    logits = head(concepts, evidence)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets)
    loss.backward()
    assert logits.shape == (3,)
    assert concepts.grad is not None and torch.isfinite(concepts.grad).all()
    assert evidence.grad is not None and torch.isfinite(evidence.grad).all()
