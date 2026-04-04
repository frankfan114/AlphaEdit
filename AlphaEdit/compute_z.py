from typing import Dict, List, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from rome import repr_tools
from util import nethook

from .AlphaEdit_hparams import AlphaEditHyperParams


def compute_z(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    request: Dict,
    hparams: AlphaEditHyperParams,
    layer: int,
    context_templates: List[str],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the value (right) vector for the rank-1 update.
    Runs a simple optimization procedure.
    """

    # Get model parameters
    lm_w, ln_f = (
        nethook.get_module(model, f"{hparams.lm_head_module}").weight.T,
        nethook.get_module(model, hparams.ln_f_module),
    )
    try:
        lm_b = nethook.get_parameter(model, f"{hparams.lm_head_module}.bias")
    except LookupError as _:
        lm_b = next(model.parameters()).new_zeros(model.config.vocab_size)

    print("Computing right vector (v)")

    def tokenize_target(target_text: str) -> torch.Tensor:
        token_ids = tok(target_text, return_tensors="pt").to("cuda")["input_ids"][0]
        if token_ids[0] == tok.bos_token_id or token_ids[0] == tok.unk_token_id:
            token_ids = token_ids[1:]
        return token_ids

    def build_inputs(target_ids: torch.Tensor):
        prompt_texts = [
            context.format(request["prompt"]) + tok.decode(target_ids[:-1])
            for context_types in context_templates
            for context in context_types
        ]
        tokenized = tok(
            [prompt.format(request["subject"]) for prompt in prompt_texts],
            return_tensors="pt",
            padding=True,
        ).to("cuda")
        targets = torch.tensor(-100, device="cuda").repeat(
            len(prompt_texts), *tokenized["input_ids"].shape[1:]
        )
        for idx in range(len(prompt_texts)):
            ex_len = tokenized["attention_mask"][idx].sum()
            targets[idx, ex_len - len(target_ids) : ex_len] = target_ids
        lookup_indices = [
            find_fact_lookup_idx(
                prompt_text, request["subject"], tok, hparams.fact_token, verbose=False
            )
            for prompt_text in prompt_texts
        ]
        return prompt_texts, tokenized, targets, lookup_indices

    # Tokenize target into list of int token IDs
    target_ids = tokenize_target(request["target_new"]["str"])

    # Compile list of rewriting and KL x/y pairs
    rewriting_prompts, input_tok, rewriting_targets, lookup_idxs = build_inputs(target_ids)
    kl_prompts = ["{} is a"]
    all_prompts = rewriting_prompts + kl_prompts
    kl_input_tok = tok(
        [prompt.format(request["subject"]) for prompt in kl_prompts],
        return_tensors="pt",
        padding=True,
    ).to("cuda")
    kl_lookup_idxs = [
        find_fact_lookup_idx(
            prompt, request["subject"], tok, hparams.fact_token, verbose=(idx == 0)
        )
        for idx, prompt in enumerate(kl_prompts)
    ]

    old_target_ids = None
    old_input_tok = None
    old_rewriting_targets = None
    old_lookup_idxs = None
    if (
        hparams.knowledge_conflict != "none"
        and "target_true" in request
        and isinstance(request["target_true"], dict)
        and request["target_true"].get("str") is not None
    ):
        old_target_ids = tokenize_target(request["target_true"]["str"])
        (
            _,
            old_input_tok,
            old_rewriting_targets,
            old_lookup_idxs,
        ) = build_inputs(old_target_ids)

    # Finalize rewrite and loss layers
    loss_layer = max(hparams.v_loss_layer, layer)
    print(f"Rewrite layer is {layer}")
    print(f"Tying optimization objective to {loss_layer}")

    # Set up an optimization over a latent vector that, when output at the
    # rewrite layer, i.e. hypothesized fact lookup location, will induce the
    # target token to be predicted at the final layer.
    if hasattr(model.config, 'n_embd'):
        delta = torch.zeros((model.config.n_embd,), requires_grad=True, device="cuda")
    elif hasattr(model.config, 'hidden_size'):
        delta = torch.zeros((model.config.hidden_size,), requires_grad=True, device="cuda")
    else:
        raise NotImplementedError
    target_init, kl_distr_init = None, None
    active_lookup_idxs = None
    record_target_init = False

    # Inserts new "delta" variable at the appropriate part of the computation
    def edit_output_fn(cur_out, cur_layer):
        nonlocal target_init

        if cur_layer == hparams.layer_module_tmp.format(layer):
            # Store initial value of the vector of interest
            if target_init is None and record_target_init:
                print("Recording initial value of v*")
                # Initial value is recorded for the clean sentence
                target_init = cur_out[0][0, active_lookup_idxs[0]].detach().clone()

            # Add intervened delta
            for i, idx in enumerate(active_lookup_idxs):

                if len(active_lookup_idxs) != len(cur_out[0]):
                    cur_out[0][idx, i, :] += delta
                else:
                    cur_out[0][i, idx, :] += delta

        return cur_out

    def forward_with_delta(input_batch, lookup_indices, retain_kl=False):
        nonlocal kl_distr_init, active_lookup_idxs, record_target_init
        active_lookup_idxs = lookup_indices
        record_target_init = target_init is None
        traced_layers = [
            hparams.layer_module_tmp.format(loss_layer),
            hparams.layer_module_tmp.format(layer),
        ]
        with nethook.TraceDict(
            module=model,
            layers=traced_layers,
            retain_input=False,
            retain_output=True,
            edit_output=edit_output_fn,
        ) as tr:
            logits = model(**input_batch).logits
            if retain_kl:
                kl_logits = torch.stack(
                    [
                        logits[i, idx, :]
                        for i, idx in enumerate(lookup_indices)
                    ],
                    dim=0,
                )
                kl_log_probs = torch.nn.functional.log_softmax(kl_logits, dim=1)
                if kl_distr_init is None:
                    kl_distr_init = kl_log_probs.detach().clone()
            else:
                kl_log_probs = None

        output = tr[hparams.layer_module_tmp.format(loss_layer)].output[0]
        return output, kl_log_probs

    def token_nll_from_output(output, targets, target_token_count):
        if output.shape[1] != targets.shape[1]:
            output = torch.transpose(output, 0, 1)
        full_repr = output[: targets.shape[0]]
        log_probs = torch.log_softmax(
            ln_f(full_repr) @ lm_w.to(full_repr.device) + lm_b.to(full_repr.device), dim=2
        )
        gathered = torch.gather(
            log_probs,
            2,
            torch.where(targets != -100, targets, 0).unsqueeze(2).to(log_probs.device),
        ).squeeze(2)
        mask = (targets != -100).float()
        return -(gathered * mask.to(gathered.device)).sum(1) / target_token_count

    # Optimizer
    opt = torch.optim.Adam([delta], lr=hparams.v_lr)
    nethook.set_requires_grad(False, model)

    # Execute optimization
    for it in range(hparams.v_num_grad_steps):
        opt.zero_grad()

        # Forward propagation
        output, _ = forward_with_delta(input_tok, lookup_idxs, retain_kl=False)
        _, kl_log_probs = forward_with_delta(kl_input_tok, kl_lookup_idxs, retain_kl=True)

        # Compute loss on rewriting targets
        nll_loss_each = token_nll_from_output(output, rewriting_targets, target_ids.size(0))
        nll_loss = nll_loss_each.mean()
        kl_loss = hparams.kl_factor * torch.nn.functional.kl_div(
            kl_distr_init, kl_log_probs, log_target=True, reduction="batchmean"
        )
        weight_decay = hparams.v_weight_decay * (
            torch.norm(delta) / torch.norm(target_init) ** 2
        )
        conflict_loss = torch.tensor(0.0, device=nll_loss.device)
        if (
            hparams.knowledge_conflict != "none"
            and old_input_tok is not None
            and old_rewriting_targets is not None
            and old_target_ids is not None
            and old_target_ids.numel() > 0
            and hparams.conflict_loss_weight > 0
        ):
            old_output, _ = forward_with_delta(old_input_tok, old_lookup_idxs, retain_kl=False)
            old_nll_loss_each = token_nll_from_output(
                old_output, old_rewriting_targets, old_target_ids.size(0)
            )
            conflict_loss = hparams.conflict_loss_weight * torch.relu(
                nll_loss_each - old_nll_loss_each + hparams.conflict_margin
            ).mean()

        loss = (
            nll_loss
            + kl_loss.to(nll_loss.device)
            + weight_decay.to(nll_loss.device)
            + conflict_loss.to(nll_loss.device)
        )
        print(
            f"loss {np.round(loss.item(), 3)} = {np.round(nll_loss.item(), 3)} + {np.round(kl_loss.item(), 3)} + {np.round(weight_decay.item(), 3)} + {np.round(conflict_loss.item(), 3)} "
            f"avg prob of [{request['target_new']['str']}] "
            f"{torch.exp(-nll_loss_each).mean().item()}"
        )
        if loss < 5e-2:
            break

        if it == hparams.v_num_grad_steps - 1:
            break

        # Backpropagate
        loss.backward()
        opt.step()

        # Project within L2 ball
        max_norm = hparams.clamp_norm_factor * target_init.norm()
        if delta.norm() > max_norm:
            with torch.no_grad():
                delta[...] = delta * max_norm / delta.norm()

    target = target_init + delta
    print(
        f"Init norm {target_init.norm()} | Delta norm {delta.norm()} | Target norm {target.norm()}"
    )

    return target


def get_module_input_output_at_words(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    layer: int,
    context_templates: List[str],
    words: List[str],
    module_template: str,
    fact_token_strategy: str,
) -> Tuple[torch.Tensor]:
    """
    Retrieves detached representations for a word at the input and
    output of a particular layer module.
    """

    word_repr_args = dict(
        model=model,
        tok=tok,
        layer=layer,
        module_template=module_template,
    )
    if "subject_" in fact_token_strategy and fact_token_strategy.index("subject_") == 0:
        context_info = dict(
            context_templates=context_templates,
            words=words,
        )
        subtoken = fact_token_strategy[len("subject_") :]
        l_input, l_output = repr_tools.get_reprs_at_word_tokens(
            track="both", subtoken=subtoken, **context_info, **word_repr_args
        )
    elif fact_token_strategy == "last":
        raise Exception("This is definitely bugged, fix it.")
        context_info = dict(
            contexts=[
                tmp[i].format(words[i]) for i, tmp in enumerate(context_templates)
            ],
            idxs=[000000],
        )
        l_input, l_output = repr_tools.get_reprs_at_idxs(
            track="both", **context_info, **word_repr_args
        )
    else:
        raise ValueError(f"fact_token={fact_token_strategy} not recognized")

    return l_input.detach(), l_output.detach()


def find_fact_lookup_idx(
    prompt: str,
    subject: str,
    tok: AutoTokenizer,
    fact_token_strategy: str,
    verbose=True,
) -> int:
    """
    Computes hypothesized fact lookup index given a sentence and subject.
    """

    ret = None
    if fact_token_strategy == "last":
        ret = -1
    elif (
        "subject_" in fact_token_strategy and fact_token_strategy.index("subject_") == 0
    ):
        ret = repr_tools.get_words_idxs_in_templates(
            tok=tok,
            context_templates=[prompt],
            words=[subject],
            subtoken=fact_token_strategy[len("subject_") :],
        )[0][0]
    else:
        raise ValueError(f"fact_token={fact_token_strategy} not recognized")

    sentence = prompt.format(subject)
    if verbose:
        print(
            f"Lookup index found: {ret} | Sentence: {sentence} | Token:",
            tok.decode(tok(sentence)["input_ids"][ret]),
        )

    return ret
