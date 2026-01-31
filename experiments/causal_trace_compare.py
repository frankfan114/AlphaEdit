#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np
import torch
from datasets import load_dataset
from matplotlib import pyplot as plt
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from rome.tok_dataset import (
    TokenizedDataset,
    dict_to_,
    flatten_masked_batch,
    length_collation,
)
from util import nethook
from util.runningstats import Covariance, tally


# -----------------------------
# Model wrapper
# -----------------------------
class ModelAndTokenizer:
    """
    Hold a GPT-style model and tokenizer. Count the number of layers.
    """

    def __init__(
        self,
        model_name_or_path=None,
        model=None,
        tokenizer=None,
        low_cpu_mem_usage=False,
        torch_dtype=None,
        device="cuda",
    ):
        self.device = device

        if tokenizer is None:
            assert model_name_or_path is not None
            tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

        if model is None:
            assert model_name_or_path is not None
            model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                low_cpu_mem_usage=low_cpu_mem_usage,
                torch_dtype=torch_dtype,
            )
            nethook.set_requires_grad(False, model)
            model.eval().to(device)

        self.tokenizer = tokenizer
        self.model = model

        self.layer_names = [
            n
            for n, _m in model.named_modules()
            if re.match(r"^(transformer|gpt_neox)\.(h|layers)\.\d+$", n)
        ]
        self.num_layers = len(self.layer_names)

    def __repr__(self):
        return (
            f"ModelAndTokenizer(model: {type(self.model).__name__} "
            f"[{self.num_layers} layers], "
            f"tokenizer: {type(self.tokenizer).__name__})"
        )


def layername(model, num, kind=None):
    if hasattr(model, "transformer"):
        if kind == "embed":
            return "transformer.wte"
        return f'transformer.h.{num}{"" if kind is None else "." + kind}'
    if hasattr(model, "gpt_neox"):
        if kind == "embed":
            return "gpt_neox.embed_in"
        if kind == "attn":
            kind = "attention"
        return f'gpt_neox.layers.{num}{"" if kind is None else "." + kind}'
    raise ValueError("unknown transformer structure")


# -----------------------------
# Token / input utilities
# -----------------------------
def make_inputs(tokenizer, prompts, device="cuda"):
    token_lists = [tokenizer.encode(p) for p in prompts]
    maxlen = max(len(t) for t in token_lists)

    if "[PAD]" in tokenizer.all_special_tokens:
        pad_id = tokenizer.all_special_ids[tokenizer.all_special_tokens.index("[PAD]")]
    else:
        pad_id = 0

    input_ids = [[pad_id] * (maxlen - len(t)) + t for t in token_lists]
    attention_mask = [[0] * (maxlen - len(t)) + [1] * len(t) for t in token_lists]

    return dict(
        input_ids=torch.tensor(input_ids).to(device),
        attention_mask=torch.tensor(attention_mask).to(device),
    )


def decode_tokens(tokenizer, token_array):
    if hasattr(token_array, "shape") and len(token_array.shape) > 1:
        return [decode_tokens(tokenizer, row) for row in token_array]
    return [tokenizer.decode([t]) for t in token_array]


def find_token_range_by_ids(tokenizer, token_array, subject: str):
    """
    Robust for GPT-2 byte-level BPE: match subject by *token id subsequence*
    instead of string-level decode substring search.

    token_array: 1D torch tensor or list of input_ids (shape [T])
    Return (tok_start, tok_end) for first match.
    """
    if isinstance(token_array, torch.Tensor):
        ids = token_array.tolist()
    else:
        ids = list(token_array)

    # Normalize a few common unicode punctuation forms
    subj = subject.strip()
    subj = subj.replace("’", "'").replace("“", '"').replace("”", '"')

    # Try several tokenization variants; GPT-2 often needs leading space
    candidates = []
    candidates.append(subj)
    if not subj.startswith(" "):
        candidates.append(" " + subj)
    candidates.append(subj.lstrip())
    if not subj.lstrip().startswith(" "):
        candidates.append(" " + subj.lstrip())

    # Deduplicate while preserving order
    seen = set()
    cand_list = []
    for c in candidates:
        if c not in seen and len(c) > 0:
            seen.add(c)
            cand_list.append(c)

    for cand in cand_list:
        subj_ids = tokenizer.encode(cand, add_special_tokens=False)
        if not subj_ids:
            continue
        L = len(subj_ids)
        for i in range(0, len(ids) - L + 1):
            if ids[i : i + L] == subj_ids:
                return (i, i + L)

    raise ValueError(f"subject token span not found by id-match: {subject!r}")


def get_next_token_probs(model, inp):
    out = model(**inp)["logits"]  # [B, T, V]
    probs = torch.softmax(out[:, -1, :], dim=1)  # [B, V]
    return probs


def get_expect_token_id(tokenizer, expect_str):
    """
    For next-token prediction, we need a single vocab id.
    If expect tokenizes into multiple tokens, take the *last* token id.
    """
    ids = tokenizer.encode(expect_str, add_special_tokens=False)
    if len(ids) == 0:
        return None
    return ids[-1]


# -----------------------------
# Causal tracing core
# -----------------------------
def trace_with_patch(
    model,
    inp,
    states_to_patch,
    answers_t,
    tokens_to_mix,
    noise=0.1,
    uniform_noise=False,
    replace=False,
    trace_layers=None,
):
    rs = np.random.RandomState(1)
    if uniform_noise:
        prng = lambda *shape: rs.uniform(-1, 1, shape)
    else:
        prng = lambda *shape: rs.randn(*shape)

    patch_spec = defaultdict(list)
    for t, l in states_to_patch:
        patch_spec[l].append(t)

    embed_layername = layername(model, 0, "embed")

    def untuple(x):
        return x[0] if isinstance(x, tuple) else x

    if isinstance(noise, float):
        noise_fn = lambda x: noise * x
    else:
        noise_fn = noise

    def patch_rep(x, layer):
        if layer == embed_layername:
            if tokens_to_mix is not None:
                b, e = tokens_to_mix
                noise_data = noise_fn(
                    torch.from_numpy(prng(x.shape[0] - 1, e - b, x.shape[2]))
                ).to(x.device)
                if replace:
                    x[1:, b:e] = noise_data
                else:
                    x[1:, b:e] += noise_data
            return x

        if layer not in patch_spec:
            return x

        h = untuple(x)
        for t in patch_spec[layer]:
            h[1:, t] = h[0, t]
        return x

    additional_layers = [] if trace_layers is None else trace_layers
    with torch.no_grad(), nethook.TraceDict(
        model,
        [embed_layername] + list(patch_spec.keys()) + additional_layers,
        edit_output=patch_rep,
    ):
        outputs_exp = model(**inp)

    probs = torch.softmax(outputs_exp.logits[1:, -1, :], dim=1).mean(dim=0)[answers_t]

    if trace_layers is not None:
        raise NotImplementedError("trace_layers not used in this script version")

    return probs


def trace_important_states(
    model,
    num_layers,
    inp,
    e_range,
    answers_t,
    noise=0.1,
    uniform_noise=False,
    replace=False,
    token_range=None,
):
    ntoks = inp["input_ids"].shape[1]
    table = []

    if token_range is None:
        token_range = range(ntoks)

    for tnum in token_range:
        row = []
        for layer in range(num_layers):
            r = trace_with_patch(
                model,
                inp,
                [(tnum, layername(model, layer))],
                answers_t,
                tokens_to_mix=e_range,
                noise=noise,
                uniform_noise=uniform_noise,
                replace=replace,
            )
            row.append(r)
        table.append(torch.stack(row))
    return torch.stack(table)


def trace_important_window(
    model,
    num_layers,
    inp,
    e_range,
    answers_t,
    kind,
    window=10,
    noise=0.1,
    uniform_noise=False,
    replace=False,
    token_range=None,
):
    ntoks = inp["input_ids"].shape[1]
    table = []

    if token_range is None:
        token_range = range(ntoks)

    for tnum in token_range:
        row = []
        for layer in range(num_layers):
            layerlist = [
                (tnum, layername(model, L, kind))
                for L in range(
                    max(0, layer - window // 2),
                    min(num_layers, layer - (-window // 2)),
                )
            ]
            r = trace_with_patch(
                model,
                inp,
                layerlist,
                answers_t,
                tokens_to_mix=e_range,
                noise=noise,
                uniform_noise=uniform_noise,
                replace=replace,
            )
            row.append(r)
        table.append(torch.stack(row))
    return torch.stack(table)


def calculate_hidden_flow(
    mt: ModelAndTokenizer,
    prompt: str,
    subject: str,
    expect_str: str,
    samples=10,
    noise=0.1,
    token_range=None,
    uniform_noise=False,
    replace=False,
    window=10,
    kind=None,
):
    """
    Causal trace wrt a *specified* expect token (target_true in your use case).
    Does NOT require top-1 == expect.
    """
    inp = make_inputs(mt.tokenizer, [prompt] * (samples + 1), device=mt.device)

    # subject token span: robust id-subsequence match
    try:
        e_range = find_token_range_by_ids(mt.tokenizer, inp["input_ids"][0], subject)
    except ValueError as e:
        return dict(
            ok=False,
            reason=str(e),
            prompt=prompt,
            subject=subject,
        )

    # choose which token we trace (expect token)
    expect_t = get_expect_token_id(mt.tokenizer, expect_str)
    if expect_t is None:
        return dict(ok=False, reason="empty expect tokenization", prompt=prompt, subject=subject)

    # base prediction stats (uncorrupted run at batch[0])
    with torch.no_grad():
        probs0 = get_next_token_probs(mt.model, inp)[0]  # [V]
        pred_t = torch.argmax(probs0).item()
        pred_str = mt.tokenizer.decode([pred_t])
        p_expect = probs0[expect_t].item()

    # corrupt-without-patch baseline (mean prob over corrupted runs)
    low_score = trace_with_patch(
        mt.model,
        inp,
        states_to_patch=[],
        answers_t=expect_t,
        tokens_to_mix=e_range,
        noise=noise,
        uniform_noise=uniform_noise,
        replace=replace,
    ).item()

    # full scan
    if not kind:
        differences = trace_important_states(
            mt.model,
            mt.num_layers,
            inp,
            e_range,
            expect_t,
            noise=noise,
            uniform_noise=uniform_noise,
            replace=replace,
            token_range=token_range,
        )
    else:
        differences = trace_important_window(
            mt.model,
            mt.num_layers,
            inp,
            e_range,
            expect_t,
            noise=noise,
            uniform_noise=uniform_noise,
            replace=replace,
            window=window,
            kind=kind,
            token_range=token_range,
        )

    differences = differences.detach().cpu()
    input_tokens = decode_tokens(mt.tokenizer, inp["input_ids"][0])
    return dict(
        ok=True,
        scores=differences,
        low_score=low_score,
        high_score=p_expect,
        p_expect=p_expect,
        pred=pred_str,
        pred_t=pred_t,
        expect=expect_str,
        expect_t=expect_t,
        input_ids=inp["input_ids"][0],
        input_tokens=input_tokens,
        subject_range=e_range,
        window=window,
        kind=kind or "",
    )


# -----------------------------
# Plot
# -----------------------------
def plot_trace_heatmap(result, savepdf=None, title=None, xlabel=None, modelname=None):
    differences = result["scores"]
    low_score = result["low_score"]
    kind = None if (not result["kind"] or result["kind"] == "None") else str(result["kind"])
    window = result.get("window", 10)

    labels = list(result["input_tokens"])
    for i in range(*result["subject_range"]):
        labels[i] = labels[i] + "*"

    with plt.rc_context(rc={"font.family": "Times New Roman"}):
        fig, ax = plt.subplots(figsize=(3.8, 2.2), dpi=220)
        h = ax.pcolor(
            differences,
            cmap={None: "Purples", "None": "Purples", "mlp": "Greens", "attn": "Reds"}[kind],
            vmin=low_score,
        )
        ax.invert_yaxis()
        ax.set_yticks([0.5 + i for i in range(len(differences))])
        ax.set_xticks([0.5 + i for i in range(0, differences.shape[1] - 1, 5)])
        ax.set_xticklabels(list(range(0, differences.shape[1] - 1, 5)))
        ax.set_yticklabels(labels)

        if not modelname:
            modelname = "GPT"
        if not kind:
            ax.set_title("Restore one layer after corrupting subject span")
            ax.set_xlabel(f"restored layer in {modelname}")
        else:
            kindname = "MLP" if kind == "mlp" else "Attn"
            ax.set_title(f"Restore {kindname} window after corrupting subject span")
            ax.set_xlabel(f"center of {window}-layer restored {kindname} window")

        cb = plt.colorbar(h)
        if title is not None:
            ax.set_title(title)
        if xlabel is not None:
            ax.set_xlabel(xlabel)

        cb.ax.set_title(f"p(expect='{str(result['expect']).strip()}')", y=-0.18, fontsize=9)

        if savepdf:
            os.makedirs(os.path.dirname(savepdf), exist_ok=True)
            plt.savefig(savepdf, bbox_inches="tight")
            plt.close()
        else:
            plt.show()


def heatmap_argmax(scores_tensor):
    """
    scores_tensor: [ntoks, nlayers]
    return (tok_idx, layer_idx, max_value)
    """
    x = scores_tensor
    flat = x.view(-1)
    k = int(torch.argmax(flat).item())
    ntoks, nlayers = x.shape[0], x.shape[1]
    tok = k // nlayers
    lay = k % nlayers
    return tok, lay, float(x[tok, lay].item())


# -----------------------------
# Noise collectors (unchanged)
# -----------------------------
def collect_embedding_std(mt, subjects):
    alldata = []
    for s in subjects:
        inp = make_inputs(mt.tokenizer, [s], device=mt.device)
        with nethook.Trace(mt.model, layername(mt.model, 0, "embed")) as t:
            mt.model(**inp)
            alldata.append(t.output[0])
    alldata = torch.cat(alldata)
    return alldata.std().item()


def get_embedding_cov(mt):
    model = mt.model
    tokenizer = mt.tokenizer

    def get_ds():
        ds_name = "wikitext"
        raw_ds = load_dataset(
            ds_name,
            dict(wikitext="wikitext-103-raw-v1", wikipedia="20220301.en")[ds_name],
            trust_remote_code=True,
        )
        try:
            maxlen = model.config.n_positions
        except Exception:
            maxlen = 100
        return TokenizedDataset(raw_ds["train"], tokenizer, maxlen=maxlen)

    ds = get_ds()
    sample_size = 1000
    batch_size = 5
    filename = None
    batch_tokens = 100

    stat = Covariance()
    loader = tally(
        stat,
        ds,
        cache=filename,
        sample_size=sample_size,
        batch_size=batch_size,
        collate_fn=length_collation(batch_tokens),
        pin_memory=True,
        random_sample=1,
        num_workers=0,
    )
    with torch.no_grad():
        for batch_group in loader:
            for batch in batch_group:
                batch = dict_to_(batch, mt.device)
                if "position_ids" in batch:
                    del batch["position_ids"]
                with nethook.Trace(model, layername(mt.model, 0, "embed")) as tr:
                    model(**batch)
                feats = flatten_masked_batch(tr.output, batch["attention_mask"])
                stat.add(feats.cpu().double())
    return stat.mean(), stat.covariance()


def make_generator_transform(mean=None, cov=None):
    d = len(mean) if mean is not None else len(cov)
    device = mean.device if mean is not None else cov.device
    layer = torch.nn.Linear(d, d, dtype=torch.double)
    nethook.set_requires_grad(False, layer)
    layer.to(device)
    layer.bias[...] = 0 if mean is None else mean
    if cov is None:
        layer.weight[...] = torch.eye(d).to(device)
    else:
        _, s, v = cov.svd()
        w = s.sqrt()[None, :] * v
        layer.weight[...] = w
    return layer


def collect_embedding_gaussian(mt):
    m, c = get_embedding_cov(mt)
    return make_generator_transform(m, c)


def collect_embedding_tdist(mt, degree=3):
    u_sample = torch.from_numpy(np.random.RandomState(2).chisquare(df=degree, size=1000))
    fixed_sample = ((degree - 2) / u_sample).sqrt()
    mvg = collect_embedding_gaussian(mt)

    def normal_to_student(x):
        gauss = mvg(x)
        size = gauss.shape[:-1].numel()
        factor = fixed_sample[:size].reshape(gauss.shape[:-1] + (1,))
        return factor * gauss

    return normal_to_student


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Causal Tracing: Base vs Edited on neighborhood probes")

    def aa(*args, **kwargs):
        parser.add_argument(*args, **kwargs)

    def parse_noise_rule(code):
        if code in ["m", "s"]:
            return code
        elif re.match(r"^[uts][\d\.]+", code):
            return code
        else:
            return float(code)

    aa("--base_model_name", default="gpt2-xl")
    aa("--edited_model_name", required=True)

    aa("--fact_file", required=True, help="counterfact_neighborhood.json (your generated probes)")
    aa("--expect_field", default="target_true", choices=["target_true", "target_new"])

    aa("--output_dir", default="results/{tag}/causal_trace_compare")
    aa("--noise_level", default="s3", type=parse_noise_rule)
    aa("--replace", default=0, type=int)

    aa("--max_cases", default=200, type=int, help="only plot/save cases with known_id <= N")
    aa("--no_tqdm", action="store_true", help="disable tqdm progress bar (good for PBS logs)")

    args = parser.parse_args()

    # tags for output paths
    def model_tag(name_or_path):
        if os.path.isdir(name_or_path):
            return os.path.basename(os.path.abspath(name_or_path))
        return name_or_path.replace("/", "_")

    base_tag = model_tag(args.base_model_name)
    edited_tag = model_tag(args.edited_model_name)
    run_tag = f"base_{base_tag}__edited_{edited_tag}__expect_{args.expect_field}"

    output_dir = args.output_dir.format(tag=run_tag)
    result_dir = f"{output_dir}/cases"
    pdf_dir = f"{output_dir}/pdfs"
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(pdf_dir, exist_ok=True)

    # dtype heuristic
    torch_dtype_base = torch.float16 if "20b" in args.base_model_name else None
    torch_dtype_edited = torch.float16 if "20b" in args.edited_model_name else None

    base_mt = ModelAndTokenizer(args.base_model_name, torch_dtype=torch_dtype_base)
    edited_mt = ModelAndTokenizer(args.edited_model_name, torch_dtype=torch_dtype_edited)

    print("[INFO] Base :", base_mt, flush=True)
    print("[INFO] Edited:", edited_mt, flush=True)
    if base_mt.num_layers != edited_mt.num_layers:
        print(f"[WARN] num_layers differ: base={base_mt.num_layers}, edited={edited_mt.num_layers}", flush=True)

    # load neighborhood probes
    with open(args.fact_file, "r", encoding="utf-8") as f:
        raw_cases = json.load(f)

    # build knowns
    knowns = []
    for case in raw_cases:
        subject = case["subject"]
        prompt = case["prompt"].format(subject)

        expect_obj = case[args.expect_field]  # {"str":..., "id":...}
        expect_str = expect_obj["str"] if isinstance(expect_obj, dict) else str(expect_obj)

        knowns.append(
            dict(
                known_id=case["known_id"],
                case_id=case.get("case_id", None),
                neighbor_index=case.get("neighbor_index", None),
                subject=subject,
                prompt=prompt,
                expect=expect_str,
                relation_id=case.get("relation_id", None),
                source=case.get("source", "neighborhood"),
            )
        )

    # noise config
    noise_level = args.noise_level
    uniform_noise = False
    if isinstance(noise_level, str):
        if noise_level.startswith("s"):
            factor = float(noise_level[1:]) if len(noise_level) > 1 else 1.0
            noise_level = factor * collect_embedding_std(
                base_mt, [k["subject"] for k in knowns[: min(len(knowns), 256)]]
            )
            print(f"[INFO] Using noise_level={noise_level} (spherical std * {factor})", flush=True)
        elif noise_level == "m":
            noise_level = collect_embedding_gaussian(base_mt)
            print("[INFO] Using multivariate gaussian noise", flush=True)
        elif noise_level.startswith("t"):
            degrees = float(noise_level[1:])
            noise_level = collect_embedding_tdist(base_mt, degrees)
            print(f"[INFO] Using t-dist noise, df={degrees}", flush=True)
        elif noise_level.startswith("u"):
            uniform_noise = True
            noise_level = float(noise_level[1:])
            print(f"[INFO] Using uniform noise level={noise_level}", flush=True)

    # tqdm settings: PBS logs often not a tty
    disable_bar = args.no_tqdm or (not sys.stdout.isatty())

    # loop
    for knowledge in tqdm(knowns, desc="cases", disable=disable_bar):
        known_id = knowledge["known_id"]
        if known_id > args.max_cases:
            continue

        for model_label, mt, mt_tag in [
            ("base", base_mt, base_tag),
            ("edited", edited_mt, edited_tag),
        ]:
            for kind in (None, "mlp", "attn"):
                kind_suffix = f"_{kind}" if kind else ""
                npz_path = f"{result_dir}/{model_label}_knowledge_{known_id}{kind_suffix}.npz"

                if not os.path.isfile(npz_path):
                    result = calculate_hidden_flow(
                        mt,
                        knowledge["prompt"],
                        knowledge["subject"],
                        expect_str=knowledge["expect"],
                        kind=kind,
                        noise=noise_level,
                        uniform_noise=uniform_noise,
                        replace=bool(args.replace),
                    )
                    if not result.get("ok", False):
                        tqdm.write(
                            f"[SKIP] {model_label} known_id={known_id} kind={kind or 'None'} "
                            f"reason={result.get('reason')}",
                            file=sys.stdout,
                        )
                        sys.stdout.flush()
                        continue

                    numpy_result = {
                        k: v.detach().cpu().numpy() if torch.is_tensor(v) else v
                        for k, v in result.items()
                    }
                    np.savez(npz_path, **numpy_result)
                else:
                    numpy_result = dict(np.load(npz_path, allow_pickle=True))

                # print quick summary (position + probs)
                scores_t = torch.tensor(numpy_result["scores"])
                tok_i, lay_i, mx = heatmap_argmax(scores_t)
                subj_r = tuple(numpy_result["subject_range"])
                pred = str(numpy_result["pred"])
                pexp = float(numpy_result["p_expect"])
                tqdm.write(
                    f"[{model_label}] id={known_id} kind={kind or 'None'} "
                    f"pred={pred.strip()} p(expect)={pexp:.4g} "
                    f"argmax(tok={tok_i}, layer={lay_i}, val={mx:.4g}) "
                    f"subject_range={subj_r}",
                    file=sys.stdout,
                )
                sys.stdout.flush()

                # plot
                plot_result = dict(numpy_result)
                plot_result["kind"] = kind

                expect_safe = re.sub(r"[^A-Za-z0-9._\- ]+", "_", str(plot_result["expect"]).strip())
                expect_safe = expect_safe.strip().replace(" ", "_")
                pdfname = f"{pdf_dir}/{model_label}_{mt_tag}_expect_{expect_safe}_known_{known_id}{kind_suffix}.pdf"

                plot_trace_heatmap(
                    plot_result,
                    savepdf=pdfname,
                    modelname=f"{model_label}:{mt_tag}",
                )

    print(f"[OK] outputs saved under: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
