#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compare `prompt` submodule weights between two SSTrack checkpoints."""

import argparse
import os
import sys
from typing import Dict

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch


def _strip_prefix(key, prefixes):
    for p in prefixes:
        if key.startswith(p):
            return key[len(p) :]
    return key


def load_net_state(path):
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location="cpu")
    if isinstance(ckpt, dict) and "net" in ckpt:
        return ckpt["net"]
    if isinstance(ckpt, dict) and all(isinstance(v, torch.Tensor) for v in ckpt.values()):
        return ckpt
    keys = list(ckpt.keys())[:10]
    raise KeyError("Unexpected checkpoint format in {}: keys={}".format(path, keys))


def normalize_keys(sd):
    # type: (Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]
    """Drop common DDP / wrapper prefixes for key alignment."""
    out = {}
    prefixes = ("module.",)
    for k, v in sd.items():
        nk = _strip_prefix(k, prefixes)
        if nk in out and nk != k:
            continue
        out[nk] = v
    return out


def prompt_keys(sd):
    # type: (Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]
    return {k: v for k, v in sd.items() if ".prompt." in k or k.startswith("prompt.")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--a",
        # default="/data4/litianhao/must2/checkpoints/train/sstrack/baseline_must_trans_enc_cope/SSTrack_ep0005.pth.tar",
        default="/data4/litianhao/must2/checkpoints/train/sstrack/baseline_must/SSTrack_ep0005.pth.tar",
        help="checkpoint path (e.g. ep5)",
    )
    ap.add_argument(
        "--b",
        # default="/data4/litianhao/must2/checkpoints/train/sstrack/baseline_must_trans_enc_cope/SSTrack_ep0050.pth.tar",
        default="/data4/litianhao/must2/checkpoints/train/sstrack/baseline_must/SSTrack_ep0050.pth.tar",
        help="checkpoint path (e.g. ep50)",
    )
    ap.add_argument("--rtol", type=float, default=1e-5)
    ap.add_argument("--atol", type=float, default=1e-6)
    args = ap.parse_args()

    sd_a = prompt_keys(normalize_keys(load_net_state(args.a)))
    sd_b = prompt_keys(normalize_keys(load_net_state(args.b)))

    keys_a = set(sd_a.keys())
    keys_b = set(sd_b.keys())
    only_a = sorted(keys_a - keys_b)
    only_b = sorted(keys_b - keys_a)
    common = sorted(keys_a & keys_b)

    print("checkpoint A:", args.a)
    print("checkpoint B:", args.b)
    print("prompt tensors in A:", len(sd_a), "| in B:", len(sd_b), "| common:", len(common))
    if only_a:
        print("only in A:", only_a)
    if only_b:
        print("only in B:", only_b)
    if not common:
        sys.stderr.write("No common prompt keys; nothing to compare.\n")
        sys.exit(1)

    all_close = True
    max_overall = 0.0
    mean_overall = 0.0
    n_elems = 0

    for k in common:
        ta, tb = sd_a[k], sd_b[k]
        if ta.shape != tb.shape:
            print(
                "[shape mismatch] {}  A{}  B{}".format(k, tuple(ta.shape), tuple(tb.shape))
            )
            all_close = False
            continue
        if ta.dtype != tb.dtype:
            tb = tb.to(ta.dtype)
        diff = (ta.float() - tb.float()).abs()
        mx = float(diff.max().item())
        mn = float(diff.mean().item())
        ne = diff.numel()
        max_overall = max(max_overall, mx)
        mean_overall += mn * ne
        n_elems += ne
        close = torch.allclose(ta.cpu(), tb.cpu(), rtol=args.rtol, atol=args.atol)
        all_close = all_close and close
        print(
            "{}\n  max|diff|={:.6e}  mean|diff|={:.6e}  allclose(rtol={}, atol={})={}".format(
                k, mx, mn, args.rtol, args.atol, close
            )
        )

    mean_overall /= max(n_elems, 1)
    print("---")
    print("overall max|diff| (over tensors): {:.6e}".format(max_overall))
    print("overall weighted mean|diff|:     {:.6e}".format(mean_overall))
    print("all tensors allclose: {}".format(all_close))
    sys.exit(0 if all_close else 2)


if __name__ == "__main__":
    main()
