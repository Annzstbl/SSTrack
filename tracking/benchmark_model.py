#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Benchmark SSTrack: parameter count, FLOPs (GMAC), and inference FPS.

Usage (from repo root or tracking/)::

  python tracking/benchmark_model.py --config baseline_hsitrack_trans_enc3
  python tracking/benchmark_model.py --config baseline_hsitrack_trans_enc3 --device cuda --fps_iters 200

Optional dependency for FLOPs: ``pip install thop``.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import torch
import torch.nn as nn

# Repo root on path
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_THIS_DIR, '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from lib.config.sstrack.config import cfg, update_config_from_file
from lib.models.sstrack import build_sstrack


def _load_cfg(config_name: str) -> None:
    yaml_path = os.path.join(_ROOT, 'experiments', 'sstrack', f'{config_name}.yaml')
    if not os.path.isfile(yaml_path):
        raise FileNotFoundError(f'Config not found: {yaml_path}')
    update_config_from_file(yaml_path)


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """Returns (total_params, trainable_params)."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


class SSTrackProfileWrapper(nn.Module):
    """Wrap SSTrack.forward (list inputs) as tensor-only forward for thop."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, *tensors: torch.Tensor) -> torch.Tensor:
        nt = self._num_templates
        ns = self._num_searches
        assert len(tensors) == nt + ns, (len(tensors), nt, ns)
        template = list(tensors[:nt])
        search = list(tensors[nt:])
        self.model.track_query = None
        out = self.model(template=template, search=search, ce_template_mask=None)
        return out[-1]['score_map']


def build_dummy_inputs(cfg) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Match TEST template/search sizes and DATA.SEARCH.LENGTH / TEST.TEMPLATE_NUMBER."""
    c_in = 8  # vit_base_patch16_224_ce uses 8 bands (HSI)
    nt = int(cfg.TEST.TEMPLATE_NUMBER)
    ns = int(cfg.DATA.SEARCH.LENGTH)
    th = tw = int(cfg.TEST.TEMPLATE_SIZE)
    sh = sw = int(cfg.TEST.SEARCH_SIZE)
    device = torch.device('cpu')
    template = [torch.randn(1, c_in, th, tw, device=device) for _ in range(nt)]
    search = [torch.randn(1, c_in, sh, sw, device=device) for _ in range(ns)]
    return template, search


def measure_flops_thop(model: nn.Module, cfg, device: torch.device) -> tuple[float | None, float | None]:
    """Returns (gmac, thop_reported_params) or (None, None) if thop unavailable or failed."""
    try:
        from thop import clever_format, profile
    except ImportError:
        return None, None

    template, search = build_dummy_inputs(cfg)
    nt, ns = len(template), len(search)
    tensors = [t.to(device) for t in template + search]

    wrapped = SSTrackProfileWrapper(model).to(device)
    wrapped._num_templates = nt
    wrapped._num_searches = ns

    model.eval()
    wrapped.eval()
    print('[FLOPs] Running thop.profile (can take tens of seconds on large ViT)...', flush=True)
    try:
        macs, params_thop = profile(wrapped, inputs=tuple(tensors), verbose=False)
    except Exception as e:
        print(f'[FLOPs] thop.profile failed: {e}')
        return None, None

    gmac = float(macs) / 1e9
    return gmac, float(params_thop)


def measure_fps(
    model: nn.Module,
    cfg,
    device: torch.device,
    warmup: int,
    iters: int,
) -> float:
    """Average FPS of one tracking forward (batch size 1)."""
    template, search = build_dummy_inputs(cfg)
    template = [t.to(device) for t in template]
    search = [t.to(device) for t in search]

    model.eval()
    model.to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            model.track_query = None
            _ = model(template=template, search=search, ce_template_mask=None)
    if device.type == 'cuda':
        torch.cuda.synchronize()

    with torch.no_grad():
        t0 = time.perf_counter()
        for _ in range(iters):
            model.track_query = None
            _ = model(template=template, search=search, ce_template_mask=None)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t1 = time.perf_counter()

    elapsed = t1 - t0
    fps = iters / elapsed
    return fps


def main():
    parser = argparse.ArgumentParser(description='SSTrack: Params / FLOPs / FPS benchmark')
    parser.add_argument(
        '--config',
        type=str,
        default='baseline_hsitrack_trans_enc3',
        help='YAML name under experiments/sstrack/ (without .yaml)',
    )
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='Device for FPS/FLOPs')
    parser.add_argument('--fps_warmup', type=int, default=20)
    parser.add_argument('--fps_iters', type=int, default=100)
    parser.add_argument('--skip_fps', action='store_true')
    parser.add_argument('--skip_flops', action='store_true')
    args = parser.parse_args()

    _load_cfg(args.config)

    print(f'Config: experiments/sstrack/{args.config}.yaml')
    print(f'TEMPLATE_NUMBER={cfg.TEST.TEMPLATE_NUMBER}, SEARCH.LENGTH={cfg.DATA.SEARCH.LENGTH}, '
          f'TEMPLATE_SIZE={cfg.TEST.TEMPLATE_SIZE}, SEARCH_SIZE={cfg.TEST.SEARCH_SIZE}')

    model = build_sstrack(cfg, training=False)
    total_p, train_p = count_parameters(model)
    print(f'#Params (total):     {total_p / 1e6:.2f} M  ({total_p})')
    print(f'#Params (trainable): {train_p / 1e6:.2f} M  ({train_p})')

    device = torch.device(args.device if args.device == 'cpu' or torch.cuda.is_available() else 'cpu')
    if args.device == 'cuda' and not torch.cuda.is_available():
        print('CUDA not available; using CPU for timed tests.')
        device = torch.device('cpu')

    if not args.skip_flops:
        model_for_flops = build_sstrack(cfg, training=False).to(device)
        gmac, thop_p = measure_flops_thop(model_for_flops, cfg, device)
        del model_for_flops
        if gmac is not None:
            # Literature often reports GMAC or 2x as FLOPs (1 MAC ≈ 2 FLOPs for mul+add)
            print(f'GMAC (thop):         {gmac:.2f} GMac  (one forward pass)')
            print(f'Approx. FLOPs:       {2 * gmac:.2f} G  (if counting mul+add separately)')
            if thop_p is not None:
                print(f'thop param estimate: {thop_p / 1e6:.2f} M  (may differ from PyTorch count)')
        else:
            print('GMAC:                (skipped — install thop: pip install thop)')
    else:
        print('GMAC:                (skipped by --skip_flops)')

    if not args.skip_fps:
        model_fps = build_sstrack(cfg, training=False)
        fps = measure_fps(model_fps, cfg, device, args.fps_warmup, args.fps_iters)
        print(f'FPS ({device.type}):        {fps:.2f}  ({args.fps_iters} forwards after {args.fps_warmup} warmup)')
        print('Note: FPS = full model.forward (template + search list); not dataloader / I/O.')
    else:
        print('FPS:                 (skipped by --skip_fps)')

    print('Done.')


if __name__ == '__main__':
    main()
