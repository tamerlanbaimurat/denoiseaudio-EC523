from __future__ import print_function
import sys
import io
import json
import os
import random
import threading
import time
from urllib.parse import quote

import matplotlib.pyplot as plt
import numpy as np
import requests
from requests.adapters import HTTPAdapter
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from fla.layers import DeltaNet

from torch_pesq import PesqLoss

#Sets the Randomized Seed to itself
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set to {seed}")

#Loading the Dataset ------------------------------------------------------------------------
def public_url(key: str) -> str:
    return f"{BASE_URL}/{quote(key.lstrip('/'), safe='/')}"

_local = threading.local()

def _session() -> requests.Session:
    if not hasattr(_local, "s"):
        s = requests.Session()
        adapter = HTTPAdapter(pool_connections=64, pool_maxsize=64)
        s.mount("https://", adapter)
        _local.s = s
    return _local.s

def fetch_bytes(key: str, timeout: int = 60) -> bytes:
    for attempt in range(MAX_RETRIES):
        try:
            r = _session().get(public_url(key), timeout=timeout)
            if r.status_code == 503:
                raise requests.exceptions.ConnectionError("503")
            r.raise_for_status()
            return r.content
        except (requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
                requests.exceptions.HTTPError):
            if attempt == MAX_RETRIES - 1:
                raise
            time.sleep(2 ** attempt + random.random())
    return b""

def fetch_manifest(split: str) -> list[dict]:
    key = f"{PREFIX}/manifests/{split}.jsonl"
    text = fetch_bytes(key).decode()
    return [json.loads(line) for line in text.splitlines() if line.strip()]

class AdaptivePool:
    """AIMD concurrency: starts fast, backs off on 503, recovers quickly."""

    def __init__(self, initial=32, minimum=4, maximum=64, grow_every=20):
        self._sem_value = initial
        self._sem = threading.Semaphore(initial)
        self._lock = threading.Lock()
        self._min = minimum
        self._max = maximum
        self._grow_every = grow_every
        self.successes = 0
        self.errors = 0

    @property
    def window(self):
        return self._sem_value

    def acquire(self):
        self._sem.acquire()

    def release_success(self):
        with self._lock:
            self.successes += 1
            if self.successes % self._grow_every == 0 and self._sem_value < self._max:
                self._sem_value += 1
                self._sem.release()
                return
        self._sem.release()

    def release_error(self):
        with self._lock:
            self.errors += 1
            new = max(self._sem_value // 2, self._min)
            while self._sem_value > new:
                self._sem.acquire(blocking=False) or None
                self._sem_value -= 1
        self._sem.release()

class STFTDataset(Dataset):
    """Fetches clean/noisy STFT pairs from HTTP, with in-memory caching."""

    def __init__(self, records: list[dict], split: str):
        self.records = records
        self.split = split
        self._cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

    @classmethod
    def from_manifest(cls, split: str) -> "STFTDataset":
        records = fetch_manifest(split)
        print(f"[{split}] manifest loaded: {len(records)} pairs")
        return cls(records, split)

    def prefetch_all(self):
        """Download every pair into RAM with adaptive concurrency."""
        from concurrent.futures import ThreadPoolExecutor, as_completed

        to_fetch = [i for i in range(len(self.records)) if i not in self._cache]
        if not to_fetch:
            print(f"[{self.split}] all {len(self.records)} pairs already cached")
            return

        pool = AdaptivePool(initial=32, minimum=4, maximum=128)

        def _download(idx):
            pool.acquire()
            try:
                rec = self.records[idx]
                noisy_bytes = fetch_bytes(rec["noisy_stft_key"])
                clean_bytes = fetch_bytes(rec["clean_stft_key"])
                noisy = torch.load(io.BytesIO(noisy_bytes), map_location="cpu",
                                   weights_only=True)["stft"]
                clean = torch.load(io.BytesIO(clean_bytes), map_location="cpu",
                                   weights_only=True)["stft"]
                pool.release_success()
                return idx, noisy, clean
            except Exception:
                pool.release_error()
                raise

        print(f"[{self.split}] prefetching {len(to_fetch)} pairs "
              f"(adaptive window: {pool.window}→{pool._max}) …")
        t0 = time.perf_counter()
        done, errors = 0, 0

        # Use a large thread pool; AdaptivePool's semaphore controls actual concurrency
        with ThreadPoolExecutor(max_workers=64) as ex:
            futures = {ex.submit(_download, i): i for i in to_fetch}
            for fut in as_completed(futures):
                try:
                    idx, noisy, clean = fut.result()
                    self._cache[idx] = (noisy, clean)
                    done += 1
                except Exception as e:
                    errors += 1
                    if errors <= 3:
                        print(f"  ERROR idx={futures[fut]}: {e}")
                total = done + errors
                if total % 500 == 0 or total == len(to_fetch):
                    elapsed = time.perf_counter() - t0
                    print(f"  {total}/{len(to_fetch)}  {elapsed:.0f}s  "
                          f"{done/max(elapsed,1):.0f} pairs/s  "
                          f"window={pool.window}  errors={errors}")
        elapsed = time.perf_counter() - t0
        print(f"[{self.split}] prefetch done: {done} OK, {errors} errors, "
              f"{elapsed:.0f}s")

        # Sequential retry for any failures
        remaining = [i for i in to_fetch if i not in self._cache]
        if remaining:
            print(f"[{self.split}] retrying {len(remaining)} failures …")
            for idx in remaining:
                try:
                    _, noisy, clean = _download(idx)
                    self._cache[idx] = (noisy, clean)
                except Exception as e:
                    print(f"  SKIP idx={idx}: {e}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        rec = self.records[idx]

        if idx in self._cache:
            noisy, clean = self._cache[idx]
        else:
            noisy = torch.load(io.BytesIO(fetch_bytes(rec["noisy_stft_key"])),
                               map_location="cpu", weights_only=True)["stft"]
            clean = torch.load(io.BytesIO(fetch_bytes(rec["clean_stft_key"])),
                               map_location="cpu", weights_only=True)["stft"]
            self._cache[idx] = (noisy, clean)

        return {
            "pair_id":     rec["pair_id"],
            "split":       self.split,
            "sample_rate": int(rec["sample_rate"]),
            "noisy":       noisy,          # complex64 [257, T]
            "clean":       clean,          # complex64 [257, T]
            "length":      noisy.shape[-1],
        }
#--------------------------------------------------------------------------------------------

#Data Helpers and DataLoaders ---------------------------------------------------------------
def stft_to_mag_phase(stft_complex):
    mag = stft_complex.abs()
    phase = stft_complex.angle()
    return mag, phase


#   helps load time_buckets
def _bucket_pad(t: int) -> int:
    """Round up to the nearest bucket size so Triton sees fewer unique shapes."""
    for b in TIME_BUCKETS:
        if t <= b:
            return b
    return ((t + 63) // 64) * 64

#   finds the magnitude of the signals
def collate_magnitude(batch):
    """
    Custom collate that converts complex STFTs to log-magnitude tensors
    and pads to a fixed bucket size (reduces Triton recompilations).

    Returns dict with keys:
        noisy_mag:   [B, 1, 257, T_bucket]  log1p magnitude
        clean_mag:   [B, 1, 257, T_bucket]  log1p magnitude
        noisy_phase: [B, 257, T_bucket]     phase (for reconstruction)
        lengths:     [B]                     original T per sample
        pair_ids:    list[str]
    """
    max_t = max(item["noisy"].shape[-1] for item in batch)
    pad_to = _bucket_pad(max_t)

    noisy_mags, clean_mags, noisy_phases, lengths, pair_ids = [], [], [], [], []

    for item in batch:
        n_mag, n_phase = stft_to_mag_phase(item["noisy"])
        c_mag, _       = stft_to_mag_phase(item["clean"])

        pad_t = pad_to - n_mag.shape[-1]
        n_mag   = F.pad(n_mag, (0, pad_t))
        c_mag   = F.pad(c_mag, (0, pad_t))
        n_phase = F.pad(n_phase, (0, pad_t))

        noisy_mags.append(torch.log1p(n_mag).unsqueeze(0))   # [1, 257, T]
        clean_mags.append(torch.log1p(c_mag).unsqueeze(0))
        noisy_phases.append(n_phase)
        lengths.append(item["length"])
        pair_ids.append(item["pair_id"])

    return {
        "noisy_mag":   torch.stack(noisy_mags),     # [B, 1, 257, T]
        "clean_mag":   torch.stack(clean_mags),      # [B, 1, 257, T]
        "noisy_phase": torch.stack(noisy_phases),    # [B, 257, T]
        "lengths":     torch.tensor(lengths, dtype=torch.int64),
        "pair_ids":    pair_ids,
    }
#--------------------------------------------------------------------------------------------

#Model Architecture -------------------------------------------------------------------------
from torch.utils.checkpoint import checkpoint as grad_checkpoint

# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

def _norm(out_ch):
    # GroupNorm: batch-size-independent and train/eval-identical,
    # which removes one big source of test-loss jitter at BATCH_SIZE=8.
    return nn.GroupNorm(num_groups=min(8, out_ch), num_channels=out_ch)

class ConvBlock(nn.Module):
    """Conv2d -> GroupNorm -> PReLU. Optionally downsamples frequency via stride."""
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=(1, 1), padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding),
            _norm(out_ch),
            nn.PReLU(out_ch),
        )

    def forward(self, x):
        return self.block(x)

class DeconvBlock(nn.Module):
    """ConvTranspose2d -> GroupNorm -> PReLU. Upsamples frequency via stride."""
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=(2, 1),
                 padding=1, output_padding=(1, 0)):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size, stride=stride,
                               padding=padding, output_padding=output_padding),
            _norm(out_ch),
            nn.PReLU(out_ch),
        )

    def forward(self, x):
        return self.block(x)

#    ---------------------------------------------------------------------------
#    Attention blocks
#    ---------------------------------------------------------------------------

class DeltaNetBlock(nn.Module):
    """Pre-norm DeltaNet attention + FFN with residual connections."""
    def __init__(self, d_model, num_heads, ffn_ratio=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = DeltaNet(
            d_model=d_model,
            num_heads=num_heads,
            use_short_conv=True,
            conv_size=4,
            use_beta=True,
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * ffn_ratio),
            nn.GELU(),
            nn.Linear(d_model * ffn_ratio, d_model),
        )
#ABLATION **-----------------------------------------------------------------------------**
    def forward(self, x):
        # h = self.norm1(x)
        # h, *_ = self.attn(h)
        # x = x + h
        x = x + self.ffn(self.norm2(x))
        return x


class FullAttentionBlock(nn.Module):
    """Pre-norm self-attention using Flash Attention via scaled_dot_product_attention."""
    def __init__(self, d_model, num_heads, ffn_ratio=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.norm1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * ffn_ratio),
            nn.GELU(),
            nn.Linear(d_model * ffn_ratio, d_model),
        )

    def forward(self, x):
        B, T, D = x.shape
        h = self.norm1(x)
        qkv = self.qkv(h).reshape(B, T, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)  # each [B, H, T, D_h]
        h = F.scaled_dot_product_attention(q, k, v)       # Flash Attention, O(T) memory
        h = h.transpose(1, 2).reshape(B, T, D)
        h = self.out_proj(h)
        x = x + h
        x = x + self.ffn(self.norm2(x))
        return x

#    ---------------------------------------------------------------------------
#    Main model
#    ---------------------------------------------------------------------------

class SpeechDenoiser(nn.Module):
    """
    CNN encoder-decoder with separate time/frequency attention branches for
    denoising and noisy/clean classification.

    Shared encoder:
        Converts the input STFT magnitude into a compact [B, 4C, 65, T] feature map.

    Task-specific attention branches:
        Each branch uses one TF pair by default:
        - TIME  (DeltaNet):      each frequency bin attends across time frames.
        - FREQ  (FullAttention): each time frame attends across frequency bins.
        The denoiser and classifier therefore both see temporal context and
        cross-frequency structure, but their attention parameters are separate.

    Classifier:
        Reads from its own attention branch and produces a noisy/clean logit.
        Its sigmoid probability is later used as a soft gate that controls how
        strongly the denoising mask is applied.

    Outputs:
        mask         [B, 1, 257, T]  sigmoid mask in [0, 1], applied in LINEAR magnitude.
        residual_raw [B, 1, 257, T]  bounded log-magnitude residual correction.
        cls_logit    [B, 1]          noisy/clean classification logit (raw, pre-sigmoid).
    """

    def __init__(self, channels=64, n_heads=8, n_tf_pairs=1, ffn_ratio=4):
        super().__init__()
        C = channels
        bottleneck_dim = C * 4   # denoiser branch width
        cls_dim = C * 2          # lighter classifier branch width
        assert bottleneck_dim % n_heads == 0, "denoiser width must be divisible by n_heads"
        assert cls_dim % n_heads == 0, "classifier width must be divisible by n_heads"

        # --- Encoder ---
        self.enc1 = ConvBlock(1, C)                                  # -> [B, 64, 257, T]
        self.enc2 = ConvBlock(C, C * 2, stride=(2, 1))              # -> [B, 128, 129, T]
        self.enc3 = ConvBlock(C * 2, bottleneck_dim, stride=(2, 1)) # -> [B, 256, 65, T]

        # --- Task-specific attention branches ---
        # Each TF pair runs DeltaNet over time, then Flash Attention over frequency.
        self.denoise_attn_layers = nn.ModuleList()
        self.classify_proj = nn.Conv2d(bottleneck_dim, cls_dim, kernel_size=1)
        self.classify_attn_layers = nn.ModuleList()
        for _ in range(n_tf_pairs):
            self.denoise_attn_layers.append(
                DeltaNetBlock(bottleneck_dim, n_heads, ffn_ratio))
            self.denoise_attn_layers.append(
                FullAttentionBlock(bottleneck_dim, n_heads, ffn_ratio))
            self.classify_attn_layers.append(
                DeltaNetBlock(cls_dim, n_heads, max(1, ffn_ratio // 2)))
            self.classify_attn_layers.append(
                FullAttentionBlock(cls_dim, n_heads, max(1, ffn_ratio // 2)))
        self.denoise_attn_norm = nn.LayerNorm(bottleneck_dim)
        self.classify_attn_norm = nn.LayerNorm(cls_dim)

        # --- Decoder (with skip connections) ---
        self.dec3 = DeconvBlock(bottleneck_dim, C * 2, stride=(2, 1),
                                padding=1, output_padding=(1, 0))
        self.dec2 = DeconvBlock(C * 4, C, stride=(2, 1),
                                padding=1, output_padding=(0, 0))
        # Two output channels: mask logits and residual correction.
        self.out = nn.Conv2d(C * 2, 2, kernel_size=3, padding=1)

        # --- Classification head (reads classifier attention output) ---
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(cls_dim, 1),
        )

    def _apply_layer(self, layer, h):
        if self.training:
            return grad_checkpoint(layer, h, use_reentrant=False)
        return layer(h)

    def _run_tf_attention(self, e3, layers, norm):
        """
        Run alternating time/frequency attention on an STFT feature map.
        Input : [B, C, F, T]
        Output: [B, C, F, T]

        Even-index layers are DeltaNet blocks over TIME: each frequency bin gets
        the full temporal sequence. Odd-index layers are Flash Attention blocks
        over FREQ: each time frame gets the full set of frequency bins.
        """
        B, C, Fbins, T = e3.shape
        h = e3.permute(0, 2, 3, 1).contiguous()         # [B, F, T, C]

        for i, layer in enumerate(layers):
            if i % 2 == 0:
                # TIME attention: pack F into batch -> [B*F, T, C]
                h = h.reshape(B * Fbins, T, C)
                h = self._apply_layer(layer, h)
                h = h.reshape(B, Fbins, T, C)
            else:
                # FREQ attention: pack T into batch -> [B*T, F, C]
                h = h.transpose(1, 2).contiguous()       # [B, T, F, C]
                h = h.reshape(B * T, Fbins, C)
                h = self._apply_layer(layer, h)
                h = h.reshape(B, T, Fbins, C).transpose(1, 2).contiguous()

        h = norm(h)                                      # LayerNorm over C
        return h.permute(0, 3, 1, 2).contiguous()        # [B, C, F, T]

    def forward(self, x):
        # x: [B, 1, 257, T]  (log1p magnitude)

        # Encoder
        e1 = self.enc1(x)    # [B, C, 257, T]
        e2 = self.enc2(e1)   # [B, 2C, 129, T]
        e3 = self.enc3(e2)   # [B, 4C, 65, T]

        # Separate task-specific TF attention branches.
        denoise_h = self._run_tf_attention(
            e3, self.denoise_attn_layers, self.denoise_attn_norm
        )                                                # [B, 4C, 65, T]
        classify_h = self.classify_proj(e3)              # [B, 2C, 65, T]
        classify_h = self._run_tf_attention(
            classify_h, self.classify_attn_layers, self.classify_attn_norm
        )                                                # [B, 2C, 65, T]

        cls_logit = self.classifier(classify_h)          # [B, 1]

        # Decoder with skip connections
        d3 = self.dec3(denoise_h)                        # [B, 2C, ?, T]
        d3 = d3[..., :e2.shape[-2], :]                   # crop freq to match enc2
        d2 = self.dec2(torch.cat([d3, e2], dim=1))       # [B, C, ?, T]
        d2 = d2[..., :e1.shape[-2], :]                   # crop freq to match enc1
        out = self.out(torch.cat([d2, e1], dim=1))       # [B, 2, 257, T]
        mask = torch.sigmoid(out[:, :1])                 # [B, 1, 257, T]
        residual_raw = out[:, 1:2]                       # [B, 1, 257, T]

        return mask, residual_raw, cls_logit
#--------------------------------------------------------------------------------------------

#Checkpoint Utilities -------------------------------------------------------------------------

def save_checkpoint(model, optimizer, scheduler, epoch, history, path):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "history": history,
    }, path)

def load_checkpoint(path, model, optimizer, scheduler):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    print(f"Resumed from epoch {ckpt['epoch']}")
    return ckpt["epoch"], ckpt["history"]

#    ---------------------------------------------------------------------------
#    Training & evaluation
#    ---------------------------------------------------------------------------

def noise_gate_from_logits(cls_logit, temperature=1.0):
    """Convert noisy/clean logits into a broadcastable soft gate in [0, 1]."""
    return torch.sigmoid(cls_logit.float() / temperature).view(-1, 1, 1, 1)

def apply_gated_hybrid_in_linear(
    mask,
    residual_raw,
    cls_logit,
    noisy_log_mag,
    oracle_gate=None,
    gate_temperature=1.0,
    detach_gate=False,
):
    """
    Apply hybrid mask+residual denoising through classifier-controlled soft gating.

    The mask removes energy in linear magnitude. The residual is a bounded
    log-magnitude correction that can add back speech energy after masking.
    The gate then blends between the original and denoised magnitudes:
        gated = (1 - p_noisy) * noisy + p_noisy * denoised
    """
    noisy_lin = torch.expm1(noisy_log_mag.float())
    masked_log = torch.log1p((mask.float() * noisy_lin).clamp(min=0.0))
    residual_log = RESIDUAL_SCALE * torch.tanh(residual_raw.float())
    denoised_log = (masked_log + residual_log).clamp(min=0.0)
    denoised_lin = torch.expm1(denoised_log)

    if oracle_gate is None:
        gate = noise_gate_from_logits(cls_logit, gate_temperature)
    else:
        gate = oracle_gate.float().view(-1, 1, 1, 1)
    if detach_gate:
        gate = gate.detach()

    gated_lin = noisy_lin + gate * (denoised_lin - noisy_lin)
    return torch.log1p(gated_lin.clamp(min=0.0)), gate

def _time_mask_like(x, lengths):
    B, C, Fbins, T = x.shape
    time_mask = torch.arange(T, device=x.device)[None, :] < lengths[:, None]
    return time_mask[:, None, None, :].expand(B, C, Fbins, T).to(x.dtype)

def masked_l1_loss(est_log, target_log, lengths):
    """L1 averaged only over real (non-padded) time frames."""
    time_mask = _time_mask_like(est_log, lengths)
    abs_err = (est_log - target_log).abs() * time_mask
    return abs_err.sum() / time_mask.sum().clamp(min=1.0)

def masked_spectral_convergence_loss(est_log, target_log, lengths):
    """Relative linear-magnitude error, averaged per item over non-padded frames."""
    time_mask = _time_mask_like(est_log, lengths)
    est_lin = torch.expm1(est_log.float()) * time_mask
    target_lin = torch.expm1(target_log.float()) * time_mask
    diff = (est_lin - target_lin).reshape(est_log.shape[0], -1)
    target_flat = target_lin.reshape(target_log.shape[0], -1)
    numerator = torch.linalg.vector_norm(diff, ord=2, dim=1)
    denominator = torch.linalg.vector_norm(target_flat, ord=2, dim=1).clamp(min=1e-8)
    return (numerator / denominator).mean()

def _build_multitask_batch(noisy_mag, clean_mag, lengths, device):
    """
    Stack [noisy, clean] along the batch dim so the model sees both in one forward.
    - Noisy inputs should reconstruct the paired clean target.
    - Clean inputs should reconstruct themselves, which teaches the gate to stay quiet.
    - Classification loss uses both halves (noisy=1, clean=0).
    """
    inputs  = torch.cat([noisy_mag, clean_mag], dim=0)      # [2B, 1, 257, T]
    targets = torch.cat([clean_mag, clean_mag], dim=0)      # [2B, 1, 257, T]
    target_lengths = torch.cat([lengths, lengths], dim=0)   # [2B]
    B = noisy_mag.shape[0]
    labels  = torch.cat(
        [torch.ones(B, 1, device=device), torch.zeros(B, 1, device=device)],
        dim=0,
    )                                                       # [2B, 1]
    return inputs, targets, target_lengths, labels

def train_one_epoch(model, loader, optimizer, device, epoch):
    model.train()
    use_oracle_gate = epoch <= GATE_WARMUP_EPOCHS
    running_l1 = 0.0
    running_sc = 0.0
    running_cls = 0.0
    n = 0
    optimizer.zero_grad()

    for step, batch in enumerate(loader):
        noisy_mag = batch["noisy_mag"].to(device)   # [B, 1, 257, T] log1p mag
        clean_mag = batch["clean_mag"].to(device)
        lengths   = batch["lengths"].to(device)

        inputs, targets, target_lengths, labels = _build_multitask_batch(
            noisy_mag, clean_mag, lengths, device
        )

        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            mask_all, residual_all, cls_logit = model(inputs)  # [2B,1,257,T], [2B,1,257,T], [2B,1]

        # During warmup, use the known noisy/clean labels as the gate so the
        # denoiser learns before relying on the predicted classifier gate.
        oracle_gate = labels if use_oracle_gate else None
        est_log, _ = apply_gated_hybrid_in_linear(
            mask_all, residual_all, cls_logit, inputs, oracle_gate=oracle_gate
        )
        l1_loss = masked_l1_loss(est_log, targets, target_lengths)
        sc_loss = masked_spectral_convergence_loss(est_log, targets, target_lengths)
        denoise_loss = l1_loss + LAMBDA_SC * sc_loss

        # Classification: supervise on both halves.
        cls_loss = classify_criterion(cls_logit.float(), labels)

        loss = (denoise_loss + LAMBDA_CLS * cls_loss) / GRAD_ACCUM_STEPS
        loss.backward()

        if (step + 1) % GRAD_ACCUM_STEPS == 0 or (step + 1) == len(loader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

        running_l1  += l1_loss.item()
        running_sc  += sc_loss.item()
        running_cls += cls_loss.item()
        n += 1

    return running_l1 / max(n, 1), running_sc / max(n, 1), running_cls / max(n, 1)

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    running_l1 = 0.0
    running_sc = 0.0
    running_cls = 0.0
    n = 0

    for batch in loader:
        noisy_mag = batch["noisy_mag"].to(device)
        clean_mag = batch["clean_mag"].to(device)
        lengths   = batch["lengths"].to(device)

        inputs, targets, target_lengths, labels = _build_multitask_batch(
            noisy_mag, clean_mag, lengths, device
        )

        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            mask_all, residual_all, cls_logit = model(inputs)

        est_log, _ = apply_gated_hybrid_in_linear(mask_all, residual_all, cls_logit, inputs)
        l1_loss = masked_l1_loss(est_log, targets, target_lengths)
        sc_loss = masked_spectral_convergence_loss(est_log, targets, target_lengths)
        cls_loss = classify_criterion(cls_logit.float(), labels)

        running_l1  += l1_loss.item()
        running_sc  += sc_loss.item()
        running_cls += cls_loss.item()
        n += 1

    return running_l1 / max(n, 1), running_sc / max(n, 1), running_cls / max(n, 1)
#--------------------------------------------------------------------------------------------

#Data Helpers and DataLoaders ---------------------------------------------------------------
#--------------------------------------------------------------------------------------------

#Data Helpers and DataLoaders ---------------------------------------------------------------
#--------------------------------------------------------------------------------------------

if __name__ == "__main__":
    #sets the seed
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    #setup dataset
    BASE_URL = "https://ec523.tamerlanbaimurat.com"
    MAX_RETRIES = 5
    _local = threading.local()

    #Define training parameters
    LAMBDA_CLS = 0.1
    LAMBDA_SC = 0.1
    GATE_WARMUP_EPOCHS = 5
    GATE_TEMPERATURE = 1.0       # lower = sharper noisy/clean gate; 1.0 is standard sigmoid
    RESIDUAL_SCALE = 0.5         # max absolute log-magnitude correction from residual head
    DETACH_GATE_FOR_DENOISE = False
    classify_criterion = nn.BCEWithLogitsLoss()

    GRAD_ACCUM_STEPS = 2   # effective batch = BATCH_SIZE * GRAD_ACCUM_STEPS = 8 * 2 = 16

    #Define Checkpoint Directory
    CHECKPOINT_DIR = './checkpoints'
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    #training set loading
    PREFIX = "ec523project"
    train_ds = STFTDataset.from_manifest("train")
    # Download all data into RAM (~30.5 GB for 23K pairs).
    # This takes ~500 seconds to fetch
    train_ds.prefetch_all()
    #test set loading
    # PREFIX = "ec523project"
    test_ds_1  = STFTDataset.from_manifest("test")
    test_ds_1.prefetch_all()

    #initialize Time buckets for signals
    TIME_BUCKETS = [100, 200, 300, 400, 500, 600, 800, 1000, 1200, 1600]

    #create the trainloader and testloader
    BATCH_SIZE = 8
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_magnitude,
        num_workers=0,   # required: HTTP session is not picklable
    )
    test_loader_1 = DataLoader(
        test_ds_1,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_magnitude,
        num_workers=0,
    )

    model = SpeechDenoiser(channels=48, n_heads=4, n_tf_pairs=1, ffn_ratio=2).to(device)
    # If you have a larger GPU, you can try: SpeechDenoiser(channels=96, n_heads=4, n_tf_pairs=1, ffn_ratio=4)

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {param_count:,}")

    # ---------------------------------------------------------------------------
    # Hyperparameters
    # ---------------------------------------------------------------------------
    EPOCHS     = 50
    LR         = 3e-4
    SAVE_EVERY = 5          # save checkpoint every N epochs

    torch.cuda.empty_cache()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    start_epoch = 0
    history = {
        "train_l1": [], "train_sc": [], "train_cls": [], "train_total": [],
        "test_l1": [], "test_sc": [], "test_cls": [], "test_total": [],
    }

    resume_path = os.path.join(CHECKPOINT_DIR, "latest_no_deltanet.pt")
    if os.path.exists(resume_path):
        start_epoch, history = load_checkpoint(resume_path, model, optimizer, scheduler)
        for k in ("train_l1", "train_sc", "train_cls", "train_total",
                "test_l1", "test_sc", "test_cls", "test_total"):
            history.setdefault(k, [])
    
    # ---------------------------------------------------------------------------
    # Training loop
    # ---------------------------------------------------------------------------
    best_test_total = min(history["test_total"]) if history["test_total"] else float("inf")

    for epoch in range(start_epoch + 1, EPOCHS + 1):
        t0 = time.perf_counter()

        train_l1, train_sc, train_cls = train_one_epoch(model, train_loader, optimizer, device, epoch)
        test_l1,  test_sc,  test_cls  = evaluate(model, test_loader_1, device)
        train_total = train_l1 + LAMBDA_SC * train_sc
        test_total = test_l1 + LAMBDA_SC * test_sc

        scheduler.step()
        elapsed = time.perf_counter() - t0

        history["train_l1"].append(train_l1)
        history["train_sc"].append(train_sc)
        history["train_cls"].append(train_cls)
        history["train_total"].append(train_total)
        history["test_l1"].append(test_l1)
        history["test_sc"].append(test_sc)
        history["test_cls"].append(test_cls)
        history["test_total"].append(test_total)

        lr_now = scheduler.get_last_lr()[0]
        gate_mode = "oracle" if epoch <= GATE_WARMUP_EPOCHS else "pred"
        print(f"Epoch {epoch:03d}/{EPOCHS} ({elapsed:.0f}s, gate={gate_mode}) | "
            f"Train L1={train_l1:.4f} SC={train_sc:.4f} cls={train_cls:.4f} | "
            f"Test  L1={test_l1:.4f} SC={test_sc:.4f} cls={test_cls:.4f} | "
            f"LR={lr_now:.2e}")

        # Save best model using the reconstruction objective: L1 + LAMBDA_SC * spectral convergence.
        if test_total < best_test_total:
            best_test_total = test_total
            save_checkpoint(model, optimizer, scheduler, epoch, history,
                            os.path.join(CHECKPOINT_DIR, "best_no_deltanet.pt"))
            print(f"  -> New best test total: {best_test_total:.4f}")

        # Periodic checkpoint
        if epoch % SAVE_EVERY == 0:
            save_checkpoint(model, optimizer, scheduler, epoch, history,
                            os.path.join(CHECKPOINT_DIR, f"epoch_{epoch:03d}.pt"))

        # Always save latest (for resume)
        save_checkpoint(model, optimizer, scheduler, epoch, history,
                        os.path.join(CHECKPOINT_DIR, "latest_no_deltanet.pt"))

    print(f"\nTraining complete. Best test total: {best_test_total:.4f}")