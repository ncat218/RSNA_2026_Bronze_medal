"""Bounded inference wrapper for GoodPJW's Apache-2.0 ConvNeXt reader v1.

The reviewed, hash-pinned preprocess.py and knee.py are bundled by the notebook
builder. This wrapper changes batching and error reporting, not image semantics.
It never imports an attached Python script or enables network downloads.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gc
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

CONTRACT = "goodpjw_convnext_v1_3ema_6slot_k12_256_micro8_v1"


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def check_checkpoint(ck, fold):
    if not isinstance(ck, dict) or not isinstance(ck.get("args"), dict):
        raise ValueError("Checkpoint must contain model and args dictionaries")
    args = ck["args"]
    expected = {"fold": fold, "res": 256, "k_eval": 12, "epochs": 14}
    for key, value in expected.items():
        if args.get(key) != value:
            raise ValueError(f"Checkpoint {fold}: unexpected {key}={args.get(key)!r}")
    if str(args.get("backbone", "")).split(".")[0] != "convnext_tiny":
        raise ValueError("Wrong reader backbone")
    state = ck.get("model")
    if not isinstance(state, dict) or not state:
        raise ValueError("Empty reader model state")
    for key, shape in {"cls_w": (12, 384), "cls_b": (12,),
                       "slot_emb.weight": (6, 384), "proj.1.weight": (384, 768)}.items():
        if key not in state or tuple(state[key].shape) != shape:
            raise ValueError(f"Wrong reader tensor shape: {key}")
    if any(not torch.isfinite(t).all().item() for t in state.values()
           if isinstance(t, torch.Tensor) and t.is_floating_point()):
        raise ValueError("Nonfinite checkpoint tensor")
    return args


@torch.inference_mode()
def microbatch_probability(model, windows, mask, pos, *, res=256, microbatch=8,
                           device="cuda", amp=True):
    """Encode only valid native triplets, then run the unmodified study head.

    Deterministic make_views(..., train=False) uses the same theta for every
    window, so group=1 is equivalent to group=k, including partial microbatches.
    No graph, GPU image batch or full-resolution float volume is retained.
    """
    from knee import make_views
    if not isinstance(microbatch, int) or microbatch < 1:
        raise ValueError("Invalid image microbatch")
    if windows.dtype != np.uint8 or windows.ndim != 5 or windows.shape[2] != 3:
        raise ValueError("Expected CPU uint8 [6,k,3,H,W] windows")
    s, k = windows.shape[:2]
    if s != 6 or np.asarray(mask).shape != (s,) or np.asarray(pos).shape != (s, k):
        raise ValueError("Slot/position shape mismatch")
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        raise ValueError("Reader refuses an all-empty study")
    dev = torch.device(device)
    flat = windows[mask].reshape(-1, *windows.shape[2:])
    features = []
    for start in range(0, len(flat), microbatch):
        images = torch.from_numpy(np.ascontiguousarray(flat[start:start + microbatch])).to(dev)
        with torch.autocast(dev.type, dtype=torch.float16, enabled=amp and dev.type == "cuda"):
            views = make_views(images, res, False, group=1)
            f = model.enc(views.contiguous(memory_format=torch.channels_last))
        features.append(f)
        del images, views
    valid = torch.cat(features)
    feat = valid.new_zeros(1, s, k, valid.shape[-1])
    mask_t = torch.from_numpy(mask).to(dev).unsqueeze(0)
    feat[0, mask_t[0]] = valid.reshape(int(mask.sum()), k, -1)
    pos_t = torch.as_tensor(pos, dtype=torch.float32, device=dev).unsqueeze(0)
    with torch.autocast(dev.type, dtype=torch.float16, enabled=amp and dev.type == "cuda"):
        result = model.head(feat, mask_t, pos_t).float().sigmoid()[0]
    out = result.cpu().numpy().astype(np.float32)
    if out.shape != (12,) or not np.isfinite(out).all() or ((out < 0) | (out > 1)).any():
        raise ValueError("Invalid reader probabilities")
    return out


def load_windows(uid, table, root, stats, study_rows, k=12):
    from knee import window_centres
    from preprocess import load_series, SIZE
    x = np.zeros((6, k, 3, SIZE, SIZE), np.uint8)
    mask, pos = np.zeros(6, bool), np.zeros((6, k), np.float32)
    selected = []
    for slot, candidates in enumerate(table.get(uid, [[] for _ in range(6)])):
        if not candidates:
            stats["absent_slots"] += 1
            continue
        sr = candidates[0]  # same largest-capped-file-count routing as public infer.py
        stats["selected_series"] += 1
        # Match the released infer.py: one selected candidate per slot, and a
        # failed/short series leaves that slot empty. load_series itself skips
        # unreadable DICOM files; n_bad does not invalidate a usable volume.
        try:
            vol, info = load_series(str(root / uid / sr))
        except Exception:
            stats["unreadable_slots_skipped"] += 1
            continue
        stats["bad_slices_skipped"] += int(info.get("n_bad", 0))
        if len(vol) < 3:
            stats["short_slots_skipped"] += 1
            continue
        if vol.dtype != np.uint8 or vol.shape[1:] != (SIZE, SIZE):
            raise ValueError("Canonical volume contract changed")
        centres = window_centres(len(vol), k, False)
        x[slot] = np.stack([vol[np.clip([j - 1, j, j + 1], 0, len(vol) - 1)] for j in centres])
        mask[slot], pos[slot] = True, centres / max(len(vol) - 1, 1)
        selected.append({"slot": slot, "series": sr, "slices": len(vol), **info})
        del vol
    if not mask.any():
        stats["empty_studies"] += 1
        raise RuntimeError(f"No usable reader series: {uid}")
    study_rows.append({"StudyInstanceUID": uid, "available_slots": int(mask.sum()),
                       "selected": selected})
    return x, mask, pos


def run(args):
    from knee import LABELS, KneeNet, build_study_table
    import knee
    import preprocess
    import timm
    import cv2
    import pydicom
    import psutil
    started = time.monotonic()
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(Path(args.manifest).read_text())
    receipt = {"status": "RUNNING", "contract": CONTRACT, "labels": LABELS,
               "models": 3, "checkpoints": manifest["checkpoints"],
               "source_hashes": manifest["source_hashes"], "failures": [],
               "fallback_studies": 0, "empty_studies": 0}
    receipt_path = out_dir / "convnext_reader_receipt.json"
    stats, rows = Counter(), []

    def save_receipt():
        receipt.update(elapsed_seconds=time.monotonic() - started, dicom_stats=dict(stats),
                       empty_studies=stats["empty_studies"])
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")

    try:
        if timm.__version__ != "1.0.22" or cv2.__version__ != "4.12.0":
            raise RuntimeError(f"Reader dependency mismatch: timm={timm.__version__}, cv2={cv2.__version__}")
        if not torch.cuda.is_available():
            raise RuntimeError("Reader needs CUDA")
        torch.set_num_threads(4)
        torch.backends.cudnn.benchmark = False
        device = torch.device("cuda:0")
        torch.cuda.reset_peak_memory_stats()
        for module in (knee, preprocess):
            name = Path(module.__file__).name
            if sha256(module.__file__) != manifest["source_hashes"][name]:
                raise ValueError(f"Imported source SHA changed: {name}")
        data = Path(args.data)
        ids = pd.read_csv(data / "test.csv", dtype={"StudyInstanceUID": str}).StudyInstanceUID.tolist()
        sample = pd.read_csv(data / "sample_submission.csv", dtype={"StudyInstanceUID": str})
        if (list(sample.columns) != ["StudyInstanceUID", *LABELS] or
                len(ids) != len(set(ids)) or set(ids) != set(sample.StudyInstanceUID)):
            raise ValueError("Competition schema/cohort mismatch")
        se = pd.read_csv(data / "test_series.csv", dtype={"StudyInstanceUID": str, "SeriesInstanceUID": str})
        se = se[se.StudyInstanceUID.isin(set(ids))]
        root = data / "test_series"
        counts = {}
        for st, sr in zip(se.StudyInstanceUID, se.SeriesInstanceUID):
            directory = root / st / sr
            counts[sr] = min(len(os.listdir(directory)), preprocess.MAX_SLICES) if directory.is_dir() else 0
        table = build_study_table(se, counts)
        models = []
        for fold, item in enumerate(manifest["checkpoints"]):
            if sha256(item["path"]) != item["sha256"]:
                raise ValueError("Reader checkpoint SHA changed")
            ck = torch.load(item["path"], map_location="cpu", weights_only=True)
            cfg = check_checkpoint(ck, fold)
            m = KneeNet(str(cfg["backbone"]).split(".")[0], pretrained=False)
            m.load_state_dict(ck["model"], strict=True)
            models.append(m.to(device).to(memory_format=torch.channels_last).eval().requires_grad_(False))
            del ck
        gc.collect()
        # Instrument existing loader without changing its pixel/order semantics.
        original_read = preprocess._read
        def traced_read(path):
            frames = original_read(path)
            stats["decoded_files"] += 1
            stats["multi_frame_files"] += int(len(frames) > 1)
            for _, iop, ipp, ps, _ in frames:
                stats["frames_missing_iop"] += int(iop is None)
                stats["frames_missing_ipp"] += int(ipp is None)
                stats["frames_missing_spacing"] += int(ps is None)
            return frames
        preprocess._read = traced_read
        raw, parity = [], []
        rss_samples = []
        for index, uid in enumerate(ids):
            if time.monotonic() - started >= args.max_seconds:
                raise TimeoutError("Shared notebook time budget exhausted")
            if psutil.virtual_memory().available < 2 * 1024 ** 3:
                raise MemoryError("Less than 2 GiB host RAM available")
            x, mask, pos = load_windows(uid, table, root, stats, rows)
            probabilities = []
            for fold, model in enumerate(models):
                if index == 0:
                    # Low-memory reference check: full public forward with ONE
                    # real series (12 windows), not an unbounded 288-window batch.
                    one = np.zeros(6, bool)
                    one[np.flatnonzero(mask)[0]] = True
                    bounded = microbatch_probability(model, x, one, pos, device=device)
                    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                        reference = model(torch.from_numpy(x).unsqueeze(0).to(device),
                                          torch.from_numpy(one).unsqueeze(0).to(device),
                                          torch.from_numpy(pos).unsqueeze(0).to(device),
                                          res=256).float().sigmoid()[0].cpu().numpy()
                    diff = float(np.abs(reference - bounded).max())
                    if not np.isfinite(diff) or diff > 2e-3:
                        raise RuntimeError(f"Microbatch/reference forward mismatch {fold}: {diff}")
                    parity.append({"fold": fold, "single_real_series_max_abs_diff": diff})
                probabilities.append(microbatch_probability(model, x, mask, pos, device=device))
            raw.append(np.stack(probabilities))
            del x, mask, pos, probabilities
            if index == 0 or (index + 1) % 25 == 0 or index == len(ids) - 1:
                gc.collect()
                rss = psutil.Process().memory_info().rss / 1024 ** 3
                rss_samples.append({"studies_completed": index + 1, "rss_gib": rss})
                if rss > 12 or (len(rss_samples) > 1 and rss - rss_samples[0]["rss_gib"] > 4):
                    raise MemoryError("Reader host-RAM growth guard")
                print("CONVNEXT", index + 1, "/", len(ids), "seconds", round(time.monotonic() - started),
                      "rss_gib", round(rss, 3), flush=True)
        values = np.stack(raw, axis=1)  # [3,N,12], not features/graphs
        mean = values.mean(axis=0)
        frame = pd.DataFrame(mean, columns=LABELS)
        frame.insert(0, "StudyInstanceUID", ids)
        output = out_dir / "convnext_probabilities.csv"
        frame.to_csv(output, index=False)
        np.savez_compressed(out_dir / "convnext_fold_probabilities.npz",
                            study_uids=np.asarray(ids), labels=np.asarray(LABELS), probabilities=values)
        (out_dir / "selected_series.json").write_text(json.dumps(rows, indent=2))
        receipt.update(status="COMPLETE", studies=len(ids), output_sha256=sha256(output),
                       parity_checks=parity, parity_tolerance=2e-3, rss_samples=rss_samples,
                       peak_cuda_allocated_gib=torch.cuda.max_memory_allocated() / 1024 ** 3,
                       peak_cuda_reserved_gib=torch.cuda.max_memory_reserved() / 1024 ** 3,
                       gpu=torch.cuda.get_device_name(0), timm=timm.__version__, opencv=cv2.__version__,
                       torch=torch.__version__, pydicom=pydicom.__version__,
                       warning="Single-series batching parity only; no independent full-stack OOF/Private claim")
        save_receipt()
    except Exception as exc:
        receipt["status"] = "FAILED"
        receipt["failures"].append(f"{type(exc).__name__}: {exc}")
        save_receipt()
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-seconds", type=float, required=True)
    run(parser.parse_args())
