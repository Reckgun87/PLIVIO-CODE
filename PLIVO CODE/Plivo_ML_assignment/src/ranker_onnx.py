# src/ranker_onnx.py
"""
PseudoLikelihoodRanker — batched masked-LM pseudo-likelihood scorer.

Features:
 - Batched-once ONNX run across all (candidate, mask_position) rows for a single utterance
 - Caps masks per candidate (self.max_masks) and candidates per utterance (stage-1 ensures candidate cap)
 - ONNX INT8 path preferred. PyTorch DistilBERT fallback (batched) if ONNX not available.
 - Short-circuits for obvious emails / digit-only candidates.
 - Heuristic scoring for candidates with no maskable tokens.
"""
from typing import List, Optional, Tuple
import os
import re
import math
import numpy as np

# set thread envs early to limit oversubscription
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

# optional imports (guarded)
try:
    import onnxruntime as ort
except Exception:
    ort = None

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForMaskedLM
except Exception:
    torch = None
    AutoTokenizer = None
    AutoModelForMaskedLM = None

from src.rules import is_valid_email

# defaults / knobs (tune via env vars if needed)
DEFAULT_MODEL_NAME = "distilbert-base-uncased"
DEFAULT_ONNX_PATH = os.environ.get("DISTIL_ONNX_PATH", "models/distilbert-base-uncased.int8.onnx")
MAX_LENGTH = int(os.environ.get("RANKER_MAX_LEN", "64"))
MAX_MASKS = int(os.environ.get("RANKER_MAX_MASKS", "1"))  # default aggressive
ONNX_INTRA_THREADS = int(os.environ.get("ORT_INTRA_THREADS", "1"))
ONNX_INTER_THREADS = int(os.environ.get("ORT_INTER_THREADS", "1"))
TORCH_NUM_THREADS = int(os.environ.get("TORCH_NUM_THREADS", "2"))
DEVICE = os.environ.get("RANKER_DEVICE", "cpu")


def _log_softmax_numpy(arr: np.ndarray) -> np.ndarray:
    """Stable log-softmax over last axis for 2D (B, V) or 1D (V)."""
    if arr.ndim == 1:
        m = arr.max()
        a = arr - m
        ea = np.exp(a)
        return a - np.log(ea.sum())
    else:
        m = arr.max(axis=1, keepdims=True)
        a = arr - m
        ea = np.exp(a)
        s = ea.sum(axis=1, keepdims=True)
        return a - np.log(s)


class PseudoLikelihoodRanker:
    def __init__(self,
                 model_name: str = DEFAULT_MODEL_NAME,
                 onnx_path: Optional[str] = DEFAULT_ONNX_PATH,
                 device: str = DEVICE,
                 max_length: int = MAX_LENGTH,
                 max_masks: int = MAX_MASKS):
        if AutoTokenizer is None:
            raise RuntimeError("transformers is required (pip install transformers)")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.device = device
        self.max_length = max_length
        self.max_masks = max_masks

        self.onnx_sess = None
        self.torch_model = None

        # Try ONNX session first
        if ort is not None and onnx_path and os.path.exists(onnx_path):
            sess_opts = ort.SessionOptions()
            sess_opts.intra_op_num_threads = ONNX_INTRA_THREADS
            sess_opts.inter_op_num_threads = ONNX_INTER_THREADS
            # try to disable memory pattern for lower tail latency
            try:
                sess_opts.enable_mem_pattern = False
            except Exception:
                pass
            providers = ["CPUExecutionProvider"]
            try:
                self.onnx_sess = ort.InferenceSession(onnx_path, sess_options=sess_opts, providers=providers)
            except Exception as e:
                print("WARN: failed to create ONNX session:", e)
                self.onnx_sess = None

        # Torch fallback
        if self.onnx_sess is None and torch is not None and AutoModelForMaskedLM is not None:
            try:
                self.torch_model = AutoModelForMaskedLM.from_pretrained(model_name)
                self.torch_model.eval()
                try:
                    torch.set_num_threads(TORCH_NUM_THREADS)
                except Exception:
                    pass
                if device != "cpu":
                    try:
                        self.torch_model.to(device)
                    except Exception:
                        pass
            except Exception as e:
                print("WARN: failed to load torch model:", e)
                self.torch_model = None

    # -------------------------
    # helpers: select mask positions
    # -------------------------
    def _select_positions(self, token_ids: np.ndarray, attention_mask: np.ndarray) -> List[int]:
        """
        Choose token positions to mask: interior positions, preferring alphanumeric tokens.
        Returns a list of positions (<= self.max_masks).
        """
        token_strs = self.tokenizer.convert_ids_to_tokens(token_ids)
        seq_len = int(attention_mask.sum())
        # internal positions only (skip first & last)
        valid = [i for i in range(1, max(1, seq_len - 1)) if i < len(token_strs)]
        # prefer alphanumeric starts
        filtered = [i for i in valid if re.match(r"^[a-zA-Z0-9].*", token_strs[i])]
        if not filtered:
            filtered = valid
        if not filtered:
            return []
        # cap and sample evenly
        if len(filtered) > self.max_masks:
            idxs = np.linspace(0, len(filtered) - 1, num=self.max_masks)
            chosen = [filtered[int(round(x))] for x in idxs]
            # unique and sorted
            chosen = sorted(list(dict.fromkeys(chosen)))
            return chosen
        return filtered

    # -------------------------
    # per-candidate scoring helpers (fallbacks)
    # -------------------------
    def _score_with_onnx(self, text: str) -> float:
        """Per-candidate ONNX scoring (single-candidate, masked per selected positions)."""
        toks = self.tokenizer(text, return_tensors="np", truncation=True, max_length=self.max_length)
        input_ids = toks["input_ids"]            # (1, L)
        attention_mask = toks["attention_mask"]  # (1, L)
        positions = self._select_positions(input_ids[0], attention_mask[0])
        if not positions:
            return float(-0.5 * len(text.split()))
        # build batch for this candidate (one row per position)
        batch = []
        batch_attn = []
        for pos in positions:
            row = input_ids[0].copy()
            row[pos] = self.tokenizer.mask_token_id
            batch.append(row)
            batch_attn.append(attention_mask[0])
        batch_ids = np.stack(batch, axis=0).astype(np.int64)
        batch_attn = np.stack(batch_attn, axis=0).astype(np.int64)
        ort_inputs = {
            self.onnx_sess.get_inputs()[0].name: batch_ids,
            self.onnx_sess.get_inputs()[1].name: batch_attn,
        }
        logits = self.onnx_sess.run(None, ort_inputs)[0]  # (B, L, V)
        rows = np.arange(len(positions))
        logits_pos = logits[rows, positions, :]  # (B, V)
        log_probs = _log_softmax_numpy(logits_pos)
        orig_tokens = np.array([input_ids[0, p] for p in positions], dtype=np.int64)
        picked = log_probs[np.arange(len(positions)), orig_tokens]
        return float(picked.sum())

    def _score_with_torch(self, text: str) -> float:
        """Per-candidate torch scoring (batched over positions for the candidate)."""
        import torch as _torch
        toks = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=self.max_length).to(self.device)
        input_ids = toks["input_ids"]
        attention_mask = toks["attention_mask"]
        seq = input_ids[0]
        seq_np = seq.cpu().numpy()
        attn_np = attention_mask[0].cpu().numpy()
        positions = self._select_positions(seq_np, attn_np)
        if not positions:
            return float(-0.5 * len(text.split()))
        B = len(positions)
        batch = seq.unsqueeze(0).repeat(B, 1).clone()
        for i, pos in enumerate(positions):
            batch[i, pos] = self.tokenizer.mask_token_id
        batch_attn = attention_mask.repeat(B, 1)
        with _torch.no_grad():
            out = self.torch_model(input_ids=batch, attention_mask=batch_attn).logits  # (B, L, V)
            rows = _torch.arange(B, device=out.device)
            cols = _torch.tensor(positions, device=out.device)
            logits_pos = out[rows, cols, :]  # (B, V)
            orig = seq.unsqueeze(0).repeat(B, 1)
            token_ids = orig[rows, cols]
            log_probs = _torch.nn.functional.log_softmax(logits_pos, dim=-1)
            picked = log_probs[rows, token_ids]
            return float(picked.sum().cpu().item())

    # -------------------------
    # batched-candidates ONNX scoring
    # -------------------------
    def _batch_score_with_onnx(self, candidates: List[str]) -> List[float]:
        """
        Batch all candidates (and their mask positions) into a single ONNX run and
        return a score per candidate.
        """
        # Tokenize each candidate (np)
        toks = [self.tokenizer(c, return_tensors="np", truncation=True, max_length=self.max_length) for c in candidates]
        input_ids_list = [t["input_ids"][0] for t in toks]          # list of (L,) arrays
        attn_list = [t["attention_mask"][0] for t in toks]

        # For each candidate compute mask positions (may be empty)
        cand_positions: List[List[int]] = []
        for ids, attn in zip(input_ids_list, attn_list):
            poss = self._select_positions(ids, attn)
            # _select_positions already caps to self.max_masks
            cand_positions.append(poss)

        # Build rows: one row per (candidate_idx, pos)
        row_arrays = []   # list of masked input_id rows
        row_attns = []    # list of attention masks for each row
        row_meta: List[Tuple[int,int]] = []  # (cand_idx, pos)
        for ci, (ids, attn, poss) in enumerate(zip(input_ids_list, attn_list, cand_positions)):
            for p in poss:
                row = ids.copy()
                row[p] = self.tokenizer.mask_token_id
                row_arrays.append(row)
                row_attns.append(attn.copy())
                row_meta.append((ci, p))

        # If there are no maskable tokens across all candidates, return simple heuristics
        if len(row_arrays) == 0:
            heur = []
            for c in candidates:
                if not c or not c.strip():
                    heur.append(float("-1e9"))
                    continue
                # quick shortcuts similar to earlier behavior
                if is_valid_email(c):
                    heur.append(float(1e9)); continue
                if re.fullmatch(r"[0-9,]+", c.strip()):
                    heur.append(float(1e8)); continue
                # generic heuristic
                sc = -0.5 * len(c.split())
                if re.search(r"[@.]", c):
                    sc += 5.0
                if re.search(r"\d", c):
                    sc += 2.0
                heur.append(float(sc))
            return heur

        batch_ids = np.stack(row_arrays, axis=0).astype(np.int64)   # (B, L)
        batch_attn = np.stack(row_attns, axis=0).astype(np.int64)   # (B, L)

        ort_inputs = {
            self.onnx_sess.get_inputs()[0].name: batch_ids,
            self.onnx_sess.get_inputs()[1].name: batch_attn,
        }

        # Run ONNX once
        logits = self.onnx_sess.run(None, ort_inputs)[0]  # (B, L, V)

        B = logits.shape[0]
        rows_idx = np.arange(B, dtype=np.int64)
        pos_idx = np.array([m[1] for m in row_meta], dtype=np.int64)

        # logits at masked positions: (B, V)
        logits_pos = logits[rows_idx, pos_idx, :]
        # log-softmax per row
        log_probs = _log_softmax_numpy(logits_pos)  # (B, V)

        # original token ids at those positions
        orig_token_ids = np.array([input_ids_list[m[0]][m[1]] for m in row_meta], dtype=np.int64)  # (B,)
        picked = log_probs[np.arange(B), orig_token_ids]  # (B,)

        # aggregate sums per candidate
        cand_scores = [0.0 for _ in candidates]
        for (ci, _), val in zip(row_meta, picked):
            cand_scores[ci] += float(val)

        # For candidates that had no mask positions, produce heuristic similarly to above
        for idx, poss in enumerate(cand_positions):
            if not poss:
                c = candidates[idx]
                if not c or not c.strip():
                    cand_scores[idx] = float("-1e9")
                    continue
                if is_valid_email(c):
                    cand_scores[idx] = float(1e9)
                    continue
                if re.fullmatch(r"[0-9,]+", c.strip()):
                    cand_scores[idx] = float(1e8)
                    continue
                sc = -0.5 * len(c.split())
                if re.search(r"[@.]", c):
                    sc += 5.0
                if re.search(r"\d", c):
                    sc += 2.0
                cand_scores[idx] = float(sc)

        return cand_scores

    # -------------------------
    # main public API: score() and choose_best()
    # -------------------------
    def score(self, sentences: List[str]) -> List[float]:
        """
        Score a list of sentences (candidates). Returns a list of floats (higher = better).
        Uses ONNX batched path if available; else uses torch fallback per-candidate.
        """
        # quick pre-short-circuit heuristics: keep ordering same as scoring output
        out_scores: List[float] = []
        # If ONNX available, use batched path (single call)
        if self.onnx_sess is not None:
            try:
                return self._batch_score_with_onnx(sentences)
            except Exception as e:
                # fallback to per-sentence scoring if batched fails
                # print warning but keep going
                print("WARN: batched ONNX scoring failed, falling back to per-candidate scoring:", e)

        # If ONNX not available or batch failed -> per-sentence torch or heuristic
        for s in sentences:
            txt = (s or "").strip()
            if not txt:
                out_scores.append(float("-1e9"))
                continue
            if is_valid_email(txt):
                out_scores.append(float(1e9)); continue
            if re.fullmatch(r"[0-9,]+", txt):
                out_scores.append(float(1e8)); continue
            # torch fallback
            if self.torch_model is not None:
                try:
                    out_scores.append(self._score_with_torch(txt))
                except Exception:
                    out_scores.append(float(-0.5 * len(txt.split())))
            else:
                sc = -0.5 * len(txt.split())
                if re.search(r"[@.]", txt):
                    sc += 5.0
                if re.search(r"\d", txt):
                    sc += 2.0
                out_scores.append(float(sc))
        return out_scores

    def choose_best(self, candidates: List[str]) -> str:
        if not candidates:
            return ""
        if len(candidates) == 1:
            return candidates[0]
        scores = self.score(candidates)
        best_idx = int(np.argmax(np.array(scores, dtype=float)))
        return candidates[best_idx]


# quick smoke test
if __name__ == "__main__":
    ranker = PseudoLikelihoodRanker(onnx_path=DEFAULT_ONNX_PATH)
    tests = [
        "john dot doe at gmail dot com",
        "john.doe@gmail.com",
        "call shivam jharwal",
        "nine nine six zero one two three",
    ]
    for t in tests:
        print(t, "->", ranker.score([t])[0])
    # batch test
    cand_list = ["john dot doe at gmail dot com", "john.doe@gmail.com", "call shivam jharwal"]
    print("Batch scores:", ranker.score(cand_list))
