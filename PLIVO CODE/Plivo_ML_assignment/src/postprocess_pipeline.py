# src/postprocess_pipeline.py
import json
import time
import re
from typing import List, Optional

from .rules import generate_candidates, is_valid_email
from .ranker_onnx import PseudoLikelihoodRanker

DIGIT_RE = re.compile(r"^[0-9,]+$")

class PostProcessor:
    def __init__(self,
                 names_lex_path: str,
                 onnx_model_path: Optional[str] = None,
                 device: str = "cpu",
                 max_length: int = 64,
                 candidate_cap: int = 2):
        # load names lexicon
        try:
            with open(names_lex_path, 'r', encoding='utf-8') as f:
                self.names_lex = [x.strip() for x in f if x.strip()]
        except Exception:
            self.names_lex = []

        # instantiate ranker (tries ONNX first, then torch fallback)
        try:
            self.ranker = PseudoLikelihoodRanker(onnx_path=onnx_model_path, device=device, max_length=max_length)
        except Exception:
            # fallback attempt without explicit onnx path
            self.ranker = PseudoLikelihoodRanker(onnx_path=None, device=device, max_length=max_length)

        # warm up ranker once to reduce first-call latency
        try:
            _ = self.ranker.score(["hello world"])
        except Exception:
            pass

        self.candidate_cap = candidate_cap

    def _short_circuit_pick(self, candidates: List[str]) -> Optional[str]:
        """
        If any candidate is a high-confidence email or digit-only string, pick it and skip ranking.
        Returns chosen candidate or None.
        """
        for c in candidates:
            if not c or not c.strip():
                continue
            if is_valid_email(c):
                return c
            if DIGIT_RE.fullmatch(c.strip()):
                return c
        return None

    def _punctuate(self, text: str) -> str:
        """Minimal sentence-final punctuation heuristic (keeps your original behavior)."""
        if not text or not text.strip():
            return text
        best = text
        lower = best.lower().strip()
        if not lower.endswith(('?', '.', ',')):
            first = lower.split()[0] if lower.split() else ""
            if first in ('can','shall','will','could','would','is','are','do','does','did','should','hey','hello'):
                best = best.rstrip() + '?'
            else:
                best = best.rstrip() + '.'
        return best

    def process_one(self, text: str) -> str:
        # generate a small set of candidates (cap to candidate_cap)
        cands = generate_candidates(text, self.names_lex, candidate_cap=self.candidate_cap)

        # 1) short-circuit: if rules produced an obvious email/number, pick it
        pick = self._short_circuit_pick(cands)
        if pick:
            return self._punctuate(pick)

        # 2) if only one candidate, return it
        if len(cands) == 1:
            return self._punctuate(cands[0])

        # 3) otherwise use vectorized ranker scoring
        best = cands[0]
        try:
            scores = self.ranker.score(cands)  # vectorized scoring: returns list[float]
            best_idx = int(max(range(len(scores)), key=lambda i: scores[i]))
            best = cands[best_idx]
        except Exception:
            # if ranking fails, gracefully fall back to first candidate
            best = cands[0]

        return self._punctuate(best)


def run_file(input_path: str,
             output_path: str,
             names_lex_path: str,
             onnx_model_path: Optional[str] = None,
             device: str = "cpu",
             max_length: int = 64,
             candidate_cap: int = 3):
    """
    Process a JSONL input file with objects containing at least `id` and `text` keys.
    Writes a JSONL with keys `id` and `text` (corrected).
    """
    pp = PostProcessor(names_lex_path, onnx_model_path=onnx_model_path, device=device,
                       max_length=max_length, candidate_cap=candidate_cap)

    with open(input_path, 'r', encoding='utf-8') as fin:
        rows = [json.loads(line) for line in fin if line.strip()]

    out = []
    for r in rows:
        src_text = r.get("text") or r.get("noisy") or r.get("transcript") or ""
        corrected = pp.process_one(src_text)
        out.append({"id": r.get("id", ""), "text": corrected})

    with open(output_path, 'w', encoding='utf-8') as f:
        for o in out:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
