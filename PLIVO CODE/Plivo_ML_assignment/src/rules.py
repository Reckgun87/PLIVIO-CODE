# src/rules.py
"""
Stage 1: Rule-based candidate generator for ASR post-processing.

Features:
 - spoken email parsing: "alice at gmail dot com" -> "alice@gmail.com"
 - spoken digit expansions: "double nine" -> "99", "oh" -> "0"
 - Indian number grouping for rupees: "150000" -> "1,50,000" (when appropriate)
 - fuzzy name correction using difflib.get_close_matches (no extra deps)
 - minimal punctuation insertion (sentence-final).
 - candidate capping to bound downstream ranking cost.

This file is defensive: generate_candidates accepts either:
 - generate_candidates(text)
 - generate_candidates(text, candidate_cap=int)
 - generate_candidates(text, names_lex=list_of_names)
 - generate_candidates(text, names_lex, candidate_cap=int)
so it is compatible with pipelines that pass a names lexicon as the 2nd arg.
"""
from typing import List, Optional, Tuple
import re
import string
from difflib import get_close_matches
import json
import os

# --- Config / knobs ---
CANDIDATE_CAP = 2
MAX_NAME_MATCH_CUT = 0.8  # cutoff for difflib matches
FILLERS = {"um", "uh", "hmm", "erm", "ah", "ahh", "mm", "like", "you know"}

# --- Load lexicons (if present) ---
def _load_lines(path: str) -> List[str]:
    try:
        with open(path, "r", encoding="utf8") as f:
            return [ln.strip() for ln in f if ln.strip()]
    except Exception:
        return []

NAMES_LEX = _load_lines("data/names_lexicon.txt")

try:
    with open("data/misspell_map.json", "r", encoding="utf8") as f:
        MISSPELL_MAP = json.load(f)
except Exception:
    MISSPELL_MAP = {}

# --- Digit word map ---
DIGIT_WORDS = {
    "zero": "0", "oh": "0", "o": "0",
    "one": "1", "two": "2", "three": "3", "four": "4", "for": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9"
}

# Precompile regexes
DOUBLE_RE = re.compile(r"\bdouble\s+(" + "|".join(re.escape(w) for w in DIGIT_WORDS.keys()) + r")\b", re.I)
TRIPLE_RE = re.compile(r"\btriple\s+(" + "|".join(re.escape(w) for w in DIGIT_WORDS.keys()) + r")\b", re.I)
SPOKEN_EMAIL_TOKS = re.compile(r"\b([a-z0-9._%+-]+)\s+(at|@)\s+(.+?)\b", re.I)
SPOKEN_EMAIL_FULL = re.compile(
    r"\b([a-z0-9._%+-]+)\s+(at|@)\s+([a-z0-9.-]+)\s+(dot|\.)\s*([a-z]{2,3}(?:\.[a-z]{2,3})?)\b",
    re.I
)
# Simple email validator
EMAIL_VALID = re.compile(r"^[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}$", re.I)

# --- Utilities ---
def indian_grouping(num: int) -> str:
    """Group an integer using Indian grouping (e.g., 1234567 -> '12,34,567')."""
    s = str(int(num))
    if len(s) <= 3:
        return s
    last3 = s[-3:]
    rest = s[:-3]
    parts = []
    while len(rest) > 2:
        parts.append(rest[-2:])
        rest = rest[:-2]
    if rest:
        parts.append(rest)
    return ",".join(parts[::-1] + [last3])

def is_valid_email(s: str) -> bool:
    s = (s or "").strip().lower()
    return bool(EMAIL_VALID.match(s))

# --- Digit expansions ---
def _expand_double_triple(text: str) -> str:
    """Expand 'double nine' -> '99', 'triple two' -> '222'."""
    def _double_match(m):
        w = m.group(1).lower()
        return DIGIT_WORDS.get(w, w) * 2

    def _triple_match(m):
        w = m.group(1).lower()
        return DIGIT_WORDS.get(w, w) * 3

    text = DOUBLE_RE.sub(_double_match, text)
    text = TRIPLE_RE.sub(_triple_match, text)
    return text

def _map_digit_words(text: str) -> str:
    """Convert standalone digit words to digits when they appear as tokens."""
    tokens = text.split()
    out = []
    for t in tokens:
        stripped = t.strip(string.punctuation).lower()
        if stripped in DIGIT_WORDS:
            # Preserve surrounding punctuation if any
            # simple approach: replace the inner word with digit
            new = t
            # replace only the lowercase form to preserve case around
            idx = t.lower().find(stripped)
            if idx != -1:
                new = t[:idx] + DIGIT_WORDS[stripped] + t[idx+len(stripped):]
            out.append(new)
        else:
            out.append(t)
    return " ".join(out)

# --- Email parsing from spoken patterns ---
COMMON_DOMAINS = ["gmail.com", "yahoo.com", "hotmail.com", "outlook.com", "gmail.co.in"]

def parse_spoken_email(text: str) -> List[str]:
    """
    Conservative parsing of spoken emails. Returns list of plausible email strings.
    Examples handled:
     - 'john dot doe at gmail dot com' -> 'john.doe@gmail.com'
     - 'mary underscore ann at outlook dot com' -> 'mary_ann@outlook.com'
     - 'alice at gmail' -> 'alice@gmail.com' and 'alice@gmail.co.in' (domain completion)
    """
    if not text:
        return []
    t = text.lower()
    # normalize common spoken tokens
    t = t.replace(" underscore ", "_")
    t = t.replace(" dot ", " . ")
    t = t.replace(" at ", " @ ")
    t = t.replace(" dash ", "-")
    t = t.replace(" hyphen ", "-")
    # attempt strict pattern first
    m = SPOKEN_EMAIL_FULL.search(t)
    cands = []
    if m:
        local = m.group(1).replace(" ", "")
        domain = m.group(3).replace(" ", "")
        tld = m.group(5).replace(" ", "")
        email = f"{local}@{domain}.{tld}"
        cands.append(email)
        return cands

    # attempt permissive capture: local @ rest
    m2 = SPOKEN_EMAIL_TOKS.search(t)
    if m2:
        local = m2.group(1).replace(" ", "")
        # rest of string after match
        rest = t[m2.end():].strip()
        rest_clean = rest.replace(" . ", ".").replace(" dot ", ".").split()[0] if rest else ""
        if rest_clean and "." in rest_clean:
            cands.append(f"{local}@{rest_clean}")
        elif rest_clean:
            for d in COMMON_DOMAINS:
                if d.startswith(rest_clean):
                    cands.append(f"{local}@{d}")

    # fallback: replace ' at ' and ' dot ' and remove spaces if it looks like an email
    if not cands and ("@" in t or (" at " in text and " dot " in text)):
        collapsed = text.lower().replace(" at ", "@").replace(" dot ", ".").replace(" ", "")
        if "@" in collapsed and "." in collapsed:
            cands.append(collapsed)

    # dedupe and basic sanity
    seen = []
    for e in cands:
        e2 = e.lower()
        if e2 not in seen:
            seen.append(e2)
    return seen

# --- Name corrections (conservative) ---
def fuzzy_name_corrections(text: str, names_lex: Optional[List[str]] = None, max_candidates: int = 3) -> List[str]:
    """
    Use get_close_matches to propose slight corrections for tokens that look like names.
    If names_lex is provided, use it; otherwise fall back to global NAMES_LEX.
    Returns candidate variants (including the original text).
    """
    lex = names_lex if (names_lex and isinstance(names_lex, list)) else NAMES_LEX
    tokens = text.split()
    candidates = [text]
    if not lex:
        # no lexicon available: apply misspell_map only
        mapped = []
        changed = False
        for t in tokens:
            key = t.lower().strip(string.punctuation)
            if key in MISSPELL_MAP:
                mapped.append(MISSPELL_MAP[key])
                changed = True
            else:
                mapped.append(t)
        if changed:
            candidates.append(" ".join(mapped))
        return candidates

    # iterate tokens, try to replace tokens that look like names
    for i, t in enumerate(tokens):
        base = t.strip(string.punctuation)
        if not base:
            continue
        # skip filler-like or short tokens
        if base.lower() in FILLERS or len(base) <= 2:
            continue
        # direct misspell map first
        key = base.lower()
        if key in MISSPELL_MAP:
            new_tokens = tokens.copy()
            new_tokens[i] = MISSPELL_MAP[key]
            candidates.append(" ".join(new_tokens))
            continue
        # close matches from lexicon
        matches = get_close_matches(base, lex, n=1, cutoff=MAX_NAME_MATCH_CUT)
        if matches:
            new_tokens = tokens.copy()
            new_tokens[i] = matches[0]
            candidates.append(" ".join(new_tokens))
    # dedupe while preserving order
    uniq = []
    for c in candidates:
        if c not in uniq:
            uniq.append(c)
        if len(uniq) >= max_candidates:
            break
    return uniq

# --- Number detection + Indian formatting (minimal) ---
def detect_and_format_number(text: str) -> Optional[str]:
    """
    Find long digit sequences and format them in Indian grouping.
    If no digits found, returns None.
    """
    if not text:
        return None
    m = re.search(r"\b(\d{4,})\b", text.replace(",", ""))
    if m:
        raw = m.group(1)
        try:
            formatted = indian_grouping(int(raw))
            return text.replace(raw, formatted)
        except Exception:
            return None
    return None

# --- Candidate generation (top-level) with robust argument handling ---
def generate_candidates(transcript: str, second_arg=None, candidate_cap: int = CANDIDATE_CAP) -> List[str]:
    """
    Given a noisy transcript string, produce a short list of candidate corrections.

    second_arg may be:
      - None (default)
      - int -> interpreted as candidate_cap
      - list -> interpreted as names_lex

    This keeps the function compatible with callers that pass a names lexicon as
    the second positional argument (e.g., generate_candidates(text, names_lex)).
    """
    # Interpret second_arg
    names_lex = None
    if isinstance(second_arg, list):
        names_lex = second_arg
    elif isinstance(second_arg, int):
        candidate_cap = second_arg
    elif second_arg is None:
        pass
    else:
        # unknown type: try to be safe (if iterable of strings, treat as names_lex)
        try:
            if all(isinstance(x, str) for x in second_arg):
                names_lex = list(second_arg)
        except Exception:
            pass

    t = (transcript or "").strip()
    if not t:
        return [""]

    # 1) simple cleaning: remove filler words and excessive punctuation
    toks = re.split(r"(\s+|[,;:.!?()\"'])", t)
    filtered = []
    for tok in toks:
        if not tok or tok.isspace():
            filtered.append(tok)
            continue
        stripped = tok.strip(string.punctuation + " ")
        if stripped.lower() in FILLERS:
            # drop filler tokens entirely
            continue
        filtered.append(tok)
    cleaned = "".join(filtered).strip()

    # 2) expand digit phrases: double/triple -> repeated digits
    expanded = _expand_double_triple(cleaned)

    # 3) map spoken digit words to digits (only when they appear as standalone tokens)
    expanded = _map_digit_words(expanded)

    # Candidate list (start with the cleaned & expanded form)
    candidates = [expanded]

    # 4) email candidates (if spoken email patterns present)
    email_cands = parse_spoken_email(expanded)
    candidates.extend(email_cands)

    # 5) numeric formatting candidate
    num_fmt = detect_and_format_number(expanded)
    if num_fmt:
        candidates.append(num_fmt)

    # 6) name-based fuzzy corrections (pass names_lex if provided)
    name_cands = fuzzy_name_corrections(expanded, names_lex, max_candidates=3)
    candidates.extend(name_cands)

    # 7) punctuation candidate: ensure sentence-final punctuation if missing
    if not re.search(r"[.?!]$", expanded):
        first_lower = expanded.strip().lower()
        if first_lower.startswith(("do ", "did ", "can ", "could ", "would ", "is ", "are ", "shall ", "should ")):
            candidates.append(expanded.rstrip() + "?")
        else:
            candidates.append(expanded.rstrip() + ".")

    # 8) dedupe and cap
    uniq = []
    for c in candidates:
        if not c:
            continue
        c_strip = " ".join(c.split())  # normalize whitespace
        if c_strip not in uniq:
            uniq.append(c_strip)
        if len(uniq) >= candidate_cap:
            break

    return uniq

# --- Quick smoke tests when run as script ---
if __name__ == "__main__":
    tests = [
        "john dot doe at gmail dot com",
        "mary underscore ann at outlook dot com",
        "my number is nine nine six oh one two three",
        "my pin is two double zero five",
        "transfer rupees one lakh fifty thousand",
        "call shivam jharwal",
        "send to rohith k",
        "email is alice at gmail",
        "i paid fifty rupees",
        "what is the time",
        "can i pay online"
    ]
    for t in tests:
        print("IN :", t)
        print("CAND:", generate_candidates(t, None))
        # also test passing names lex
        print("CAND(with lex):", generate_candidates(t, NAMES_LEX))
        print("-" * 60)
