#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
import random
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Sequence, Any

import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy as scipy_entropy

try:
    from openai import OpenAI  # pip install openai>=1.0.0
except Exception:
    OpenAI = None

# =====================
# Numerics
# =====================

EPS = 1e-12

def _safe_log(x: float) -> float:
    return math.log(max(float(x), EPS))

def shannon_entropy(probs: Dict[str, float]) -> float:
    if not probs: return 0.0
    arr = np.array(list(probs.values()), dtype=float)
    s = float(arr.sum())
    if s <= 0: return 0.0
    arr = arr / s
    return float(scipy_entropy(arr, base=np.e))

def top2_margin(probs: Dict[str, float]) -> float:
    if not probs: return 0.0
    vals = sorted(probs.values(), reverse=True)
    if len(vals) == 1: return vals[0]
    return float(vals[0] - vals[1])

def top2_log_gap_from_probs(probs: Dict[str, float]) -> float:
    if not probs: return 0.0
    vals = sorted(probs.values(), reverse=True)
    if len(vals) < 2:
        return float('inf') if vals and vals[0] > 0 else 0.0
    p1, p2 = max(vals[0], EPS), max(vals[1], EPS)
    return float(_safe_log(p1) - _safe_log(p2))

def top2_log_gap_with_residual(probs: Dict[str, float]) -> Tuple[float, float, float, float]:
    """Residual-aware μ: p2_eff = max(p2_seen, 1 - sum(top_k)); returns (mu_eff, p1, p2_eff, p_res)."""
    if not probs: return 0.0, 0.0, 0.0, 0.0
    vals = sorted(probs.values(), reverse=True)
    p1 = max(vals[0], EPS)
    p2_seen = max(vals[1], EPS) if len(vals) >= 2 else EPS
    s = sum(vals)
    p_res = max(0.0, 1.0 - s)
    p2_eff = max(p2_seen, p_res, EPS)
    mu_eff = float(_safe_log(p1) - _safe_log(p2_eff))
    return mu_eff, p1, p2_eff, p_res

def js_divergence_over_dicts(dlist: List[Dict[str, float]]) -> float:
    if len(dlist) < 2: return 0.0
    keys = set().union(*[set(d.keys()) for d in dlist])
    mats = []
    for d in dlist:
        vec = np.array([d.get(k, 0.0) for k in keys], dtype=float)
        s = float(vec.sum())
        vec = vec / s if s > 0 else np.ones(len(keys), dtype=float) / len(keys)
        mats.append(vec)
    js_vals = []
    for i in range(len(mats)):
        for j in range(i + 1, len(mats)):
            js = float(jensenshannon(mats[i], mats[j]) ** 2)
            js_vals.append(js)
    return float(sum(js_vals) / max(1, len(js_vals)))

# =====================
# Evidence parsing (only used by PMD path)
# =====================

_BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")

def _extract_bullets(lines: List[str]) -> List[str]:
    out: List[str] = []
    for ln in lines:
        m = _BULLET.match(ln)
        if m: out.append(_BULLET.sub("", ln).strip())
    return [x for x in out if x]

def _split_user_text(user_text: str) -> Tuple[str, List[str], str]:
    lines = user_text.splitlines()
    if not lines: return "", [], ""
    start = None
    for i, ln in enumerate(lines):
        if ln.strip().lower().startswith(("evidence:", "facts:", "context:", "examples:")):
            start = i; break
        if _BULLET.match(ln):
            start = i; break
    if start is None:
        return user_text.strip(), [], ""
    evid_lines = []; j = start
    while j < len(lines) and (lines[j].strip() and (_BULLET.match(lines[j]) or j == start)):
        evid_lines.append(lines[j]); j += 1
    header = "\n".join(lines[:start]).strip()
    footer = "\n".join(lines[j:]).strip()
    items = _extract_bullets(evid_lines)
    if not items:
        items = [ln.split(":", 1)[1].strip() for ln in evid_lines if ":" in ln]
    return header, items, footer

def _join_user_text(header: str, items: List[str], footer: str) -> str:
    parts: List[str] = []
    if header: parts.append(header)
    if items:
        parts.append("Evidence:")
        parts.extend([f"- {it}" for it in items])
    if footer: parts.append(footer)
    return "\n".join(parts).strip()

# =====================
# SMD variants (LITE + closed_book)
# =====================

_CAPITAL_SEQ = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b")
_YEAR = re.compile(r"\b(1|2)\d{3}\b")
_NUMBER = re.compile(r"\b\d+(?:\.\d+)?\b")
_QUOTED = re.compile(r"([“\"'])(.+?)\1")

def _extract_blocks(text: str) -> List[str]:
    lines = [ln for ln in text.splitlines() if ln.strip() != ""]
    return lines if len(lines) >= 2 else [text]

def permute_prompt_blocks(text: str, seed: int) -> str:
    rng = random.Random(seed)
    blocks = _extract_blocks(text)
    idx = list(range(len(blocks))); rng.shuffle(idx)
    return "\n".join([blocks[i] for i in idx])

def mask_entities_numbers(text: str, strength: float, rng: random.Random, mask_token: str = "[…]") -> str:
    def mask_matches(pattern: re.Pattern, s: str) -> str:
        def repl(m): return mask_token if rng.random() < strength else m.group(0)
        return pattern.sub(repl, s)
    s = text
    s = mask_matches(_QUOTED, s)
    s = mask_matches(_CAPITAL_SEQ, s)
    s = mask_matches(_YEAR, s)
    s = mask_matches(_NUMBER, s)
    return s

def make_variants_lite(text: str, m: int, seeds: Sequence[int]) -> List[str]:
    if len(seeds) < m: seeds = list(seeds) * ((m + len(seeds) - 1)//len(seeds))
    out = []
    for k in range(m):
        out.append(permute_prompt_blocks(text, seed=int(seeds[k])))
    return out

def make_skeletons_closed_book(text: str, m: int, seeds: Sequence[int], mask_levels: Sequence[float] = (0.25,0.35,0.5,0.65,0.8,0.9), mask_token: str = "[…]") -> List[str]:
    levels = list(mask_levels)
    if len(seeds) < m: seeds = list(seeds) * ((m + len(seeds) - 1)//len(seeds))
    if len(levels) < m:
        times = (m + len(levels) - 1)//len(levels)
        levels = (levels * times)[:m]
    out = []
    for k in range(m):
        rng = random.Random(int(seeds[k]))
        lvl = float(levels[k])
        masked = mask_entities_numbers(text, lvl, rng, mask_token=mask_token)
        out.append(permute_prompt_blocks(masked, seed=int(seeds[k])))
    return out

# =====================
# Chat backend
# =====================

class ChatBackend:
    def __init__(self, model: str, api_key: Optional[str] = None, request_timeout: float = 60.0,
                 system_prompt: str = "You are a precise assistant. Answer in one sentence that ends with a period.") -> None:
        if OpenAI is None:
            raise ImportError("Install `openai>=1.0.0` and set OPENAI_API_KEY.")
        self.client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY",""))
        self.model = model
        self.request_timeout = float(request_timeout)
        self.system_prompt = system_prompt

    def _messages(self, prompt_text: str, prefix: str) -> List[Dict[str, str]]:
        return [
            {"role":"system","content": self.system_prompt},
            {"role":"user","content": f"{prompt_text.strip()}\n\nAnswer:\n"},
            {"role":"assistant","content": prefix}
        ]

    def chat_chunk(self, prompt_text: str, prefix: str, max_tokens: int = 32, temperature: float = 0.0, top_logprobs: int = 5, **kwargs) -> Tuple[List[str], List[Dict[str,float]]]:
        msgs = self._messages(prompt_text, prefix)
        params = dict(model=self.model, messages=msgs, temperature=temperature, max_tokens=max_tokens, logprobs=True, top_logprobs=top_logprobs)
        params.update(kwargs)
        if "timeout" in params:
            params["request_timeout"] = params.pop("timeout")
        resp = self.client.chat.completions.create(**params)
        toks: List[str] = []; dists: List[Dict[str,float]] = []
        try:
            choice = resp.choices[0]
            content = getattr(choice, "logprobs", None)
            content = getattr(content, "content", None) if content is not None else None
            if content:
                for seg in content:
                    tok = getattr(seg, "token", None)
                    if tok is None and isinstance(seg, dict): tok = seg.get("token")
                    toks.append(tok or "")
                    top = getattr(seg, "top_logprobs", None)
                    if top is None and isinstance(seg, dict): top = seg.get("top_logprobs")
                    probs: Dict[str,float] = {}
                    if top:
                        for it in top:
                            tt = getattr(it, "token", None) or (it.get("token") if isinstance(it, dict) else None)
                            lp = getattr(it, "logprob", None) or (it.get("logprob") if isinstance(it, dict) else None)
                            if tt is None or lp is None: continue
                            probs[str(tt)] = float(math.exp(float(lp)))
                        Z = sum(probs.values())
                        if Z > 0:
                            for k in list(probs.keys()):
                                probs[k] /= Z
                    dists.append(probs)
            else:
                txt = choice.message.content or ""
                toks = [txt]; dists = [{}]
        except Exception:
            pass
        return toks, dists

    def chat_one_token(self, prompt_text: str, prefix: str, temperature: float = 0.0, top_logprobs: int = 5, **kwargs) -> Tuple[str, Dict[str,float]]:
        toks, dists = self.chat_chunk(prompt_text, prefix, max_tokens=1, temperature=temperature, top_logprobs=top_logprobs, **kwargs)
        tok = toks[0] if toks else ""
        dist = dists[0] if dists else {}
        if not dist and tok: dist = {tok: 1.0}
        if tok and tok not in dist:
            dist = {**dist, tok: max(1e-9, 1.0 - sum(dist.values()))}
        return tok, dist

# =====================
# Aggregation & whitespace
# =====================

def aggregate_expected_log_weighted(dists: List[Dict[str,float]], weights: Optional[List[float]] = None) -> Dict[str,float]:
    if not dists: return {}
    if weights is None: weights = [1.0]*len(dists)
    assert len(weights) == len(dists)
    tokens = set().union(*[d.keys() for d in dists])
    W = float(sum(weights))
    agg: Dict[str,float] = {}
    for t in tokens:
        s = 0.0
        for w, d in zip(weights, dists):
            p = d.get(t, EPS)
            s += float(w) * _safe_log(p)
        agg[t] = float(math.exp(s / max(W, EPS)))
    Z = sum(agg.values())
    if Z > 0:
        for k in list(agg.keys()):
            agg[k] /= Z
    return agg

def enforce_whitespace_compat(dist: Dict[str,float], prev_char: Optional[str]) -> Dict[str,float]:
    if not dist: return dist
    def ok(tok: str) -> bool:
        if not tok: return False
        if prev_char and prev_char.isalnum():
            return tok.startswith((" ", ".", ",", ":", ";", "(", "-", "—", "–"))
        return True
    out = {t:p for t,p in dist.items() if ok(t)}
    return out if out else dist

def select_argmax_whitespace_aware(dist: Dict[str,float], prev_char: Optional[str], prefer_token: Optional[str] = None) -> str:
    dist2 = enforce_whitespace_compat(dist, prev_char)
    if not dist2: return ""
    best_p = max(dist2.values())
    cands = [t for t,p in dist2.items() if abs(p - best_p) < 1e-12]
    if len(cands) == 1: return cands[0]
    if prefer_token and prefer_token in cands: return prefer_token
    want_space_lead = (prev_char is not None) and (prev_char not in ("", " ", "\n", "\t"))
    if want_space_lead:
        c2 = [t for t in cands if t.startswith(" ")]
        if c2: cands = c2
    else:
        c2 = [t for t in cands if not t.startswith(" ")]
        if c2: cands = c2
    return sorted(cands)[0]

# =====================
# Certificates
# =====================

def p_flip_bounded(mu: float, r: float) -> float:
    """Upper bound under δ ∈ [-2r, 2r] (Hoeffding-style); 0 if μ ≥ 2r."""
    if r <= 0: return 0.0
    if mu >= 2.0 * r: return 0.0
    return math.exp(- (mu * mu) / (8.0 * r * r))

def p_flip_gaussian(mu: float, r: float, sigma_over_r: float = 0.75) -> float:
    """Assume ε_i ~ N(0, σ²), indep.; δ ~ N(0, 2σ²); P(δ>μ) = 0.5 * erfc( μ / (2σ) )."""
    if r <= 0: return 0.0
    sigma = max(EPS, sigma_over_r * r)
    z = mu / (2.0 * sigma)
    # 0.5 * erfc(z) is numerically stable for moderate z
    return 0.5 * math.erfc(z)

# =====================
# Determinism decoder
# =====================

@dataclass
class LadderConfig:
    chunk_size: int = 48
    noise_radius: float = 0.02
    jsd_gate: float = 0.15
    sigma: float = 0.02
    alpha: float = 1e-4
    rho: float = 0.2
    max_k: int = 12
    smd_m: int = 4
    smd_mode: str = "lite"   # lite | closed_book
    smd_warmup: int = 5
    use_ecl: bool = False
    ecl_mode: str = "sort_lex"
    pitb_epsilon: float = 1e-4
    abstain_on_nonrobust: bool = False
    temperature: float = 0.0
    top_logprobs: int = 5
    max_sentences: int = 1
    force_period: bool = True
    lambda_base: float = 2.0
    use_residual: bool = True

    # New: certificate controls
    cert_policy: str = "det_then_stat"  # det_then_stat | stat_only | det_only
    alpha_cert: float = 0.05            # accept if P_flip ≤ alpha_cert
    noise_model: str = "bounded"        # bounded | gaussian
    sigma_over_r: float = 0.75          # for gaussian model

class DeterminismChatDecoder:
    def __init__(self, backend: ChatBackend, ladder: Optional[LadderConfig] = None, seed: int = 42) -> None:
        self.backend = backend
        self.cfg = ladder or LadderConfig()
        self.rng = random.Random(seed)
        self.calls_chat_chunks = 0
        self.calls_variants = 0
        self.robust_steps = 0
        self.abstained_steps = 0
        self.stage_hist: Dict[str,int] = {}
        self.emitted_tokens: List[str] = []
        self.ng_counts: Dict[Tuple[str,...], int] = {}
        self.sentences_emitted = 0

    def _bump_stage(self, name: str): self.stage_hist[name] = self.stage_hist.get(name, 0) + 1
    def _margin_threshold(self, K_eff: float = 1.0) -> float:
        return math.tanh(self.cfg.sigma * math.sqrt(max(0.0, math.log(1.0/self.cfg.alpha))) / max(1e-9, math.sqrt(K_eff)))
    def _assemble(self, header: str, items: List[str], footer: str) -> str:
        return _join_user_text(header, items, footer)

    def _update_ngram_guard(self, n: int = 8) -> bool:
        if len(self.emitted_tokens) < n: return False
        ng = tuple(self.emitted_tokens[-n:])
        c = self.ng_counts.get(ng, 0) + 1
        self.ng_counts[ng] = c
        return c >= 2

    def _maybe_bump_sentence(self, token: str, generated: str) -> bool:
        if any(ch in token for ch in (".","!","?","。","！","？")):
            m = re.findall(r"[\.!\?。！？](?:\s|$)", generated)
            new_count = len(m)
            if new_count > self.sentences_emitted:
                self.sentences_emitted = new_count
                if self.sentences_emitted >= max(0, int(self.cfg.max_sentences)):
                    return True
        return False

    # --- PMD (evidence) ---
    def _pmd_basis_or_es(self, header: str, items: List[str], footer: str, prefix: str) -> Tuple[str, Dict[str,float], str, Dict[str,Any], str]:
        dists: List[Dict[str,float]] = []; base_tok = None
        dedup = list(dict.fromkeys(items))
        bases = [dedup, list(reversed(dedup)), sorted(dedup, key=lambda s: s.lower()), sorted(dedup, key=lambda s: (len(s), s.lower()))]
        uniq=[]; seen=set()
        for arr in bases:
            key = tuple(arr)
            if key not in seen: uniq.append(arr); seen.add(key)
        for idx, order in enumerate(uniq):
            tok, p = self.backend.chat_one_token(self._assemble(header, order, footer), prefix, temperature=0.0, top_logprobs=self.cfg.top_logprobs)
            if idx == 0: base_tok = tok
            dists.append(p); self.calls_variants += 1
        jsd = js_divergence_over_dicts(dists)
        agg = self._aggregate_anchor_and_select(dists, base_dist=dists[0], prev_char=prefix[-1] if prefix else None, base_tok=base_tok)
        stage = "mb_pmd"
        if jsd > self.cfg.jsd_gate and len(uniq) < self.cfg.max_k:
            K = len(dists)
            while K < self.cfg.max_k:
                perm = dedup[:]; self.rng.shuffle(perm)
                tok, p = self.backend.chat_one_token(self._assemble(header, perm, footer), prefix, temperature=0.0, top_logprobs=self.cfg.top_logprobs)
                dists.append(p); self.calls_variants += 1; K += 1
                agg = self._aggregate_anchor_and_select(dists, base_dist=dists[0], prev_char=prefix[-1] if prefix else None, base_tok=base_tok)
                margin = top2_margin(agg)
                K_eff = K / max(1e-9, (1.0 + (K - 1) * self.cfg.rho))
                thr = self._margin_threshold(K_eff)
                if margin >= thr: break
            stage = "spmd_es"
        chosen = max(agg.items(), key=lambda kv: kv[1])[0] if agg else (base_tok or "")
        return chosen, agg, stage, {"K": len(dists), "jsd": jsd}, base_tok or ""

    # --- SMD (no-evidence) ---
    def _smd_basis_or_es(self, prompt_text: str, prefix: str) -> Tuple[str, Dict[str,float], str, Dict[str,Any], str]:
        base_tok_base, base_dist = self.backend.chat_one_token(prompt_text, prefix, temperature=0.0, top_logprobs=self.cfg.top_logprobs)
        self.calls_variants += 1
        seeds = list(range(max(1, self.cfg.smd_m - 1)))
        if self.cfg.smd_mode == "closed_book":
            variants = make_skeletons_closed_book(prompt_text, m=len(seeds), seeds=seeds)
        else:
            variants = make_variants_lite(prompt_text, m=len(seeds), seeds=seeds)
        dists = [base_dist]
        for v in variants:
            tok, p = self.backend.chat_one_token(v, prefix, temperature=0.0, top_logprobs=self.cfg.top_logprobs)
            dists.append(p); self.calls_variants += 1
        jsd = js_divergence_over_dicts(dists)
        agg = self._aggregate_anchor_and_select(dists, base_dist=base_dist, prev_char=prefix[-1] if prefix else None, base_tok=base_tok_base)
        stage = "smd_basis"
        if jsd > self.cfg.jsd_gate and len(dists) < self.cfg.max_k:
            K = len(dists)
            while K < self.cfg.max_k:
                if self.cfg.smd_mode == "closed_book":
                    v = make_skeletons_closed_book(prompt_text, m=1, seeds=[self.rng.randint(0,10_000)])[0]
                else:
                    v = make_variants_lite(prompt_text, m=1, seeds=[self.rng.randint(0,10_000)])[0]
                tok, p = self.backend.chat_one_token(v, prefix, temperature=0.0, top_logprobs=self.cfg.top_logprobs)
                dists.append(p); self.calls_variants += 1; K += 1
                agg = self._aggregate_anchor_and_select(dists, base_dist=base_dist, prev_char=prefix[-1] if prefix else None, base_tok=base_tok_base)
                margin = top2_margin(agg)
                K_eff = K / max(1e-9, (1.0 + (K - 1) * self.cfg.rho))
                thr = self._margin_threshold(K_eff)
                if margin >= thr: break
            stage = "smd_es"
        chosen = max(agg.items(), key=lambda kv: kv[1])[0] if agg else (base_tok_base or "")
        return chosen, agg, stage, {"K": len(dists), "jsd": jsd}, base_tok_base or ""

    def _aggregate_anchor_and_select(self, dists: List[Dict[str,float]], base_dist: Dict[str,float], prev_char: Optional[str], base_tok: Optional[str]) -> Dict[str,float]:
        weights = [self.cfg.lambda_base] + [1.0]*(len(dists)-1)
        agg = aggregate_expected_log_weighted(dists, weights=weights)
        allowed = set(base_dist.keys())
        if allowed:
            agg = {t:p for t,p in agg.items() if t in allowed}
            if not agg: agg = dict(base_dist)
        agg = enforce_whitespace_compat(agg, prev_char)
        return agg

    # --- Certificates ---
    def _p_flip(self, mu: float, r: float) -> float:
        model = self.cfg.noise_model.lower()
        if model == "gaussian":
            return p_flip_gaussian(mu, r, sigma_over_r=self.cfg.sigma_over_r)
        # default bounded
        return p_flip_bounded(mu, r)

    def _certify(self, mu: float, r: float) -> Tuple[bool, str, float]:
        """
        Return (accept, cert_kind, p_flip) where cert_kind ∈ {"det","stat",""}.
        Policy:
          det_only:     accept iff μ ≥ 2r
          stat_only:    accept iff p_flip ≤ α (no det check)
          det_then_stat (default): accept det if μ ≥ 2r else stat if p_flip ≤ α
        """
        policy = self.cfg.cert_policy
        # deterministic
        if policy in ("det_only", "det_then_stat"):
            if mu >= 2.0 * r:
                return True, "det", 0.0
        # statistical
        if policy in ("stat_only", "det_then_stat"):
            p = self._p_flip(mu, r)
            if p <= self.cfg.alpha_cert:
                return True, "stat", p
            return False, "stat", p
        return False, "", float("nan")

    # --- main loop ---
    def generate(self, user_text: str, max_tokens: int = 256, stop: Optional[str] = None) -> Tuple[str, List[Dict[str,Any]], Dict[str,Any]]:
        header, items, footer = _split_user_text(user_text)
        prompt_canon = self._assemble(header, items, footer)
        generated = ""
        trace: List[Dict[str,Any]] = []
        n_emitted = 0
        B = int(self.cfg.chunk_size)

        while n_emitted < max_tokens:
            toks, dists = self.backend.chat_chunk(prompt_canon, generated, max_tokens=min(B, max_tokens - n_emitted), temperature=self.cfg.temperature, top_logprobs=self.cfg.top_logprobs)
            self.calls_chat_chunks += 1
            if not toks: break

            for tok, dist in zip(toks, dists):
                # No logprobs: accept token; sentence stop & n-gram guard protect looping
                if not dist:
                    chosen = tok
                    generated += chosen
                    self.emitted_tokens.append(chosen)
                    n_emitted += 1
                    self._bump_stage("chunk_no_logprobs")
                    trace.append({"step": n_emitted, "stage": "chunk_no_logprobs", "token": chosen})
                    if stop and generated.endswith(stop): return self._finalize_text(generated), trace, self._summarize(n_emitted)
                    if self._maybe_bump_sentence(chosen, generated): return self._finalize_text(generated), trace, self._summarize(n_emitted)
                    if self._update_ngram_guard(): return self._finalize_text(generated), trace, self._summarize(n_emitted)
                    continue

                # Robustness with residual-aware μ
                if self.cfg.use_residual:
                    mu, p1, p2_eff, p_res = top2_log_gap_with_residual(dist)
                else:
                    mu = top2_log_gap_from_probs(dist); p1 = dist.get(max(dist, key=dist.get), 0.0); p2_eff = None; p_res = None

                accept, cert_kind, pflip = self._certify(mu, self.cfg.noise_radius)
                if accept:
                    chosen = tok  # preserve whitespace/style
                    generated += chosen
                    self.emitted_tokens.append(chosen)
                    n_emitted += 1
                    self.robust_steps += 1
                    stage = "robust_det" if cert_kind == "det" else "robust_stat"
                    self._bump_stage(stage)
                    trace.append({
                        "step": n_emitted, "stage": stage, "token": chosen,
                        "mu": round(mu, 4), "p_flip": (None if math.isnan(pflip) else round(pflip, 6)),
                        "p1": round(p1, 6), "p2_eff": (None if p2_eff is None else round(p2_eff, 6)),
                        "p_residual": (None if p_res is None else round(p_res, 6)),
                        "alpha_cert": self.cfg.alpha_cert, "noise_model": self.cfg.noise_model,
                    })
                    if stop and generated.endswith(stop): return self._finalize_text(generated), trace, self._summarize(n_emitted)
                    if self._maybe_bump_sentence(chosen, generated): return self._finalize_text(generated), trace, self._summarize(n_emitted)
                    if self._update_ngram_guard(): return self._finalize_text(generated), trace, self._summarize(n_emitted)
                    continue

                # Non-robust → warm-up / stabilize one token
                if not items and n_emitted < self.cfg.smd_warmup:
                    chosen = tok
                    generated += chosen
                    self.emitted_tokens.append(chosen)
                    n_emitted += 1
                    self._bump_stage("early_base")
                    trace.append({
                        "step": n_emitted, "stage": "early_base", "token": chosen,
                        "mu": round(mu, 4), "p_flip": round(pflip, 6) if math.isfinite(pflip) else None
                    })
                    if stop and generated.endswith(stop): return self._finalize_text(generated), trace, self._summarize(n_emitted)
                    if self._maybe_bump_sentence(chosen, generated): return self._finalize_text(generated), trace, self._summarize(n_emitted)
                    if self._update_ngram_guard(): return self._finalize_text(generated), trace, self._summarize(n_emitted)
                    continue

                if items:
                    chosen, agg, stage, meta, base_tok = self._pmd_basis_or_es(header, items, footer, prefix=generated)
                else:
                    chosen, agg, stage, meta, base_tok = self._smd_basis_or_es(prompt_canon, prefix=generated)

                if not chosen:
                    # fall back to base tok; otherwise PITB
                    if tok:
                        chosen = tok; agg = dist; stage = "fallback_base"; meta = {"K":1, "jsd":0.0}
                    else:
                        # PITB bias
                        fp = hashlib.sha256((prompt_canon or "").encode("utf-8")).hexdigest()
                        out = dict(dist)
                        for t in list(out.keys()):
                            h = hashlib.sha256(f"pitb_v5|{fp}|{t}".encode("utf-8")).digest()
                            u = int.from_bytes(h[:8], "big") / float(1<<64)
                            out[t] = math.exp(_safe_log(out[t]) + (2*u - 1)*self.cfg.pitb_epsilon)
                        Z = sum(out.values())
                        if Z>0:
                            for k in list(out.keys()):
                                out[k] /= Z
                        agg = enforce_whitespace_compat(out, generated[-1] if generated else None)
                        chosen = select_argmax_whitespace_aware(agg, generated[-1] if generated else None, prefer_token=None)
                        stage = "pitb"; meta = {"K":1, "jsd":0.0}

                generated += chosen
                self.emitted_tokens.append(chosen)
                n_emitted += 1
                self._bump_stage(stage)
                trace.append({
                    "step": n_emitted, "stage": stage, "token": chosen,
                    "entropy": round(shannon_entropy(agg), 4) if isinstance(agg, dict) else None,
                    "top2_margin": round(top2_margin(agg), 4) if isinstance(agg, dict) else None,
                    "jsd_perms": round(meta.get("jsd", 0.0), 4) if isinstance(meta, dict) else None,
                    "K_used": int(meta.get("K", 1)) if isinstance(meta, dict) else 1,
                })
                if stop and generated.endswith(stop): return self._finalize_text(generated), trace, self._summarize(n_emitted)
                if self._maybe_bump_sentence(chosen, generated): return self._finalize_text(generated), trace, self._summarize(n_emitted)
                if self._update_ngram_guard(): return self._finalize_text(generated), trace, self._summarize(n_emitted)
                break  # discard rest of chunk once we stabilize one token

        return self._finalize_text(generated), trace, self._summarize(n_emitted)

    def _finalize_text(self, text: str) -> str:
        if self.cfg.force_period and (not text.strip().endswith((".", "!", "?", "。", "！", "？"))):
            if text and (text[-1].isalnum() or text[-1] in (")", "]", "\"", "'")):
                return text + "."
        return text

    def _summarize(self, n_steps: int) -> Dict[str,Any]:
        extra = self.calls_variants
        overhead = (n_steps + extra) / max(1, n_steps) if n_steps > 0 else 1.0
        return {
            "api_calls_chat_chunks": self.calls_chat_chunks,
            "api_calls_variants": self.calls_variants,
            "overhead_vs_base": overhead,
            "robust_steps": self.robust_steps,
            "abstained_steps": self.abstained_steps,
            "stage_hist": dict(self.stage_hist),
        }

# =====================
# CLI
# =====================

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Chat-mode deterministic decoding with statistical RABeL certificates.")
    ap.add_argument("--prompt", type=str, required=True)
    ap.add_argument("--model", type=str, default="gpt-4o-mini")
    ap.add_argument("--api-key", type=str, default=os.getenv("OPENAI_API_KEY", ""))

    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--chunk-size", type=int, default=48)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-logprobs", type=int, default=5)
    ap.add_argument("--stop", type=str, default=None)

    ap.add_argument("--noise-radius", type=float, default=0.02, help="r in log-prob units for noise.")
    ap.add_argument("--sigma", type=float, default=0.02)
    ap.add_argument("--alpha", type=float, default=1e-4)
    ap.add_argument("--rho", type=float, default=0.2)
    ap.add_argument("--max-k", type=int, default=12)
    ap.add_argument("--jsd-gate", type=float, default=0.15)

    ap.add_argument("--smd-m", type=int, default=4)
    ap.add_argument("--smd-mode", type=str, default="lite", choices=["lite","closed_book"])
    ap.add_argument("--smd-warmup", type=int, default=5)
    ap.add_argument("--use-ecl", action="store_true")
    ap.add_argument("--ecl-mode", type=str, default="sort_lex", choices=["none","sort_lex","sort_len","digest"])
    ap.add_argument("--pitb-epsilon", type=float, default=1e-4)
    ap.add_argument("--abstain-on-nonrobust", action="store_true")
    ap.add_argument("--max-sentences", type=int, default=1)
    ap.add_argument("--force-period", action="store_true")
    ap.add_argument("--no-force-period", dest="force_period", action="store_false")
    ap.set_defaults(force_period=True)
    ap.add_argument("--lambda-base", type=float, default=2.0)
    ap.add_argument("--no-residual", dest="use_residual", action="store_false")
    ap.add_argument("--use-residual", dest="use_residual", action="store_true")
    ap.set_defaults(use_residual=True)

    # Statistical certificate flags
    ap.add_argument("--cert-policy", type=str, default="det_then_stat", choices=["det_then_stat","stat_only","det_only"])
    ap.add_argument("--alpha-cert", type=float, default=0.05, help="Accept if estimated flip probability ≤ alpha (e.g., 0.05, 0.01).")
    ap.add_argument("--noise-model", type=str, default="bounded", choices=["bounded","gaussian"])
    ap.add_argument("--sigma-over-r", type=float, default=0.75, help="For gaussian noise, σ = (sigma_over_r) * r.")

    return ap.parse_args()

def main():
    args = parse_args()
    backend = ChatBackend(model=args.model, api_key=args.api_key)

    ladder = LadderConfig(
        chunk_size=args.chunk_size,
        noise_radius=args.noise_radius,
        jsd_gate=args.jsd_gate,
        sigma=args.sigma, alpha=args.alpha, rho=args.rho,
        max_k=args.max_k, smd_m=args.smd_m, smd_mode=args.smd_mode, smd_warmup=args.smd_warmup,
        use_ecl=args.use_ecl, ecl_mode=args.ecl_mode,
        pitb_epsilon=args.pitb_epsilon,
        abstain_on_nonrobust=args.abstain_on_nonrobust,
        temperature=args.temperature, top_logprobs=args.top_logprobs,
        max_sentences=args.max_sentences,
        force_period=args.force_period,
        lambda_base=max(1.0, float(args.lambda_base)),
        use_residual=args.use_residual,
        cert_policy=args.cert_policy,
        alpha_cert=float(args.alpha_cert),
        noise_model=args.noise_model,
        sigma_over_r=float(args.sigma_over_r),
    )

    dec = DeterminismChatDecoder(backend, ladder=ladder)
    text, trace, stats = dec.generate(args.prompt, max_tokens=args.max_tokens, stop=args.stop)

    print("\n=== OUTPUT ===\n")
    print(text)
    print("\n=== TRACE (first 60 steps) ===\n")
    for row in trace[:60]:
        print(row)
    print("\n=== STATS ===\n")
    print(json.dumps(stats, indent=2))

if __name__ == "__main__":
    main()
