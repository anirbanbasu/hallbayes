# Statistical RABeL Certificates in Chat-Mode Deterministic Decoding

A Python implementation of **RABeL (Robustness-Aware Bias Elimination in Language models)** with statistical certificates for deterministic text generation. This system provides provable robustness guarantees for LLM outputs while maintaining high-quality generation.

## Overview

This implementation extends RABeL with **statistical certificates** that provide confidence bounds on the stability of generated tokens under adversarial perturbations. The system uses a multi-stage ladder approach:

1. **Deterministic Certificate**: If margin μ ≥ 2r, the token is provably robust
2. **Statistical Certificate**: Otherwise, estimate flip probability P(δ > μ) and accept if P_flip ≤ α
3. **Fallback Stabilization**: Use PMD (with evidence) or SMD-LITE (without evidence) for non-robust tokens

## Features

- **Dual Certificate System**: Both deterministic and statistical robustness guarantees
- **Residual-Aware Margins**: Accounts for unseen tokens in top-k distributions
- **Evidence-Based Generation**: Special handling for structured prompts with evidence/facts
- **Chunked Generation**: Efficient batch processing with dynamic fallback
- **Whitespace Compatibility**: Ensures proper token boundaries and formatting
- **N-gram Repeat Prevention**: Guards against infinite loops
- **Sentence-Level Control**: Configurable stopping at sentence boundaries

## Installation

### Requirements

```bash
pip install openai>=1.0.0 numpy scipy
```

### Environment Setup

Set your OpenAI API key:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

## Usage

### Basic Command Line Usage

```bash
python stable_decode_chat.py \
    --prompt "Explain quantum computing" \
    --model gpt-4o-mini \
    --max-tokens 64
```

### Advanced Configuration

```bash
python stable_decode_chat.py \
    --prompt "What are the benefits of renewable energy?" \
    --model gpt-4o-mini \
    --max-tokens 128 \
    --noise-radius 0.02 \
    --cert-policy det_then_stat \
    --alpha-cert 0.05 \
    --noise-model bounded \
    --chunk-size 48 \
    --max-sentences 2
```

### Evidence-Based Generation

Format your prompt with evidence bullets:
```bash
python stable_decode_chat.py \
    --prompt "Evidence:
- Solar panels convert sunlight to electricity
- Wind turbines generate power from wind
- Both are renewable energy sources
What is the main advantage of these technologies?" \
    --model gpt-4o-mini
```

## Key Parameters

### Certificate Configuration
- `--cert-policy`: Certificate strategy (`det_then_stat`, `stat_only`, `det_only`)
- `--alpha-cert`: Statistical acceptance threshold (default: 0.05 for 95% confidence)
- `--noise-model`: Noise assumption (`bounded` or `gaussian`)
- `--noise-radius`: Perturbation radius r in log-prob units (default: 0.02)
- `--sigma-over-r`: Gaussian noise parameter σ = sigma_over_r × r (default: 0.75)

### Generation Control
- `--max-tokens`: Maximum tokens to generate (default: 64)
- `--chunk-size`: Tokens per API call (default: 48)
- `--max-sentences`: Stop after N sentences (default: 1)
- `--force-period`: Add period if missing (default: True)
- `--temperature`: Sampling temperature (default: 0.0)
- `--top-logprobs`: Number of top tokens with probabilities (default: 5)

### Robustness Parameters
- `--sigma`: Margin threshold parameter (default: 0.02)
- `--alpha`: Significance level for margin testing (default: 1e-4)
- `--rho`: Correlation discount factor (default: 0.2)
- `--max-k`: Maximum permutation variants (default: 12)
- `--jsd-gate`: JSD threshold for triggering stabilization (default: 0.15)

### SMD Configuration (for non-evidence prompts)
- `--smd-m`: Initial variants for SMD (default: 4)
- `--smd-mode`: Variant generation method (`lite` or `closed_book`)
- `--smd-warmup`: Skip stabilization for first N tokens (default: 5)
- `--lambda-base`: Weight for base distribution (default: 2.0)
- `--use-residual`: Use residual-aware margins (default: True)

## Certificate Models

### Bounded Noise Model (Default)
Assumes perturbations δ ∈ [-2r, 2r] with Hoeffding-style bound:
```
P_flip ≤ exp(-μ²/(8r²)) when μ < 2r
P_flip = 0 when μ ≥ 2r (deterministic)
```

### Gaussian Noise Model
Assumes ε_i ~ N(0, σ²) independently:
```
P_flip = 0.5 × erfc(μ/(2σ))
where σ = sigma_over_r × r
```

## Output Format

The script provides three types of output:

1. **Generated Text**: The final stabilized output
2. **Trace**: Token-by-token generation details including:
   - Stage (robust_det, robust_stat, smd_basis, etc.)
   - Token value
   - Margin μ
   - Flip probability
   - Distribution statistics
3. **Statistics**: Summary metrics including:
   - API call counts
   - Overhead vs baseline
   - Robust steps count
   - Stage histogram

## Example Output

```json
{
  "step": 5,
  "stage": "robust_stat",
  "token": " energy",
  "mu": 1.234,
  "p_flip": 0.0234,
  "p1": 0.876,
  "p2_eff": 0.234,
  "p_residual": 0.123,
  "alpha_cert": 0.05,
  "noise_model": "bounded"
}
```

## Architecture Components

### Core Modules
- **ChatBackend**: Handles OpenAI API interactions
- **DeterminismChatDecoder**: Main decoder with certificate logic
- **LadderConfig**: Configuration dataclass
- **Certificate Functions**: `p_flip_bounded()`, `p_flip_gaussian()`

### Stabilization Methods
- **PMD (Permutation-based Marginal Debiasing)**: For evidence-based prompts
- **SMD-LITE (Semantic Marginal Debiasing)**: For open-ended prompts
- **PITB (Pseudo-random Implicit Tie-Breaking)**: Final fallback

### Utilities
- Evidence parsing and bullet extraction
- Whitespace compatibility enforcement
- N-gram repeat detection
- Entropy and margin calculations
- Jensen-Shannon divergence computation

## Research Background

This implementation is based on research in robust language model decoding, providing theoretical guarantees on output stability under adversarial perturbations. The statistical certificates extend deterministic guarantees to provide probabilistic bounds when strict determinism is not achievable.

## Limitations

- Requires OpenAI API access and usage costs
- Statistical certificates assume specific noise models
- Performance overhead increases with stricter robustness requirements
- Limited to models that provide log probabilities

## License

This implementation is provided for research and educational purposes. Please refer to the original research papers for academic citations.

## Troubleshooting

### Common Issues

1. **No API Key**: Ensure `OPENAI_API_KEY` is set in environment
2. **Import Errors**: Install required packages: `pip install openai numpy scipy`
3. **Rate Limits**: Reduce `--chunk-size` or add delays between requests
4. **Memory Issues**: Reduce `--max-k` or `--smd-m` parameters

### Debug Mode

For detailed debugging, examine the trace output which shows step-by-step decision making:
```bash
python stable_decode_chat.py --prompt "test" --max-tokens 10 2>&1 | tee debug.log
```

## Contributing

Contributions are welcome! Areas for improvement:
- Additional noise models
- Optimized variant generation
- Extended certificate types
- Performance optimizations
- Additional language model backends
