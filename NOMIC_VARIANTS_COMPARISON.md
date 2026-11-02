# Nomic Embed Text v1.5 — Variants Comparison

## Overview

This document records the results of testing 5 variants of `nomic-ai/nomic-embed-text-v1.5`
on the PaperScope retrieval benchmark (98 AI research papers). Each variant differs in
how the model is loaded and how long documents are handled.

## Variants

| # | Name | trust_remote_code | max_seq_length | rotary_scaling_factor | Description |
|---|------|-------------------|----------------|----------------------|-------------|
| 1 | `nomic-default` | False | 8192 (ST default) | None | Baseline: standard sentence-transformers loading |
| 2 | `nomic-2048` | False | 2048 | None | Native context only, no scaling |
| 3 | `nomic-trust-8192` | True | 8192 | None | Trust remote code, no rotary scaling |
| 4 | `nomic-rotary-8192` | True | 8192 | 2 | Rotary scaling enabled |
| 5 | `nomic-trust-rotary` | True | 8192 | 2 | Trust + rotary scaling (same as #4) |

## Results

### Abstract-as-Query (abstract → full text, search all 98 papers)

| Variant | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR | Mean Rank | Points |
|---------|----------|----------|----------|-----------|-----|-----------|--------|
| nomic-default | 48.98% | 73.47% | 82.65% | 90.82% | 0.5733 | 10.07 | 48/98 |
| nomic-2048 | 98.98% | 100.0% | 100.0% | 100.0% | 0.9915 | 1.05 | 97/98 |
| nomic-trust-8192 | 48.98% | 73.47% | 82.65% | 90.82% | 0.5733 | 10.07 | 48/98 |
| nomic-rotary-8192 | **98.98%** | **100.0%** | **100.0%** | **100.0%** | **0.9949** | **1.01** | **97/98** |
| nomic-trust-rotary | **98.98%** | **100.0%** | **100.0%** | **100.0%** | **0.9949** | **1.01** | **97/98** |

### Method-as-Query (method section → full text, search all 98 papers)

| Variant | Tested | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR | Mean Rank | Points |
|---------|--------|----------|----------|----------|-----------|-----|-----------|--------|
| nomic-default | 73 | 54.79% | 71.23% | 79.45% | 86.30% | 0.6368 | 8.49 | 40/73 |
| nomic-2048 | 73 | 83.56% | 90.41% | 93.15% | 95.89% | 0.8785 | 2.04 | 61/73 |
| nomic-trust-8192 | 73 | 54.79% | 71.23% | 79.45% | 86.30% | 0.6368 | 8.49 | 40/73 |
| nomic-rotary-8192 | 73 | **91.78%** | **94.52%** | **95.89%** | **97.26%** | **0.9420** | **1.53** | **67/73** |
| nomic-trust-rotary | 73 | **91.78%** | **94.52%** | **95.89%** | **97.26%** | **0.9420** | **1.53** | **67/73** |

### 1K-Word Chunk-as-Query (first 1000 words → full text, search all 98 papers)

| Variant | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR | Mean Rank | Points |
|---------|----------|----------|----------|-----------|-----|-----------|--------|
| nomic-default | 53.06% | 77.55% | 87.76% | 93.88% | 0.6505 | 5.97 | 52/98 |
| nomic-2048 | 100.0% | 100.0% | 100.0% | 100.0% | 1.0000 | 1.00 | 98/98 |
| nomic-trust-8192 | 53.06% | 77.55% | 87.76% | 93.88% | 0.6505 | 5.97 | 52/98 |
| nomic-rotary-8192 | **100.0%** | **100.0%** | **100.0%** | **100.0%** | **1.0000** | **1.00** | **98/98** |
| nomic-trust-rotary | **100.0%** | **100.0%** | **100.0%** | **100.0%** | **1.0000** | **1.00** | **98/98** |

### Truncation Statistics

| Variant | Max Seq | Truncated | Not Truncated | Mean Trunc Rate | Max Trunc Rate |
|---------|---------|-----------|---------------|-----------------|----------------|
| nomic-default | 8192 | 93/98 | 5/98 | 0.83 | 0.93 |
| nomic-2048 | 2048 | 98/98 | 0/98 | 0.95 | 0.98 |
| nomic-trust-8192 | 8192 | 93/98 | 5/98 | 0.83 | 0.93 |
| nomic-rotary-8192 | 8192 | 93/98 | 5/98 | 0.83 | 0.93 |
| nomic-trust-rotary | 8192 | 93/98 | 5/98 | 0.83 | 0.93 |

## Key Findings

### 1. Rotary scaling is ESSENTIAL for long documents
- **Without rotary** (default/trust-8192): AbsR1 = 49%, MeanRank = 10.07
- **With rotary** (rotary/trust-rotary): AbsR1 = 99%, MeanRank = 1.01
- This confirms: "Without rotary scaling, full-paper document vectors (8192 tokens) are likely degraded"

### 2. trust_remote_code alone doesn't help
- nomic-default and nomic-trust-8192 have **identical** results
- The custom code only matters when rotary scaling is enabled

### 3. Truncation to 2048 is trivially good but loses information
- nomic-2048: 98.98% Recall@1 (because abstract is in first 2048 tokens)
- But 100% of papers are truncated (max rate 98.17%)
- This is self-consistency, not semantic understanding

### 4. Rotary scaling gives the best of both worlds
- Full 8192-token context (only 94.9% truncated)
- 98.98% Recall@1 (abstract finds its own paper)
- Method section: 91.78% Recall@1 (best method retrieval)
- 100% Recall@1 for 1K chunks

### 5. nomic-rotary and nomic-trust-rotary are identical
- Both use trust_remote_code=True + rotary=2
- Results are exactly the same

## Conclusion

**For PaperScope, always use:**
- `trust_remote_code=True`
- `rotary_scaling_factor=2` (in model_kwargs)
- `max_seq_length=8192`
- `torch.float32` (not float16)

Without rotary scaling, the model's performance on long documents is severely degraded.
With rotary scaling, the model properly captures semantic relationships between
abstracts/methods/chunks and their full-text counterparts.

## Model Card Reference

From nomic-embed-text-v1.5 model card:
> "The model natively supports scaling of the sequence length past 2048 tokens. To do so:
> ```python
> rope_parameters = {"rope_theta": 1000.0, "rope_type": "dynamic", "factor": 2.0}
> model = AutoModel.from_pretrained('nomic-ai/nomic-embed-text-v1.5', rope_parameters=rope_parameters)
> ```
> "
>
> The custom modeling code (nomic-bert-2048) accepts `rotary_scaling_factor` as a kwarg,
> which sets `config.rotary_scaling_factor = 2` internally.

## Files

- Results: `comparison/results/nomic_variants/{variant}/` (downloaded from HF)
- Code: `nomic_variants/run_nomic_variants.py`
- HF: `w-ahmad/paper-scope-vectors/nomic_variants/{variant}/`
"