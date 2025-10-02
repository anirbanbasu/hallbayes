# DRKG Hallucination Assessment System

A safety-critical system for evaluating medical claims using the Drug Repurposing Knowledge Graph (DRKG) with hallucination risk assessment.

## Overview

This system combines:
- **DRKG**: A comprehensive biomedical knowledge graph with drug-gene-disease relationships
- **Entity Linking**: Maps drug and disease names to DRKG entities using DrugBank vocabulary
- **Evidence Collection**: Finds multi-hop paths between entities using PyTorch Geometric
- **Hallucination Assessment**: Uses Information Shift Ratio (ISR) to detect when models hallucinate vs. properly use evidence

## Installation

### 1. Download DRKG Dataset

```bash
# Download and extract DRKG (~58MB)
wget https://dgl-data.s3-us-west-2.amazonaws.com/dataset/DRKG/drkg.tar.gz
tar -xzf drkg.tar.gz
mkdir -p ./data/drkg
mv drkg.tsv ./data/drkg/
rm -rf drkg.tar.gz drkg/
```

### 2. Install Dependencies

```bash
pip install torch torch-geometric pandas numpy rapidfuzz lxml openai transformers
```

### 3. Required Files

Place these files in the same directory and ensure the `hallbayes` package is installed (e.g. `pip install ..` from the repo root):
- `drkg_hallbayes.py` - Main script
- `drkg_link_and_collect.py` - DRKG processing with PyTorch Geometric

### 4. DrugBank Vocabulary (Optional but Recommended)

1. Register for free academic account at [DrugBank](https://go.drugbank.com/releases/latest)
2. Download "DrugBank Vocabulary" CSV
3. Save as `./drugbank-vocabulary.csv`

### 5. Set OpenAI API Key

```bash
export OPENAI_API_KEY="your-api-key-here"
```

After downloading DRKG, run a simple query:

```bash
python drkg_hallbayes.py \
  --prompt "does ibuprofen treat headaches?" \
  --max-hops 2 \
  --isr-threshold 0.1 \
  --h-star 0.2
```

## How It Works

### 1. Entity Linking
The system identifies medical entities in your query:
- **Drugs**: Mapped to DrugBank IDs (e.g., "ibuprofen" → DB01050)
- **Diseases**: Mapped to MeSH or DOID terms (e.g., "headache" → MESH:D006261)
- **Genes**: Mapped to NCBI Gene IDs

### 2. Evidence Collection
Searches DRKG for paths connecting the entities:
- Uses k-hop subgraph extraction for efficiency
- Scores paths based on relation types and graph structure
- Collects top evidence facts

### 3. Hallucination Assessment
Evaluates if the model can safely answer using ISR methodology:
- **ISR (Information Shift Ratio)**: Measures if evidence changes model's answer
- **B2T (Bits to Trust)**: Information needed for confidence
- **RoH (Risk of Hallucination)**: Upper bound on hallucination probability

### 4. Decision Logic
- **ANSWER**: If ISR > threshold AND sufficient evidence
- **REFUSE**: If evidence weak or model ignoring evidence

## Command Line Options

```bash
python drkg_hallbayes.py \
  --prompt "medical question"           # Required: Your query
  --max-hops 3                          # Graph traversal depth (default: 3)
  --isr-threshold 0.5                   # ISR threshold for answering (default: 1.0)
  --h-star 0.1                          # Target hallucination rate (default: 0.05)
  --max-evidence 20                     # Max evidence facts (default: 20)
  --beam-width 500                      # Beam search width (default: 500)
  --model gpt-4o-mini                   # OpenAI model (default: gpt-4o-mini)
```

## Recommended Thresholds

### For well-established treatments
```bash
--isr-threshold 0.1 --h-star 0.2
```

### For drug mechanisms
```bash
--isr-threshold 0.3 --h-star 0.15
```

### For side effects/safety
```bash
--isr-threshold 0.5 --h-star 0.05
```

### For exploratory/off-label uses
```bash
--isr-threshold 0.8 --h-star 0.05
```

## Example Queries

### Simple treatment questions
```bash
python drkg_hallbayes.py \
  --prompt "does metformin treat diabetes?" \
  --max-hops 2 --isr-threshold 0.1 --h-star 0.2

python drkg_hallbayes.py \
  --prompt "can aspirin prevent heart attacks?" \
  --max-hops 3 --isr-threshold 0.2 --h-star 0.15
```

### Drug mechanism questions
```bash
python drkg_hallbayes.py \
  --prompt "how does tamoxifen work against breast cancer?" \
  --max-hops 4 --isr-threshold 0.3 --h-star 0.15

python drkg_hallbayes.py \
  --prompt "how does allopurinol reduce uric acid?" \
  --max-hops 3 --isr-threshold 0.3 --h-star 0.15
```

### Drug interaction queries
```bash
python drkg_hallbayes.py \
  --prompt "does warfarin interact with aspirin?" \
  --max-hops 2 --isr-threshold 0.3 --h-star 0.1
```

### Side effect queries
```bash
python drkg_hallbayes.py \
  --prompt "can prednisone cause weight gain?" \
  --max-hops 3 --isr-threshold 0.5 --h-star 0.1
```

## Understanding the Output

### Safety Metrics
- **Δ̄ (Delta-bar)**: Information gain from evidence (higher = more informative)
- **B2T**: Bits needed to trust the answer (lower = easier to trust)
- **ISR**: Information Shift Ratio (higher = evidence changes answer more)
- **RoH bound**: Upper bound on hallucination risk (lower = safer)

### Decision Outcomes
- **ANSWER**: Model can reliably answer using evidence
- **REFUSE**: Evidence insufficient or model not using it properly

### Evidence Quality
The system shows found paths, e.g.:
```
Ibuprofen → PTGS1 → Pain
```
This indicates ibuprofen affects PTGS1 (COX-1) gene, which relates to pain.

## GNBR Relation Codes

Common relation types in evidence:
- `treats`: Direct therapeutic relationship
- `GNBR::T`: Treats/therapeutic
- `GNBR::N`: Inhibits
- `GNBR::E`: Affects expression
- `GNBR::B`: Binds
- `GNBR::J`: Associated with disease

## Troubleshooting

### "No entities recognized"
- Use specific drug/disease names
- Check spelling (e.g., "acetaminophen" not "acetominophen")
- Try including DrugBank IDs: "DB01050 (ibuprofen)"

### "No paths found"
- Increase `--max-hops` (try 4 or 5)
- The relationship might not exist in DRKG
- Try related terms (e.g., "pain" instead of "headache")

### "REFUSE" despite good evidence
- Lower `--isr-threshold` for well-known facts
- Increase `--h-star` to be less conservative
- Model might already know the answer (ISR ≈ 0)

### OpenAI errors
- Check `OPENAI_API_KEY` is set
- Verify API key is valid
- Check rate limits

## Limitations

- **Coverage**: DRKG doesn't contain all drug-disease relationships
- **Currency**: DRKG data may not include latest discoveries
- **Entity Linking**: Some drug names might not be recognized without DrugBank vocabulary
- **Conservative**: System errs on side of safety, may refuse valid queries

## Citation

If you use this system, please cite:
- DRKG: [Ioannidis et al., 2020](https://github.com/gnn4dr/DRKG)
- DrugBank: [Wishart et al., 2018](https://www.drugbank.ca)

## License
MIT License - See individual component licenses for dependencies.
