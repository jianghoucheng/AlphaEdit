# AlphaEdit Failed Edits Analysis - Quick Start Guide

## Overview

This guide helps you analyze failed edits in your AlphaEdit runs, understand why they failed, and provides actionable recommendations for improvement.

## Tools Available

### 1. **analyze_failed_edits.py** - Quick Summary Analysis
Provides high-level statistics about edit success/failure rates.

**Usage:**
```bash
python3 analyze_failed_edits.py \
    --run-dir results/AlphaEdit/run_050 \
    --export failed_edits_run050.json \
    --create-retry-dataset retry_dataset_run050.json \
    --show-details 20
```

**What it does:**
- Counts total successes/failures
- Identifies failure categories
- Exports failed edits to JSON
- Creates a retry dataset for re-running failed edits

### 2. **diagnose_edit_failures.py** - Deep Diagnostic Analysis
Provides detailed diagnosis of why edits failed and generates recommendations.

**Usage:**
```bash
python3 diagnose_edit_failures.py \
    --run-dir results/AlphaEdit/run_050 \
    --output-report diagnostic_report_run050.json \
    --output-recommendations recommendations_run050.json
```

**What it does:**
- Analyzes probability distributions for each edit
- Identifies specific failure patterns (weak edit, poor generalization, etc.)
- Generates parameter tuning recommendations
- Creates both JSON and human-readable reports

### 3. **examine_edit.py** - Individual Edit Inspector
Examines specific edits in detail with visual explanations.

**Usage:**
```bash
# Examine a specific case by ID
python3 examine_edit.py --run-dir results/AlphaEdit/run_050 --case-id 0

# Examine a specific file
python3 examine_edit.py --case-file results/AlphaEdit/run_050/10_edits-case_0.json

# Compare multiple cases
python3 examine_edit.py --compare \
    results/AlphaEdit/run_050/10_edits-case_0.json \
    results/AlphaEdit/run_050/10_edits-case_1.json \
    results/AlphaEdit/run_050/10_edits-case_2.json
```

**What it does:**
- Shows detailed probabilities for rewrite, paraphrase, and neighborhood prompts
- Displays generated text samples
- Provides case-specific recommendations
- Identifies overgeneralization issues

## Understanding Failure Modes

### 1. **Paraphrase Generalization Failed** (Most Common - ~44%)
- **Symptom:** Edit works on exact prompt but fails on paraphrases
- **Cause:** Edit is too narrow and specific
- **Solution:** 
  - Increase `mom2_update_weight` (15000 → 20000-25000)
  - Reduce `kl_factor` (0.0625 → 0.03-0.04)
  - Increase `v_num_grad_steps` (25 → 50)

### 2. **Weak Edit** (~44%)
- **Symptom:** Edit succeeds but target_new probability is low
- **Cause:** Edit strength is insufficient
- **Solution:**
  - Increase `v_lr` (0.1 → 0.5 or 1.0)
  - Increase `v_num_grad_steps` (25 → 50)

### 3. **Rewrite Failure** (~1%)
- **Symptom:** Edit fails even on original prompt
- **Cause:** Edit is too weak or target change is unrealistic
- **Solution:**
  - Significantly increase `v_lr` (0.1 → 1.0)
  - Increase `v_num_grad_steps` (25 → 100)
  - Verify prompt template and target are reasonable

### 4. **Overgeneralization** (~4%)
- **Symptom:** Edit affects unrelated facts (high neighborhood changes)
- **Cause:** Edit is too broad
- **Solution:**
  - Decrease `mom2_update_weight` (15000 → 10000)
  - Reduce `clamp_norm_factor` (0.75 → 0.5)
  - Increase `nullspace_threshold` (0.02 → 0.03-0.05)

## Key Metrics Explained

### Success Rates
- **Rewrite Success Rate:** Does the edit work on the original prompt?
- **Paraphrase Success Rate:** Does it work on paraphrased versions?
- **Neighborhood Preservation:** Are unrelated facts preserved?

### Probabilities
- **P(target_new):** Model's confidence in the new target
- **P(target_true):** Model's confidence in the original target
- **Ideal:** P(target_new) >> P(target_true) (much larger)

## Parameter Tuning Guide

### Edit Strength Parameters

| Parameter | Default | Increase When | Decrease When |
|-----------|---------|---------------|---------------|
| `v_lr` | 0.1 | Rewrite fails, weak edits | Overgeneralization, incoherence |
| `v_num_grad_steps` | 25 | Weak edits, poor generalization | Overfitting, too slow |
| `v_weight_decay` | 0.5 | Model becomes unstable | Edits too weak |

### Edit Scope Parameters

| Parameter | Default | Increase When | Decrease When |
|-----------|---------|---------------|---------------|
| `mom2_update_weight` | 15000 | Poor generalization | Overgeneralization |
| `kl_factor` | 0.0625 | Too much model drift | Edit too constrained |
| `clamp_norm_factor` | 0.75 | Need stronger edits | Overgeneralization |

### Model Architecture Parameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| `layers` | [4,5,6,7,8] | Earlier layers = broader, later = specific |
| `nullspace_threshold` | 0.02 | Higher = more constrained edit |
| `L2` | 10 | Regularization strength |

## Workflow for Improving Results

### Step 1: Initial Analysis
```bash
python3 analyze_failed_edits.py --run-dir results/AlphaEdit/run_XXX --show-details 20
```
This gives you the overall picture.

### Step 2: Deep Diagnosis
```bash
python3 diagnose_edit_failures.py --run-dir results/AlphaEdit/run_XXX
cat recommendations_runXXX.txt
```
Read the recommendations carefully.

### Step 3: Examine Specific Cases
```bash
python3 examine_edit.py --run-dir results/AlphaEdit/run_XXX --case-id <ID>
```
Pick a few representative failures and understand them deeply.

### Step 4: Adjust Parameters
Based on the recommendations, create a new hparams file with adjusted parameters.

### Step 5: Re-run
```bash
python experiments/evaluate.py \
    --alg_name AlphaEdit \
    --model_name <model> \
    --hparams_fname <new_hparams>.json \
    --ds_name mcf \
    --dataset_size_limit 2000 \
    --num_edits 10
```

### Step 6: Compare Results
Compare the new run with the old one to see if improvements worked.

## Common Patterns by Relation Type

Based on analysis of run_050, these relations had the most failures:

1. **P413** (Position played) - 61 failures
2. **P103** (Native language) - 54 failures  
3. **P1412** (Languages spoken) - 50 failures
4. **P136** (Genre) - 47 failures
5. **P30** (Continent) - 47 failures

For these challenging relations, you may need:
- Higher `v_lr` (0.5-1.0)
- More `v_num_grad_steps` (50-100)
- Higher `mom2_update_weight` (20000-25000)

## Understanding the Outputs

### failed_edits_runXXX.json
Contains all failed edits with their metrics. Use for batch processing or analysis.

### retry_dataset_runXXX.json  
Ready-to-use dataset containing only failed edits. Can be used to re-run just the failures.

### diagnostic_report_runXXX.json
Detailed diagnostic information for each edit, including probability analysis and specific failure reasons.

### recommendations_runXXX.json / .txt
Actionable recommendations for parameter tuning based on observed failure patterns.

## Tips for Success

1. **Start conservative:** Don't change too many parameters at once
2. **Monitor overgeneralization:** A successful edit that breaks other knowledge is worse than no edit
3. **Check generation quality:** Look at the generated text to ensure coherence
4. **Balance tradeoffs:** 
   - Stronger edits → Better success but more side effects
   - Broader edits → Better generalization but more overgeneralization
5. **Relation-specific tuning:** Different relations may need different parameters

## Quick Reference: What to Check First

If you see:
- **Low rewrite success (<95%):** Increase `v_lr` and `v_num_grad_steps`
- **Low paraphrase success (<70%):** Increase `mom2_update_weight`, decrease `kl_factor`
- **High neighborhood changes (>30%):** Decrease `mom2_update_weight`, decrease `clamp_norm_factor`
- **Incoherent generations:** Decrease all edit strengths, increase regularization

## Contact & Support

For more information about AlphaEdit and its parameters, see:
- `AlphaEdit/AlphaEdit_hparams.py` - Parameter definitions
- `AlphaEdit/AlphaEdit_main.py` - Main algorithm implementation
- `hparams/AlphaEdit/` - Example hyperparameter configurations

Good luck with your edits! 🚀
