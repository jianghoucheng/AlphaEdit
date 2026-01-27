# AlphaEdit Failed Edits Analysis Toolkit

This toolkit provides comprehensive analysis and diagnostic tools for understanding and improving AlphaEdit model editing results.

## 🎯 What This Toolkit Does

Analyzes your AlphaEdit runs to:
1. **Identify which edits failed** and why
2. **Diagnose specific failure modes** (weak edits, poor generalization, overgeneralization)
3. **Provide actionable recommendations** for parameter tuning
4. **Generate retry datasets** for re-running failed edits
5. **Compare multiple runs** to track improvement over time

## 📊 Quick Start

### Option 1: One-Command Analysis (Recommended)
```bash
./quick_analysis.sh results/AlphaEdit/run_050
```
This runs all analysis tools and generates a complete report.

### Option 2: Individual Tools

**Basic Analysis:**
```bash
python3 analyze_failed_edits.py --run-dir results/AlphaEdit/run_050 --show-details 20
```

**Deep Diagnostic:**
```bash
python3 diagnose_edit_failures.py --run-dir results/AlphaEdit/run_050
```

**Examine Specific Case:**
```bash
python3 examine_edit.py --run-dir results/AlphaEdit/run_050 --case-id 0
```

**Compare Runs:**
```bash
python3 compare_runs.py --runs results/AlphaEdit/run_030 results/AlphaEdit/run_040 results/AlphaEdit/run_050
```

## 📁 Tools Overview

| Tool | Purpose | Output |
|------|---------|--------|
| `quick_analysis.sh` | One-command comprehensive analysis | All reports |
| `analyze_failed_edits.py` | Quick statistics and retry dataset | JSON files |
| `diagnose_edit_failures.py` | Deep diagnostic with recommendations | JSON + TXT reports |
| `examine_edit.py` | Detailed individual case inspection | Console output |
| `compare_runs.py` | Multi-run comparison and trends | Comparison tables |

## 📚 Documentation

- **[ANALYSIS_SUMMARY.md](ANALYSIS_SUMMARY.md)** - Executive summary of your results
- **[FAILED_EDITS_GUIDE.md](FAILED_EDITS_GUIDE.md)** - Complete usage guide with examples
- **recommendations_run050.txt** - Parameter tuning recommendations (generated)

## 🔍 Understanding Your Results

### Key Metrics

**Success Rates:**
- **Rewrite Success:** Does the edit work on the original prompt? (Target: >95%)
- **Paraphrase Success:** Does it generalize to paraphrases? (Target: >70%)
- **Neighborhood Preservation:** Are unrelated facts preserved? (Target: >80%)

**Probabilities:**
- **P(target_new):** Model's confidence in the new target
- **P(target_true):** Model's confidence in the original target
- **Goal:** P(target_new) >> P(target_true), ideally ratio > 5.0

### Common Failure Modes

1. **Paraphrase Generalization Failed** (~44% of failures)
   - Edit works on exact prompt but fails on paraphrases
   - **Solution:** Increase mom2_update_weight, reduce kl_factor

2. **Weak Edit** (~44% of failures)
   - Edit succeeds but with low confidence
   - **Solution:** Increase v_lr and v_num_grad_steps

3. **Rewrite Failure** (~1% of failures)
   - Edit fails even on original prompt
   - **Solution:** Significantly increase v_lr and v_num_grad_steps

4. **Overgeneralization** (~4% of failures)
   - Edit affects unrelated facts
   - **Solution:** Decrease mom2_update_weight, reduce clamp_norm_factor

## 🔧 Parameter Tuning Guide

### For Poor Generalization (Most Common Issue)

```json
{
  "mom2_update_weight": 20000,  // ↑ from 15000
  "kl_factor": 0.04,             // ↓ from 0.0625
  "v_num_grad_steps": 50,        // ↑ from 25
  "v_lr": 0.5                    // keep or increase
}
```

### For Overgeneralization

```json
{
  "clamp_norm_factor": 0.5,      // ↓ from 0.75
  "nullspace_threshold": 0.03,   // ↑ from 0.02
  "mom2_update_weight": 10000    // ↓ from 15000
}
```

### For Weak Edits

```json
{
  "v_lr": 1.0,                   // ↑ from 0.1-0.5
  "v_num_grad_steps": 100,       // ↑ from 25
  "v_weight_decay": 0.3          // adjust as needed
}
```

## 📈 Workflow for Improvement

1. **Analyze Current Run**
   ```bash
   ./quick_analysis.sh results/AlphaEdit/run_050
   ```

2. **Read Recommendations**
   ```bash
   cat recommendations_run050.txt
   ```

3. **Examine Specific Failures**
   ```bash
   python3 examine_edit.py --run-dir results/AlphaEdit/run_050 --case-id 0
   ```

4. **Create New Config**
   - Copy existing hparams file
   - Adjust parameters based on recommendations
   - Save as new file (e.g., `improved_params.json`)

5. **Test on Small Subset**
   ```bash
   python experiments/evaluate.py \
       --alg_name AlphaEdit \
       --model_name <model> \
       --hparams_fname improved_params.json \
       --dataset_size_limit 200 \
       --num_edits 10
   ```

6. **Compare Results**
   ```bash
   python3 compare_runs.py --runs results/AlphaEdit/run_050 results/AlphaEdit/run_NEW
   ```

7. **Iterate** until you achieve target metrics

## 📊 Example Analysis Output

```
Analysis for: results/AlphaEdit/run_050

Total Edits: 1603
Rewrite Success Rate: 1592/1603 (99.3%)
Paraphrase Success Rate: 890/1603 (55.5%)
Both Success: 884/1603 (55.1%)

Failed Edits: 719

Failure Categories:
  paraphrase_generalization_failed: 708
  target_new_probability_too_low: 9
  rewrite_incorrect_despite_probabilities: 2

Relations with Most Failures (Top 5):
  P413: 61 failures
  P103: 54 failures
  P1412: 50 failures
  P136: 47 failures
  P30: 47 failures
```

## 🎯 Target Metrics for Success

| Metric | Current (run_050) | Target | Status |
|--------|-------------------|--------|--------|
| Rewrite Success | 99.3% | >95% | ✅ Excellent |
| Paraphrase Success | 55.5% | >70% | ⚠️ Needs improvement |
| Both Success | 55.1% | >65% | ⚠️ Needs improvement |
| Neighborhood Preservation | 11.8% | >80% | ❌ Critical issue |

## 📦 Generated Files

After running analysis, you'll have:

- `failed_edits_<run>.json` - All failed edits with detailed metrics
- `retry_dataset_<run>.json` - Dataset for re-running just the failures
- `diagnostic_<run>.json` - Detailed diagnostic data (structured)
- `recommendations_<run>.json` - Parameter recommendations (structured)
- `recommendations_<run>.txt` - Parameter recommendations (human-readable)

## 🔄 Using Failed Edits for Retry

The `retry_dataset_<run>.json` file contains only failed edits in a format ready for re-running:

```python
# Example: Load and use retry dataset
import json

with open('retry_dataset_run050.json', 'r') as f:
    failed_cases = json.load(f)

# Each case has:
# - case_id
# - requested_rewrite (subject, target_new, target_true, relation_id)
# - original_failure_reason
# - original_probabilities

# You can filter by failure reason:
weak_edits = [c for c in failed_cases if 'probability_too_low' in c['original_failure_reason']]
generalization_failures = [c for c in failed_cases if 'generalization' in c['original_failure_reason']]
```

## 🤝 Tips for Success

1. **Start with small tests** - Don't run on full dataset until parameters are tuned
2. **Balance tradeoffs** - Stronger edits may cause more side effects
3. **Monitor overgeneralization** - A successful edit that breaks other knowledge is worse than no edit
4. **Use relation-specific configs** - Different relation types may need different parameters
5. **Check generation quality** - Examine generated text to ensure coherence
6. **Iterate systematically** - Change one or two parameters at a time

## 🐛 Troubleshooting

**Q: Analysis shows no failed edits but I know some failed**

A: Check that you're analyzing the correct run directory and that result files exist.

**Q: All edits marked as failures**

A: Your parameters may be too conservative. Try increasing v_lr and v_num_grad_steps.

**Q: Edits work but model generates nonsense**

A: You have overgeneralization. Reduce edit strength parameters (clamp_norm_factor, mom2_update_weight).

**Q: Different relations have very different success rates**

A: Consider creating relation-specific hyperparameter configs.

## 📞 Support

For detailed information:
- See [ANALYSIS_SUMMARY.md](ANALYSIS_SUMMARY.md) for your specific results
- See [FAILED_EDITS_GUIDE.md](FAILED_EDITS_GUIDE.md) for comprehensive guide
- Check generated `recommendations_*.txt` files for specific advice

## 🚀 Next Steps

1. Start with: `./quick_analysis.sh results/AlphaEdit/run_050`
2. Read: `ANALYSIS_SUMMARY.md` and `recommendations_run050.txt`
3. Adjust parameters and test on small subset
4. Compare results and iterate

Good luck with your model editing! 🎉
