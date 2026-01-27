# News AlphaEdit Evaluation Runs Analysis

## Executive Summary

Analysis of AlphaEdit runs on news evaluation datasets across multiple models (Llama3, GPT-J, GPT2-XL) and edit types (temporal, random, temporal_tag).

### 📊 Key Findings

#### Overall Performance Comparison

| Run | Model | Type | Total | Rewrite% | Para% | Both% | Neigh% |
|-----|-------|------|-------|----------|-------|-------|--------|
| temporal/llama3/run_030 | Llama3-8B | Temporal | 1603 | **95.1%** | **65.3%** | **64.9%** | **16.9%** |
| temporal/gpt-j/run_040 | GPT-J-6B | Temporal | 1603 | 99.3% | 56.8% | 56.5% | 12.1% |
| temporal/gpt2-xl/run_031 | GPT2-XL | Temporal | 1603 | 97.3% | 49.9% | 49.4% | 6.9% |
| temporal/gpt2-xl/run_041 | GPT2-XL | Temporal | 1603 | 97.6% | 50.8% | 50.3% | 7.5% |
| random/gpt2-xl/run_030 | GPT2-XL | Random | 1239 | 93.6% | 63.4% | 62.8% | 17.0% |
| random/gpt2-xl/run_047 | GPT2-XL | Random | 1603 | 97.9% | 51.3% | 51.0% | 7.1% |

**Bold** = Best performance

### 🎯 Critical Insights

1. **Llama3 Shows Best Overall Performance**
   - Highest paraphrase success: 65.3%
   - Best overall success: 64.9%
   - Good neighborhood preservation: 16.9%

2. **Generalization is the Main Challenge**
   - All models show significantly lower paraphrase vs rewrite success
   - Average gap: 30-40 percentage points
   - Indicates edits are too narrow and specific

3. **Neighborhood Preservation is Concerning**
   - Most runs show <13% preservation (>87% overgeneralization!)
   - Only Llama3 temporal and GPT2-XL random show decent preservation (~17%)
   - This indicates edits are affecting unrelated facts

4. **Model Architecture Matters**
   - **Llama3-8B:** Best generalization, best overall
   - **GPT-J-6B:** High rewrite success but poor generalization
   - **GPT2-XL:** Varied performance, generally struggles with generalization

## Detailed Analysis by Model

### 🦙 Llama3-8B (temporal/run_030) - BEST PERFORMER

**Strengths:**
- ✅ Highest paraphrase success (65.3%)
- ✅ Best overall success rate (64.9%)
- ✅ Good neighborhood preservation (16.9%)
- ✅ Balanced rewrite/paraphrase performance

**Weaknesses:**
- ⚠️ Still has 35% paraphrase failure rate
- ⚠️ 563 total failed edits

**Failure Breakdown:**
- Paraphrase generalization: 484 cases (86.0%)
- Rewrite failure: 63 cases (11.2%)
- Other: 16 cases (2.8%)

**Most Problematic Relations:**
1. P413 (Position played) - 43 failures
2. P30 (Continent) - 40 failures
3. P103 (Native language) - 39 failures
4. P131 (Located in) - 32 failures
5. P37 (Official language) - 30 failures

**Recommendations for Llama3:**
```json
{
  "mom2_update_weight": 18000,  // Slight increase from current
  "kl_factor": 0.05,             // Slight reduction
  "v_num_grad_steps": 40,        // Moderate increase
  "v_lr": 0.4                    // Keep moderate
}
```

### 🤖 GPT-J-6B (temporal/run_040)

**Strengths:**
- ✅ Excellent rewrite success (99.3%)
- ✅ Strong edit insertion

**Weaknesses:**
- ❌ Poor paraphrase generalization (56.8%)
- ❌ Low neighborhood preservation (12.1%)
- ❌ Large gap between rewrite and paraphrase (42.5 points)

**Failure Breakdown:**
- Paraphrase generalization: 692 cases (97.5%)
- Weak edits: 685 cases (96.6%)
- Rewrite failure: 11 cases (1.5%)

**Most Problematic Relations:**
1. P413 (Position played) - 59 failures
2. P103 (Native language) - 53 failures
3. P1412 (Languages spoken) - 53 failures
4. P30 (Continent) - 47 failures
5. P106 (Occupation) - 47 failures

**Diagnosis:**
GPT-J achieves high rewrite success but with **weak, narrow edits** that don't generalize. The edits are just strong enough to work on exact prompts but fail on variations.

**Recommendations for GPT-J:**
```json
{
  "mom2_update_weight": 22000,  // Significant increase for broader edits
  "kl_factor": 0.03,             // Major reduction for flexibility
  "v_num_grad_steps": 60,        // Significant increase
  "v_lr": 0.6,                   // Increase edit strength
  "clamp_norm_factor": 0.6       // Reduce to control overgeneralization
}
```

### 📝 GPT2-XL (temporal/run_031 & run_041)

**Strengths:**
- ✅ Consistent rewrite success (~97%)
- ✅ Run_041 shows slight improvement over run_031

**Weaknesses:**
- ❌ Poorest paraphrase generalization (~50%)
- ❌ Very low neighborhood preservation (6.9-7.5%)
- ❌ Highest overgeneralization problem

**Comparison run_031 vs run_041:**
- Rewrite: 97.3% → 97.6% (+0.3%)
- Paraphrase: 49.9% → 50.8% (+0.9%)
- Neighborhood: 6.9% → 7.5% (+0.6%)

Minor improvement but still underperforming.

**Recommendations for GPT2-XL:**
```json
{
  "mom2_update_weight": 20000,   // Increase for generalization
  "kl_factor": 0.04,              // Reduce constraint
  "v_num_grad_steps": 70,         // Significant increase
  "v_lr": 0.5,                    // Moderate increase
  "clamp_norm_factor": 0.4,       // Major reduction for overgeneralization
  "nullspace_threshold": 0.04     // Increase for tighter constraint
}
```

### 🎲 Random vs Temporal Edits (GPT2-XL)

Comparing random/run_030 (1239 edits) vs temporal runs:

| Metric | Random/run_030 | Temporal/run_031 | Temporal/run_041 |
|--------|----------------|------------------|------------------|
| Rewrite | 93.6% | 97.3% | 97.6% |
| Paraphrase | 63.4% | 49.9% | 50.8% |
| Overall | 62.8% | 49.4% | 50.3% |
| Neighborhood | 17.0% | 6.9% | 7.5% |

**Key Observations:**
1. **Random edits generalize BETTER** than temporal edits (63.4% vs ~50%)
2. **Random edits preserve neighborhoods BETTER** (17.0% vs ~7%)
3. **Temporal edits have higher rewrite success** but at cost of generalization

**Hypothesis:** Temporal edits may be more complex or contradictory to existing knowledge, making them harder to generalize while staying specific.

## Common Failure Patterns Across All Models

### 1. Paraphrase Generalization (Most Common)
- **Affects:** 86-98% of failures
- **Symptom:** Works on exact prompt, fails on variations
- **Cause:** Edits are too narrow and tied to specific wording

### 2. Weak Edits
- **Affects:** GPT-J heavily (685 cases), others moderately
- **Symptom:** Low P(target_new) even when "successful"
- **Cause:** Edit strength insufficient

### 3. Overgeneralization
- **Affects:** All models, especially GPT2-XL
- **Symptom:** Low neighborhood preservation (<17%)
- **Cause:** Edits affecting unrelated facts

### 4. Problematic Relations (Consistent Across Models)
- **P413** (Position played) - Always in top 3 failures
- **P103** (Native language) - Always in top 5 failures
- **P30** (Continent) - Consistently problematic
- **P131** (Located in) - High failure rate
- **P1412** (Languages spoken) - Difficult to edit

## Model-Specific Strengths

### When to Use Each Model

**Llama3-8B:**
- ✅ General purpose editing
- ✅ When generalization is critical
- ✅ Balanced performance needed
- ✅ Moderate computational budget

**GPT-J-6B:**
- ✅ When rewrite accuracy is most important
- ✅ Less concern about generalization
- ✅ Specific, targeted edits
- ⚠️ Requires additional tuning for generalization

**GPT2-XL:**
- ⚠️ Challenging for knowledge editing
- ⚠️ Needs significant parameter tuning
- ⚠️ High overgeneralization risk
- ℹ️ Random edits perform better than temporal

## Actionable Recommendations

### Immediate Priority Actions

#### 1. For Llama3 (Currently Best)
**Goal:** Improve from 65% to 75%+ paraphrase success

```bash
# Create improved config
cat > hparams/AlphaEdit/llama3_improved.json << EOF
{
  "model_name": "Llama3-8B",
  "layers": [4, 5, 6, 7, 8, 9],  # Add one more layer
  "mom2_update_weight": 18000,
  "kl_factor": 0.05,
  "v_num_grad_steps": 40,
  "v_lr": 0.4,
  "clamp_norm_factor": 0.65,
  "nullspace_threshold": 0.025
}
EOF

# Test on small subset
python experiments/evaluate.py \
    --alg_name AlphaEdit \
    --model_name meta-llama/Meta-Llama-3-8B-Instruct \
    --hparams_fname llama3_improved.json \
    --dataset_size_limit 200 \
    --num_edits 10
```

#### 2. For GPT-J (Fix Generalization)
**Goal:** Improve paraphrase from 57% to 70%+

```json
{
  "mom2_update_weight": 22000,
  "kl_factor": 0.03,
  "v_num_grad_steps": 60,
  "v_lr": 0.6,
  "clamp_norm_factor": 0.6
}
```

#### 3. For GPT2-XL (Major Overhaul Needed)
**Goal:** Improve paraphrase from 50% to 65%+ AND fix overgeneralization

```json
{
  "mom2_update_weight": 20000,
  "kl_factor": 0.04,
  "v_num_grad_steps": 70,
  "v_lr": 0.5,
  "clamp_norm_factor": 0.4,
  "nullspace_threshold": 0.04
}
```

### Relation-Specific Tuning

Create specialized configs for problematic relations:

**For P413 (Position played):**
```json
{
  "v_lr": 0.7,
  "v_num_grad_steps": 80,
  "mom2_update_weight": 25000
}
```

**For P103/P1412 (Languages):**
```json
{
  "v_lr": 0.6,
  "v_num_grad_steps": 70,
  "mom2_update_weight": 23000,
  "kl_factor": 0.03
}
```

## Comparison: News Runs vs Main AlphaEdit Runs

### Main AlphaEdit run_050 (from earlier analysis):
- Model: Llama3-8B (likely) or GPT-J
- Rewrite: 99.3%
- Paraphrase: 55.5%
- Neighborhood: 11.8%

### News Temporal Llama3 run_030:
- Model: Llama3-8B
- Rewrite: 95.1%
- Paraphrase: 65.3%
- Neighborhood: 16.9%

**Key Differences:**
1. News run has **BETTER paraphrase** (+9.8 points)
2. News run has **BETTER neighborhood** (+5.1 points)
3. News run has slightly **LOWER rewrite** (-4.2 points)

**Hypothesis:** News runs use better tuned parameters that prioritize generalization over rewrite accuracy. This is the correct tradeoff!

## Testing Strategy

### Phase 1: Validate Llama3 Improvements (Week 1)
```bash
# Test improved Llama3 config
./quick_analysis.sh news_alphaedit_eval/runs/temporal/llama3/run_NEW

# Target metrics:
# - Paraphrase: 65% → 75%
# - Neighborhood: 16.9% → 20%+
# - Rewrite: maintain >93%
```

### Phase 2: Fix GPT-J Generalization (Week 2)
```bash
# Test with broader edits
./quick_analysis.sh news_alphaedit_eval/runs/temporal/gpt-j/run_NEW

# Target metrics:
# - Paraphrase: 56.8% → 70%
# - Neighborhood: 12.1% → 15%+
# - Rewrite: maintain >95%
```

### Phase 3: Relation-Specific Configs (Week 3)
```bash
# Create and test specialized configs for P413, P103, P1412
# Run on filtered dataset with only these relations
```

### Phase 4: Compare Edit Types (Week 4)
```bash
# Test whether temporal_tag performs better than temporal
# Analyze temporal_tag/gpt-j/run_050 and temporal_tag/gpt2-xl/run_048
```

## Files Generated for Analysis

### Comparison Files
- `news_runs_comparison.json` - Complete comparison data

### Llama3 Diagnostics
- `news_diagnostic_llama3_temporal.json` - Detailed diagnostics
- `news_recommendations_llama3_temporal.json` - Structured recommendations
- `news_recommendations_llama3_temporal.txt` - Human-readable guide

### GPT-J Diagnostics
- `news_diagnostic_gptj_temporal.json` - Detailed diagnostics
- `news_recommendations_gptj_temporal.json` - Structured recommendations
- `news_recommendations_gptj_temporal.txt` - Human-readable guide

## Next Steps

1. **Immediate:** Read the model-specific recommendation files
   ```bash
   cat news_recommendations_llama3_temporal.txt
   cat news_recommendations_gptj_temporal.txt
   ```

2. **This Week:** Test improved configs on Llama3 (best performer)

3. **Next Week:** Address GPT-J generalization issues

4. **Ongoing:** Develop relation-specific configs for P413, P103, P1412

5. **Research:** Investigate why random edits generalize better than temporal

## Questions for Further Investigation

1. **Why do random edits generalize better?**
   - Are temporal edits more contradictory to existing knowledge?
   - Do they require different hyperparameters?

2. **Why is Llama3 superior?**
   - Architecture differences?
   - Better pre-training for knowledge editing?
   - Optimal parameter range found by chance?

3. **Can we transfer Llama3's success to other models?**
   - Test Llama3 parameters on GPT-J
   - Analyze what makes Llama3's edits more generalizable

4. **What about temporal_tag runs?**
   - Only 2 runs: gpt-j/run_050 and gpt2-xl/run_048
   - Need to analyze these separately

## Summary

**Best Current Setup:** Llama3-8B on temporal edits (64.9% overall success)

**Main Challenge:** Paraphrase generalization across all models

**Critical Issue:** Overgeneralization (low neighborhood preservation)

**Recommended Model Priority:**
1. **Llama3-8B** - Best overall, tune to 75%+
2. **GPT-J-6B** - High rewrite success, needs generalization work
3. **GPT2-XL** - Needs major parameter overhaul

**Key Insight:** The news temporal/llama3/run_030 represents your best performing configuration. Use this as the baseline for improvements rather than the main AlphaEdit runs.

---

For detailed tool usage, see [TOOLKIT_README.md](TOOLKIT_README.md) and [FAILED_EDITS_GUIDE.md](FAILED_EDITS_GUIDE.md).
