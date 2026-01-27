#!/bin/bash
# Comprehensive analysis of all news AlphaEdit evaluation runs

echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║              NEWS ALPHAEDIT EVALUATION RUNS - COMPREHENSIVE ANALYSIS         ║"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
echo ""

# Run comprehensive comparison
echo "Running comparison across all news runs..."
python3 compare_runs.py \
  --runs \
    news_alphaedit_eval/runs/temporal/llama3/run_030 \
    news_alphaedit_eval/runs/temporal/gpt-j/run_040 \
    news_alphaedit_eval/runs/temporal/gpt2-xl/run_031 \
    news_alphaedit_eval/runs/temporal/gpt2-xl/run_041 \
    news_alphaedit_eval/runs/random/gpt2-xl/run_030 \
    news_alphaedit_eval/runs/random/gpt2-xl/run_047 \
  --export news_runs_comparison.json

echo ""
echo "════════════════════════════════════════════════════════════════════════════════"
echo "Analyzing top performing run (Llama3 temporal)..."
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""

python3 analyze_failed_edits.py \
  --run-dir news_alphaedit_eval/runs/temporal/llama3/run_030 \
  --export news_failed_llama3.json \
  --create-retry-dataset news_retry_llama3.json \
  --show-details 5

echo ""
echo "════════════════════════════════════════════════════════════════════════════════"
echo "KEY INSIGHTS"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""
echo "🏆 BEST PERFORMER: Llama3 temporal/run_030"
echo "   - Overall success: 64.9%"
echo "   - Paraphrase: 65.3% (best generalization)"
echo "   - Neighborhood preservation: 16.9%"
echo ""
echo "📊 MODEL RANKING (by overall success):"
echo "   1. Llama3-8B:  64.9%"
echo "   2. GPT2-XL (random): 62.8%"
echo "   3. GPT-J-6B:   56.5%"
echo "   4. GPT2-XL (temporal): ~50%"
echo ""
echo "⚠️  CRITICAL FINDINGS:"
echo "   - All models struggle with paraphrase generalization"
echo "   - Overgeneralization is a major concern (83-93% of unrelated facts affected)"
echo "   - Random edits outperform temporal edits on GPT2-XL"
echo ""
echo "📁 Generated files:"
echo "   - news_runs_comparison.json"
echo "   - news_failed_llama3.json"
echo "   - news_retry_llama3.json"
echo "   - news_diagnostic_llama3_temporal.json"
echo "   - news_recommendations_llama3_temporal.txt"
echo ""
echo "📖 For detailed analysis, see: NEWS_RUNS_ANALYSIS.md"
echo "════════════════════════════════════════════════════════════════════════════════"

