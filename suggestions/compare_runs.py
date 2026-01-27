#!/usr/bin/env python3
"""
Compare multiple runs to track improvement over time.
"""
import json
import argparse
from pathlib import Path
from collections import defaultdict


def analyze_run_summary(run_dir):
    """Get summary statistics for a run."""
    run_path = Path(run_dir)
    edit_files = sorted(run_path.glob('*_edits-case_*.json'))
    
    if not edit_files:
        return None
    
    stats = {
        'run_name': run_path.name,
        'total_edits': 0,
        'rewrite_success': 0,
        'paraphrase_success': 0,
        'both_success': 0,
        'avg_rewrite_prob_new': [],
        'avg_paraphrase_prob_new': [],
        'avg_neighborhood_preservation': [],
        'failure_by_relation': defaultdict(int)
    }
    
    for edit_file in edit_files:
        try:
            with open(edit_file, 'r') as f:
                data = json.load(f)
            
            post = data['post']
            rewrite = data['requested_rewrite']
            
            rewrite_correct = post.get('rewrite_prompts_correct', [])
            para_correct = post.get('paraphrase_prompts_correct', [])
            neigh_correct = post.get('neighborhood_prompts_correct', [])
            
            stats['total_edits'] += 1
            
            rewrite_success = all(rewrite_correct) if rewrite_correct else False
            para_success = all(para_correct) if para_correct else False
            
            if rewrite_success:
                stats['rewrite_success'] += 1
            if para_success:
                stats['paraphrase_success'] += 1
            if rewrite_success and para_success:
                stats['both_success'] += 1
            
            # Track probabilities
            rewrite_probs = post.get('rewrite_prompts_probs', [])
            if rewrite_probs:
                avg_new = sum(p['target_new'] for p in rewrite_probs) / len(rewrite_probs)
                stats['avg_rewrite_prob_new'].append(avg_new)
            
            para_probs = post.get('paraphrase_prompts_probs', [])
            if para_probs:
                avg_new = sum(p['target_new'] for p in para_probs) / len(para_probs)
                stats['avg_paraphrase_prob_new'].append(avg_new)
            
            if neigh_correct:
                preservation = sum(neigh_correct) / len(neigh_correct)
                stats['avg_neighborhood_preservation'].append(preservation)
            
            # Track failures by relation
            if not rewrite_success or not para_success:
                relation = rewrite.get('relation_id', 'unknown')
                stats['failure_by_relation'][relation] += 1
                
        except Exception as e:
            print(f"Error processing {edit_file}: {e}")
    
    # Calculate averages
    if stats['avg_rewrite_prob_new']:
        stats['avg_rewrite_prob_new'] = sum(stats['avg_rewrite_prob_new']) / len(stats['avg_rewrite_prob_new'])
    else:
        stats['avg_rewrite_prob_new'] = 0
    
    if stats['avg_paraphrase_prob_new']:
        stats['avg_paraphrase_prob_new'] = sum(stats['avg_paraphrase_prob_new']) / len(stats['avg_paraphrase_prob_new'])
    else:
        stats['avg_paraphrase_prob_new'] = 0
    
    if stats['avg_neighborhood_preservation']:
        stats['avg_neighborhood_preservation'] = sum(stats['avg_neighborhood_preservation']) / len(stats['avg_neighborhood_preservation'])
    else:
        stats['avg_neighborhood_preservation'] = 0
    
    stats['failure_by_relation'] = dict(stats['failure_by_relation'])
    
    return stats


def compare_runs(run_dirs):
    """Compare multiple runs."""
    all_stats = []
    
    for run_dir in run_dirs:
        print(f"Analyzing {run_dir}...")
        stats = analyze_run_summary(run_dir)
        if stats:
            all_stats.append(stats)
    
    return all_stats


def print_comparison_table(all_stats):
    """Print comparison table."""
    if not all_stats:
        print("No runs to compare")
        return
    
    print(f"\n{'='*120}")
    print("RUN COMPARISON")
    print(f"{'='*120}")
    
    # Header
    print(f"\n{'Run':<15} {'Total':<8} {'Rewrite%':<10} {'Para%':<10} {'Both%':<10} "
          f"{'AvgP(new)Rew':<13} {'AvgP(new)Para':<13} {'Neigh%':<10}")
    print("-" * 120)
    
    # Rows
    for stats in all_stats:
        total = stats['total_edits']
        if total == 0:
            continue
        
        rewrite_pct = 100 * stats['rewrite_success'] / total
        para_pct = 100 * stats['paraphrase_success'] / total
        both_pct = 100 * stats['both_success'] / total
        neigh_pct = 100 * stats['avg_neighborhood_preservation']
        
        print(f"{stats['run_name']:<15} {total:<8} {rewrite_pct:<10.1f} {para_pct:<10.1f} {both_pct:<10.1f} "
              f"{stats['avg_rewrite_prob_new']:<13.4f} {stats['avg_paraphrase_prob_new']:<13.4f} {neigh_pct:<10.1f}")
    
    print("\nLegend:")
    print("  Rewrite%: Success rate on original prompts")
    print("  Para%: Success rate on paraphrased prompts")
    print("  Both%: Success rate on both")
    print("  AvgP(new): Average probability assigned to target_new")
    print("  Neigh%: Neighborhood preservation rate (higher = less overgeneralization)")


def print_improvement_analysis(all_stats):
    """Analyze improvement over runs."""
    if len(all_stats) < 2:
        print("\nNeed at least 2 runs to analyze improvement")
        return
    
    print(f"\n{'='*120}")
    print("IMPROVEMENT ANALYSIS")
    print(f"{'='*120}")
    
    baseline = all_stats[0]
    
    for i, stats in enumerate(all_stats[1:], 1):
        print(f"\n{stats['run_name']} vs {baseline['run_name']}:")
        
        # Calculate changes
        total_base = baseline['total_edits']
        total_curr = stats['total_edits']
        
        if total_base > 0 and total_curr > 0:
            rewrite_change = (stats['rewrite_success'] / total_curr) - (baseline['rewrite_success'] / total_base)
            para_change = (stats['paraphrase_success'] / total_curr) - (baseline['paraphrase_success'] / total_base)
            both_change = (stats['both_success'] / total_curr) - (baseline['both_success'] / total_base)
            neigh_change = stats['avg_neighborhood_preservation'] - baseline['avg_neighborhood_preservation']
            
            print(f"  Rewrite Success: {rewrite_change:+.1%} ({100*stats['rewrite_success']/total_curr:.1f}% vs {100*baseline['rewrite_success']/total_base:.1f}%)")
            print(f"  Paraphrase Success: {para_change:+.1%} ({100*stats['paraphrase_success']/total_curr:.1f}% vs {100*baseline['paraphrase_success']/total_base:.1f}%)")
            print(f"  Both Success: {both_change:+.1%} ({100*stats['both_success']/total_curr:.1f}% vs {100*baseline['both_success']/total_base:.1f}%)")
            print(f"  Neighborhood Preservation: {neigh_change:+.1%} ({100*stats['avg_neighborhood_preservation']:.1f}% vs {100*baseline['avg_neighborhood_preservation']:.1f}%)")
            
            # Overall assessment
            if both_change > 0.05:
                print(f"  ✓ Significant improvement!")
            elif both_change > 0:
                print(f"  ↗ Slight improvement")
            elif both_change > -0.05:
                print(f"  → No significant change")
            else:
                print(f"  ✗ Performance decreased")


def export_comparison(all_stats, output_file):
    """Export comparison to JSON."""
    with open(output_file, 'w') as f:
        json.dump({
            'runs': all_stats,
            'comparison_date': str(Path.cwd())
        }, f, indent=2)
    
    print(f"\nComparison exported to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Compare multiple AlphaEdit runs'
    )
    parser.add_argument(
        '--runs',
        type=str,
        nargs='+',
        required=True,
        help='Paths to run directories to compare'
    )
    parser.add_argument(
        '--export',
        type=str,
        default=None,
        help='Export comparison to JSON file'
    )
    
    args = parser.parse_args()
    
    all_stats = compare_runs(args.runs)
    
    print_comparison_table(all_stats)
    print_improvement_analysis(all_stats)
    
    if args.export:
        export_comparison(all_stats, args.export)


if __name__ == '__main__':
    main()
