#!/usr/bin/env python3
"""
Analyze AlphaEdit runs to identify and categorize failed edits.
"""
import json
import os
from pathlib import Path
from collections import defaultdict
import argparse


def analyze_edit_result(result_file):
    """
    Analyze a single edit result file and determine if it succeeded or failed.
    
    Returns:
        dict: Analysis results including success/failure status and metrics
    """
    with open(result_file, 'r') as f:
        data = json.load(f)
    
    # Extract key metrics
    case_id = data['case_id']
    rewrite = data['requested_rewrite']
    post_metrics = data['post']
    
    # Check if rewrite was successful
    rewrite_success = all(post_metrics.get('rewrite_prompts_correct', []))
    paraphrase_success = all(post_metrics.get('paraphrase_prompts_correct', []))
    
    # Calculate success rates
    rewrite_rate = sum(post_metrics.get('rewrite_prompts_correct', [])) / len(post_metrics.get('rewrite_prompts_correct', [1]))
    paraphrase_rate = sum(post_metrics.get('paraphrase_prompts_correct', [])) / len(post_metrics.get('paraphrase_prompts_correct', [1]))
    neighborhood_rate = sum(post_metrics.get('neighborhood_prompts_correct', [])) / len(post_metrics.get('neighborhood_prompts_correct', [1]))
    
    # Check probability differences
    rewrite_probs = post_metrics.get('rewrite_prompts_probs', [{}])[0]
    prob_target_new = rewrite_probs.get('target_new', 0)
    prob_target_true = rewrite_probs.get('target_true', 0)
    
    # Determine failure reason
    failure_reason = None
    if not rewrite_success:
        if prob_target_new < prob_target_true:
            failure_reason = "target_new_probability_too_low"
        elif abs(prob_target_new - prob_target_true) < 1.0:
            failure_reason = "insufficient_probability_difference"
        else:
            failure_reason = "rewrite_incorrect_despite_probabilities"
    elif not paraphrase_success:
        failure_reason = "paraphrase_generalization_failed"
    
    return {
        'case_id': case_id,
        'file': str(result_file),
        'subject': rewrite.get('subject', 'N/A'),
        'relation_id': rewrite.get('relation_id', 'N/A'),
        'target_new': rewrite['target_new']['str'],
        'target_true': rewrite['target_true']['str'],
        'rewrite_success': rewrite_success,
        'paraphrase_success': paraphrase_success,
        'rewrite_rate': rewrite_rate,
        'paraphrase_rate': paraphrase_rate,
        'neighborhood_rate': neighborhood_rate,
        'prob_target_new': prob_target_new,
        'prob_target_true': prob_target_true,
        'failure_reason': failure_reason,
        'num_edits': data.get('num_edits', 1),
        'execution_time': data.get('time', 0)
    }


def analyze_run(run_dir):
    """
    Analyze all edit results in a run directory.
    
    Returns:
        dict: Summary statistics and list of failed edits
    """
    run_path = Path(run_dir)
    
    # Find all edit result files
    edit_files = sorted(run_path.glob('*_edits-case_*.json'))
    
    if not edit_files:
        print(f"No edit result files found in {run_dir}")
        return None
    
    results = []
    failed_edits = []
    failure_categories = defaultdict(list)
    
    for edit_file in edit_files:
        try:
            analysis = analyze_edit_result(edit_file)
            results.append(analysis)
            
            if not analysis['rewrite_success'] or not analysis['paraphrase_success']:
                failed_edits.append(analysis)
                if analysis['failure_reason']:
                    failure_categories[analysis['failure_reason']].append(analysis)
        except Exception as e:
            print(f"Error analyzing {edit_file}: {e}")
    
    # Calculate summary statistics
    total_edits = len(results)
    rewrite_successes = sum(1 for r in results if r['rewrite_success'])
    paraphrase_successes = sum(1 for r in results if r['paraphrase_success'])
    both_successes = sum(1 for r in results if r['rewrite_success'] and r['paraphrase_success'])
    
    avg_rewrite_rate = sum(r['rewrite_rate'] for r in results) / total_edits if total_edits > 0 else 0
    avg_paraphrase_rate = sum(r['paraphrase_rate'] for r in results) / total_edits if total_edits > 0 else 0
    avg_neighborhood_rate = sum(r['neighborhood_rate'] for r in results) / total_edits if total_edits > 0 else 0
    
    return {
        'run_dir': str(run_path),
        'total_edits': total_edits,
        'rewrite_successes': rewrite_successes,
        'paraphrase_successes': paraphrase_successes,
        'both_successes': both_successes,
        'failed_edits_count': len(failed_edits),
        'avg_rewrite_rate': avg_rewrite_rate,
        'avg_paraphrase_rate': avg_paraphrase_rate,
        'avg_neighborhood_rate': avg_neighborhood_rate,
        'failure_categories': {k: len(v) for k, v in failure_categories.items()},
        'failed_edits': failed_edits,
        'all_results': results
    }


def print_summary(summary):
    """Print a summary of the analysis."""
    print(f"\n{'='*80}")
    print(f"Analysis for: {summary['run_dir']}")
    print(f"{'='*80}")
    print(f"\nTotal Edits: {summary['total_edits']}")
    print(f"Rewrite Success Rate: {summary['rewrite_successes']}/{summary['total_edits']} ({100*summary['rewrite_successes']/summary['total_edits']:.1f}%)")
    print(f"Paraphrase Success Rate: {summary['paraphrase_successes']}/{summary['total_edits']} ({100*summary['paraphrase_successes']/summary['total_edits']:.1f}%)")
    print(f"Both Success: {summary['both_successes']}/{summary['total_edits']} ({100*summary['both_successes']/summary['total_edits']:.1f}%)")
    print(f"\nFailed Edits: {summary['failed_edits_count']}")
    
    print(f"\nAverage Success Rates:")
    print(f"  Rewrite: {100*summary['avg_rewrite_rate']:.1f}%")
    print(f"  Paraphrase: {100*summary['avg_paraphrase_rate']:.1f}%")
    print(f"  Neighborhood: {100*summary['avg_neighborhood_rate']:.1f}%")
    
    if summary['failure_categories']:
        print(f"\nFailure Categories:")
        for category, count in summary['failure_categories'].items():
            print(f"  {category}: {count}")


def print_failed_edits_details(failed_edits, limit=10):
    """Print detailed information about failed edits."""
    print(f"\n{'='*80}")
    print(f"Failed Edits Details (showing up to {limit})")
    print(f"{'='*80}")
    
    for i, edit in enumerate(failed_edits[:limit], 1):
        print(f"\n{i}. Case ID: {edit['case_id']}")
        print(f"   Subject: {edit['subject']}")
        print(f"   Relation: {edit['relation_id']}")
        print(f"   Target True: {edit['target_true']} → Target New: {edit['target_new']}")
        print(f"   Rewrite Success: {edit['rewrite_success']} (rate: {100*edit['rewrite_rate']:.1f}%)")
        print(f"   Paraphrase Success: {edit['paraphrase_success']} (rate: {100*edit['paraphrase_rate']:.1f}%)")
        print(f"   Probabilities - New: {edit['prob_target_new']:.4f}, True: {edit['prob_target_true']:.4f}")
        print(f"   Failure Reason: {edit['failure_reason']}")
        print(f"   File: {Path(edit['file']).name}")


def export_failed_edits(failed_edits, output_file):
    """Export failed edits to a JSON file for further analysis."""
    export_data = {
        'total_failed': len(failed_edits),
        'failed_edits': failed_edits
    }
    
    with open(output_file, 'w') as f:
        json.dump(export_data, f, indent=2)
    
    print(f"\nFailed edits exported to: {output_file}")


def create_retry_dataset(failed_edits, output_file):
    """
    Create a dataset file with failed edits that can be retried.
    This extracts the requested_rewrite information in a format that can be used
    to re-run the edits with different parameters.
    """
    retry_data = []
    
    for edit in failed_edits:
        retry_entry = {
            'case_id': edit['case_id'],
            'requested_rewrite': {
                'subject': edit['subject'],
                'target_new': {'str': edit['target_new']},
                'target_true': {'str': edit['target_true']},
                'relation_id': edit['relation_id']
            },
            'original_failure_reason': edit['failure_reason'],
            'original_probabilities': {
                'target_new': edit['prob_target_new'],
                'target_true': edit['prob_target_true']
            }
        }
        retry_data.append(retry_entry)
    
    with open(output_file, 'w') as f:
        json.dump(retry_data, f, indent=2)
    
    print(f"Retry dataset created: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze AlphaEdit runs to identify and categorize failed edits'
    )
    parser.add_argument(
        '--run-dir',
        type=str,
        required=True,
        help='Path to the run directory (e.g., results/AlphaEdit/run_050)'
    )
    parser.add_argument(
        '--export',
        type=str,
        default=None,
        help='Export failed edits to JSON file'
    )
    parser.add_argument(
        '--create-retry-dataset',
        type=str,
        default=None,
        help='Create a retry dataset file with failed edits'
    )
    parser.add_argument(
        '--show-details',
        type=int,
        default=10,
        help='Number of failed edits to show in detail (default: 10)'
    )
    
    args = parser.parse_args()
    
    # Analyze the run
    summary = analyze_run(args.run_dir)
    
    if summary is None:
        return
    
    # Print summary
    print_summary(summary)
    
    # Print details of failed edits
    if summary['failed_edits']:
        print_failed_edits_details(summary['failed_edits'], limit=args.show_details)
    
    # Export if requested
    if args.export:
        export_failed_edits(summary['failed_edits'], args.export)
    
    # Create retry dataset if requested
    if args.create_retry_dataset:
        create_retry_dataset(summary['failed_edits'], args.create_retry_dataset)


if __name__ == '__main__':
    main()
