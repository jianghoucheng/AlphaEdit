#!/usr/bin/env python3
"""
Interactive tool to examine specific failed edits and understand what the model
is producing vs. what was expected.
"""
import json
import argparse
from pathlib import Path
from collections import defaultdict


def display_edit_details(result_file, show_generations=True):
    """Display detailed information about a specific edit."""
    with open(result_file, 'r') as f:
        data = json.load(f)
    
    rewrite = data['requested_rewrite']
    post = data['post']
    
    print(f"\n{'='*80}")
    print(f"EDIT CASE {data['case_id']}")
    print(f"{'='*80}")
    
    # Basic information
    print(f"\nSubject: {rewrite.get('subject', 'N/A')}")
    print(f"Relation ID: {rewrite.get('relation_id', 'N/A')}")
    print(f"Prompt Template: {rewrite.get('prompt', 'N/A')}")
    print(f"\nTarget Change:")
    print(f"  FROM (true): {rewrite['target_true']['str']}")
    print(f"  TO (new):    {rewrite['target_new']['str']}")
    
    # Execution info
    print(f"\nExecution Time: {data.get('time', 0):.2f} seconds")
    print(f"Number of Edits in Batch: {data.get('num_edits', 1)}")
    print(f"Grouped with Cases: {data.get('grouped_case_ids', [])}")
    
    # Rewrite prompt results
    print(f"\n{'='*80}")
    print("REWRITE PROMPT RESULTS (Original Prompt)")
    print(f"{'='*80}")
    
    rewrite_probs = post.get('rewrite_prompts_probs', [])
    rewrite_correct = post.get('rewrite_prompts_correct', [])
    
    for i, (probs, correct) in enumerate(zip(rewrite_probs, rewrite_correct), 1):
        status = "✓ CORRECT" if correct else "✗ INCORRECT"
        print(f"\nRewrite Prompt {i}: {status}")
        print(f"  P(target_new={rewrite['target_new']['str']}): {probs['target_new']:.6f}")
        print(f"  P(target_true={rewrite['target_true']['str']}): {probs['target_true']:.6f}")
        print(f"  Ratio (new/true): {probs['target_new']/probs['target_true']:.4f}" if probs['target_true'] > 0 else "  Ratio: inf")
        
        if probs['target_new'] > probs['target_true']:
            print(f"  → Model prefers NEW target (good)")
        else:
            print(f"  → Model still prefers TRUE target (bad - edit failed)")
    
    # Paraphrase prompt results
    print(f"\n{'='*80}")
    print("PARAPHRASE PROMPT RESULTS (Generalization Test)")
    print(f"{'='*80}")
    
    para_probs = post.get('paraphrase_prompts_probs', [])
    para_correct = post.get('paraphrase_prompts_correct', [])
    
    if para_probs:
        success_rate = sum(para_correct) / len(para_correct) * 100
        print(f"Overall Paraphrase Success Rate: {success_rate:.1f}% ({sum(para_correct)}/{len(para_correct)})")
        
        for i, (probs, correct) in enumerate(zip(para_probs, para_correct), 1):
            status = "✓ CORRECT" if correct else "✗ INCORRECT"
            print(f"\nParaphrase {i}: {status}")
            print(f"  P(target_new): {probs['target_new']:.6f}")
            print(f"  P(target_true): {probs['target_true']:.6f}")
            print(f"  Ratio (new/true): {probs['target_new']/probs['target_true']:.4f}" if probs['target_true'] > 0 else "  Ratio: inf")
    
    # Neighborhood prompt results
    print(f"\n{'='*80}")
    print("NEIGHBORHOOD PROMPT RESULTS (Unrelated Facts - Should Remain Unchanged)")
    print(f"{'='*80}")
    
    neigh_probs = post.get('neighborhood_prompts_probs', [])
    neigh_correct = post.get('neighborhood_prompts_correct', [])
    
    if neigh_probs:
        # For neighborhood, "correct" means the model correctly produces the true target
        # (i.e., the edit didn't affect unrelated facts)
        success_rate = sum(neigh_correct) / len(neigh_correct) * 100
        print(f"Neighborhood Preservation Rate: {success_rate:.1f}% ({sum(neigh_correct)}/{len(neigh_correct)})")
        print(f"(Higher is better - means edit is specific and doesn't affect unrelated facts)")
        
        affected_count = sum(1 for c in neigh_correct if not c)
        if affected_count > 0:
            print(f"\n⚠ WARNING: {affected_count} neighborhood facts were affected by the edit!")
        
        # Show a few examples
        print(f"\nSample Neighborhood Results:")
        for i, (probs, correct) in enumerate(list(zip(neigh_probs, neigh_correct))[:5], 1):
            status = "✓ PRESERVED" if correct else "✗ AFFECTED"
            print(f"  Neighbor {i}: {status} (P_new={probs['target_new']:.4f}, P_true={probs['target_true']:.4f})")
    
    # Generation quality
    if show_generations and 'text' in post and post['text']:
        print(f"\n{'='*80}")
        print("GENERATED TEXT SAMPLES")
        print(f"{'='*80}")
        print(f"\nShowing {min(3, len(post['text']))} of {len(post['text'])} generation samples:")
        print(f"(These show how the model generates text after the edit)")
        
        for i, text in enumerate(post['text'][:3], 1):
            print(f"\n--- Sample {i} ---")
            # Truncate very long text
            display_text = text if len(text) < 300 else text[:297] + "..."
            print(display_text)
        
        if 'ngram_entropy' in post:
            print(f"\nN-gram Entropy: {post['ngram_entropy']:.4f}")
            print(f"(Measures generation diversity - higher is more diverse)")
        
        if 'reference_score' in post:
            print(f"Reference Score: {post['reference_score']:.4f}")
            print(f"(Similarity to reference generations)")
    
    # Diagnosis and recommendations
    print(f"\n{'='*80}")
    print("DIAGNOSIS & RECOMMENDATIONS")
    print(f"{'='*80}")
    
    rewrite_success = all(rewrite_correct) if rewrite_correct else False
    para_success = all(para_correct) if para_correct else False
    
    if rewrite_success and para_success:
        print("\n✓ EDIT SUCCESSFUL")
        print("The edit works on both original and paraphrased prompts.")
    elif rewrite_success and not para_success:
        print("\n⚠ PARTIAL SUCCESS - POOR GENERALIZATION")
        print("\nThe edit works on the exact prompt but fails on paraphrases.")
        print("This means the edit is too narrow and specific.")
        print("\nRECOMMENDATIONS:")
        print("  1. Increase mom2_update_weight (e.g., 15000 → 20000) for broader edits")
        print("  2. Reduce kl_factor (e.g., 0.0625 → 0.04) to allow more flexibility")
        print("  3. Increase v_num_grad_steps (e.g., 25 → 50) for stronger optimization")
        print("  4. Try editing different or additional layers")
    else:
        print("\n✗ EDIT FAILED")
        print("\nThe edit does not work even on the original prompt.")
        print("\nRECOMMENDATIONS:")
        print("  1. Significantly increase v_lr (e.g., 0.1 → 0.5 or 1.0)")
        print("  2. Increase v_num_grad_steps (e.g., 25 → 50 or 100)")
        print("  3. Check if the target change is too drastic or unrealistic")
        print("  4. Verify the prompt template is correct")
    
    # Check for overgeneralization
    if neigh_correct:
        neigh_success_rate = sum(neigh_correct) / len(neigh_correct)
        if neigh_success_rate < 0.7:
            print("\n⚠ OVERGENERALIZATION DETECTED")
            print(f"The edit is affecting {100*(1-neigh_success_rate):.1f}% of unrelated facts.")
            print("\nRECOMMENDATIONS:")
            print("  1. Decrease mom2_update_weight (e.g., 15000 → 10000) for more specific edits")
            print("  2. Reduce clamp_norm_factor (e.g., 0.75 → 0.5)")
            print("  3. Increase nullspace_threshold to constrain the edit more")


def compare_multiple_edits(result_files):
    """Compare multiple failed edits to find common patterns."""
    print(f"\n{'='*80}")
    print(f"COMPARING {len(result_files)} EDITS")
    print(f"{'='*80}")
    
    comparisons = []
    
    for result_file in result_files:
        with open(result_file, 'r') as f:
            data = json.load(f)
        
        rewrite = data['requested_rewrite']
        post = data['post']
        
        rewrite_correct = post.get('rewrite_prompts_correct', [])
        para_correct = post.get('paraphrase_prompts_correct', [])
        
        rewrite_probs = post.get('rewrite_prompts_probs', [{}])[0]
        
        comparisons.append({
            'case_id': data['case_id'],
            'subject': rewrite.get('subject', 'N/A'),
            'relation': rewrite.get('relation_id', 'N/A'),
            'target_change': f"{rewrite['target_true']['str']} → {rewrite['target_new']['str']}",
            'rewrite_success': all(rewrite_correct) if rewrite_correct else False,
            'para_success': all(para_correct) if para_correct else False,
            'prob_new': rewrite_probs.get('target_new', 0),
            'prob_true': rewrite_probs.get('target_true', 0),
            'file': Path(result_file).name
        })
    
    # Print comparison table
    print(f"\n{'ID':<8} {'Relation':<10} {'Rewrite':<10} {'Paraphrase':<12} {'P(new)':<10} {'P(true)':<10} {'Subject':<30}")
    print("-" * 110)
    
    for comp in comparisons:
        rewrite_status = "✓" if comp['rewrite_success'] else "✗"
        para_status = "✓" if comp['para_success'] else "✗"
        
        print(f"{comp['case_id']:<8} {comp['relation']:<10} {rewrite_status:<10} {para_status:<12} "
              f"{comp['prob_new']:<10.4f} {comp['prob_true']:<10.4f} {comp['subject'][:30]:<30}")
    
    # Find common patterns
    relations = [c['relation'] for c in comparisons]
    relation_counts = {}
    for rel in relations:
        relation_counts[rel] = relation_counts.get(rel, 0) + 1
    
    print(f"\nCommon Relations in Failed Edits:")
    for rel, count in sorted(relation_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {rel}: {count} occurrences")


def main():
    parser = argparse.ArgumentParser(
        description='Examine specific failed edits in detail'
    )
    parser.add_argument(
        '--case-file',
        type=str,
        help='Path to a specific case result file to examine'
    )
    parser.add_argument(
        '--case-id',
        type=int,
        help='Case ID to examine (requires --run-dir)'
    )
    parser.add_argument(
        '--run-dir',
        type=str,
        help='Run directory (required if using --case-id)'
    )
    parser.add_argument(
        '--compare',
        type=str,
        nargs='+',
        help='Compare multiple case files'
    )
    parser.add_argument(
        '--no-generations',
        action='store_true',
        help='Skip showing generated text samples'
    )
    
    args = parser.parse_args()
    
    if args.compare:
        compare_multiple_edits(args.compare)
    elif args.case_file:
        display_edit_details(args.case_file, show_generations=not args.no_generations)
    elif args.case_id is not None and args.run_dir:
        run_path = Path(args.run_dir)
        # Find the file for this case
        case_files = list(run_path.glob(f'*_edits-case_{args.case_id}.json'))
        if not case_files:
            print(f"Error: No file found for case {args.case_id} in {args.run_dir}")
            return
        if len(case_files) > 1:
            print(f"Warning: Multiple files found for case {args.case_id}:")
            for f in case_files:
                print(f"  {f}")
            print(f"Using: {case_files[0]}")
        display_edit_details(case_files[0], show_generations=not args.no_generations)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
