#!/usr/bin/env python3
"""
Deep diagnostic tool to understand why edits failed and provide recommendations
for improving edit success rates.
"""
import json
import argparse
from pathlib import Path
from collections import defaultdict
import numpy as np


class EditFailureDiagnostic:
    """Diagnose and provide insights on failed edits."""
    
    def __init__(self, result_file):
        with open(result_file, 'r') as f:
            self.data = json.load(f)
        self.case_id = self.data['case_id']
        self.rewrite = self.data['requested_rewrite']
        self.post = self.data['post']
    
    def get_diagnosis(self):
        """Generate comprehensive diagnosis of the edit."""
        diagnosis = {
            'case_id': self.case_id,
            'subject': self.rewrite.get('subject', 'N/A'),
            'relation_id': self.rewrite.get('relation_id', 'N/A'),
            'target_change': f"{self.rewrite['target_true']['str']} → {self.rewrite['target_new']['str']}",
            'issues': [],
            'recommendations': [],
            'probabilities': {}
        }
        
        # Analyze rewrite prompts
        rewrite_probs = self.post.get('rewrite_prompts_probs', [])
        rewrite_correct = self.post.get('rewrite_prompts_correct', [])
        
        if rewrite_probs:
            avg_new_prob = np.mean([p['target_new'] for p in rewrite_probs])
            avg_true_prob = np.mean([p['target_true'] for p in rewrite_probs])
            diagnosis['probabilities']['rewrite'] = {
                'target_new': float(avg_new_prob),
                'target_true': float(avg_true_prob),
                'ratio': float(avg_new_prob / avg_true_prob) if avg_true_prob > 0 else float('inf')
            }
            
            if not all(rewrite_correct):
                diagnosis['issues'].append({
                    'type': 'rewrite_failure',
                    'severity': 'critical',
                    'description': 'Model fails to produce target_new on original rewrite prompt',
                    'details': f"New prob: {avg_new_prob:.4f}, True prob: {avg_true_prob:.4f}"
                })
                diagnosis['recommendations'].append({
                    'priority': 1,
                    'action': 'increase_edit_strength',
                    'description': 'Increase v_lr (learning rate) or v_num_grad_steps to make edit stronger',
                    'suggested_params': {
                        'v_lr': 'Try 0.5 → 1.0',
                        'v_num_grad_steps': 'Try 25 → 50'
                    }
                })
            elif avg_new_prob < 1.0 and avg_true_prob > avg_new_prob:
                diagnosis['issues'].append({
                    'type': 'weak_edit',
                    'severity': 'medium',
                    'description': 'Edit succeeds but target_new probability is low',
                    'details': f"New prob: {avg_new_prob:.4f} < True prob: {avg_true_prob:.4f}"
                })
        
        # Analyze paraphrase prompts
        para_probs = self.post.get('paraphrase_prompts_probs', [])
        para_correct = self.post.get('paraphrase_prompts_correct', [])
        
        if para_probs:
            avg_new_prob = np.mean([p['target_new'] for p in para_probs])
            avg_true_prob = np.mean([p['target_true'] for p in para_probs])
            para_success_rate = sum(para_correct) / len(para_correct)
            
            diagnosis['probabilities']['paraphrase'] = {
                'target_new': float(avg_new_prob),
                'target_true': float(avg_true_prob),
                'ratio': float(avg_new_prob / avg_true_prob) if avg_true_prob > 0 else float('inf'),
                'success_rate': float(para_success_rate)
            }
            
            if para_success_rate < 1.0:
                diagnosis['issues'].append({
                    'type': 'paraphrase_generalization',
                    'severity': 'high' if para_success_rate < 0.5 else 'medium',
                    'description': 'Edit does not generalize well to paraphrased prompts',
                    'details': f"Success rate: {para_success_rate*100:.1f}%, New prob: {avg_new_prob:.4f}, True prob: {avg_true_prob:.4f}"
                })
                
                if avg_true_prob > avg_new_prob:
                    diagnosis['recommendations'].append({
                        'priority': 2,
                        'action': 'improve_generalization',
                        'description': 'Edit is too narrow and doesn\'t generalize. Consider:',
                        'suggested_params': {
                            'mom2_update_weight': 'Increase from 15000 to 20000 (broader edit)',
                            'kl_factor': 'Reduce from 0.0625 to 0.03 (less constraint)',
                            'layers': 'Try editing more layers or different layer range'
                        }
                    })
        
        # Analyze neighborhood prompts
        neigh_probs = self.post.get('neighborhood_prompts_probs', [])
        neigh_correct = self.post.get('neighborhood_prompts_correct', [])
        
        if neigh_probs:
            avg_new_prob = np.mean([p['target_new'] for p in neigh_probs])
            avg_true_prob = np.mean([p['target_true'] for p in neigh_probs])
            neigh_success_rate = sum(neigh_correct) / len(neigh_correct)
            
            diagnosis['probabilities']['neighborhood'] = {
                'target_new': float(avg_new_prob),
                'target_true': float(avg_true_prob),
                'ratio': float(avg_new_prob / avg_true_prob) if avg_true_prob > 0 else float('inf'),
                'success_rate': float(neigh_success_rate)
            }
            
            if neigh_success_rate > 0.3:
                diagnosis['issues'].append({
                    'type': 'overgeneralization',
                    'severity': 'low',
                    'description': 'Edit affects unrelated facts (high neighborhood success)',
                    'details': f"Neighborhood success rate: {neigh_success_rate*100:.1f}%"
                })
                diagnosis['recommendations'].append({
                    'priority': 3,
                    'action': 'reduce_overgeneralization',
                    'description': 'Edit is affecting unrelated facts',
                    'suggested_params': {
                        'mom2_update_weight': 'Decrease from 15000 to 10000 (more specific edit)',
                        'clamp_norm_factor': 'Reduce from 0.75 to 0.5',
                        'nullspace_threshold': 'Increase threshold to edit smaller subspace'
                    }
                })
        
        # Check generation quality if available
        if 'text' in self.post and self.post['text']:
            diagnosis['has_generation'] = True
            diagnosis['generation_samples'] = len(self.post['text'])
        
        # Overall assessment
        rewrite_success = all(rewrite_correct) if rewrite_correct else False
        para_success = all(para_correct) if para_correct else False
        
        if rewrite_success and para_success:
            diagnosis['overall_status'] = 'success'
        elif rewrite_success and not para_success:
            diagnosis['overall_status'] = 'partial_success_poor_generalization'
        else:
            diagnosis['overall_status'] = 'failure'
        
        return diagnosis


def analyze_failure_patterns(run_dir):
    """Analyze patterns in edit failures across a run."""
    run_path = Path(run_dir)
    edit_files = sorted(run_path.glob('*_edits-case_*.json'))
    
    all_diagnoses = []
    pattern_stats = defaultdict(lambda: defaultdict(int))
    relation_failures = defaultdict(int)
    
    for edit_file in edit_files:
        try:
            diag = EditFailureDiagnostic(edit_file)
            diagnosis = diag.get_diagnosis()
            all_diagnoses.append(diagnosis)
            
            # Count failure patterns
            if diagnosis['overall_status'] != 'success':
                for issue in diagnosis['issues']:
                    pattern_stats[issue['type']][issue['severity']] += 1
                
                # Track which relations fail most
                relation_failures[diagnosis['relation_id']] += 1
        except Exception as e:
            print(f"Error processing {edit_file}: {e}")
    
    return {
        'diagnoses': all_diagnoses,
        'pattern_stats': dict(pattern_stats),
        'relation_failures': dict(relation_failures)
    }


def print_diagnostic_summary(analysis):
    """Print summary of diagnostic analysis."""
    print(f"\n{'='*80}")
    print("EDIT FAILURE DIAGNOSTIC SUMMARY")
    print(f"{'='*80}")
    
    total = len(analysis['diagnoses'])
    success = sum(1 for d in analysis['diagnoses'] if d['overall_status'] == 'success')
    partial = sum(1 for d in analysis['diagnoses'] if 'partial' in d['overall_status'])
    failure = total - success - partial
    
    print(f"\nOverall Results:")
    print(f"  Success: {success}/{total} ({100*success/total:.1f}%)")
    print(f"  Partial Success (poor generalization): {partial}/{total} ({100*partial/total:.1f}%)")
    print(f"  Failure: {failure}/{total} ({100*failure/total:.1f}%)")
    
    print(f"\nFailure Pattern Statistics:")
    for pattern, severities in analysis['pattern_stats'].items():
        total_count = sum(severities.values())
        print(f"  {pattern}: {total_count}")
        for severity, count in severities.items():
            print(f"    - {severity}: {count}")
    
    print(f"\nRelations with Most Failures (Top 10):")
    sorted_relations = sorted(analysis['relation_failures'].items(), key=lambda x: x[1], reverse=True)
    for relation, count in sorted_relations[:10]:
        print(f"  {relation}: {count} failures")


def export_diagnostic_report(analysis, output_file):
    """Export detailed diagnostic report to JSON."""
    report = {
        'summary': {
            'total_edits': len(analysis['diagnoses']),
            'success_count': sum(1 for d in analysis['diagnoses'] if d['overall_status'] == 'success'),
            'partial_success_count': sum(1 for d in analysis['diagnoses'] if 'partial' in d['overall_status']),
            'failure_count': sum(1 for d in analysis['diagnoses'] if d['overall_status'] == 'failure'),
        },
        'pattern_statistics': analysis['pattern_stats'],
        'relation_failures': analysis['relation_failures'],
        'detailed_diagnoses': analysis['diagnoses']
    }
    
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\nDetailed diagnostic report saved to: {output_file}")


def create_recommendations_document(analysis, output_file):
    """Create a document with actionable recommendations."""
    # Collect all unique recommendations
    all_recommendations = []
    for diag in analysis['diagnoses']:
        if diag['overall_status'] != 'success':
            all_recommendations.extend(diag.get('recommendations', []))
    
    # Group by action type
    grouped_recs = defaultdict(list)
    for rec in all_recommendations:
        grouped_recs[rec['action']].append(rec)
    
    # Create recommendations document
    doc = {
        'title': 'AlphaEdit Failure Analysis: Recommendations for Improvement',
        'general_recommendations': [],
        'parameter_tuning_guide': {},
        'specific_cases': []
    }
    
    # General recommendations based on most common issues
    if analysis['pattern_stats'].get('paraphrase_generalization', {}):
        doc['general_recommendations'].append({
            'issue': 'Poor Paraphrase Generalization',
            'description': 'Most common failure mode: edits work on exact prompt but fail on paraphrases',
            'solutions': [
                'Increase mom2_update_weight from 15000 to 20000-25000 for broader edits',
                'Reduce kl_factor from 0.0625 to 0.03-0.04 to allow more flexibility',
                'Try editing additional layers or changing layer range',
                'Consider using different fact_token strategy (e.g., subject_last vs last)'
            ]
        })
    
    if analysis['pattern_stats'].get('weak_edit', {}):
        doc['general_recommendations'].append({
            'issue': 'Weak Edits',
            'description': 'Edits succeed but with low confidence in target_new',
            'solutions': [
                'Increase v_lr (learning rate) from 0.1 to 0.5 or 1.0',
                'Increase v_num_grad_steps from 25 to 50',
                'Adjust v_weight_decay to find better balance'
            ]
        })
    
    # Parameter tuning guide
    doc['parameter_tuning_guide'] = {
        'v_lr': {
            'description': 'Learning rate for computing the optimal edit',
            'default': 0.1,
            'low_values': 'Use 0.01-0.1 for subtle, localized edits',
            'high_values': 'Use 0.5-1.0 for stronger, more aggressive edits',
            'symptoms_to_increase': ['Rewrite fails', 'Target_new probability too low'],
            'symptoms_to_decrease': ['Overgeneralization', 'Model becomes incoherent']
        },
        'v_num_grad_steps': {
            'description': 'Number of optimization steps for computing edit',
            'default': 25,
            'low_values': 'Use 10-20 for faster, weaker edits',
            'high_values': 'Use 50-100 for stronger, more thorough edits',
            'symptoms_to_increase': ['Weak edits', 'Poor paraphrase generalization'],
            'symptoms_to_decrease': ['Overfitting', 'Computational cost too high']
        },
        'mom2_update_weight': {
            'description': 'Controls the breadth/specificity of the edit',
            'default': 15000,
            'low_values': 'Use 5000-10000 for very specific, narrow edits',
            'high_values': 'Use 20000-30000 for broader, more generalizable edits',
            'symptoms_to_increase': ['Poor paraphrase generalization', 'Edit too narrow'],
            'symptoms_to_decrease': ['Overgeneralization', 'Affects unrelated facts']
        },
        'kl_factor': {
            'description': 'Regularization to keep model close to original distribution',
            'default': 0.0625,
            'low_values': 'Use 0.01-0.03 for more flexible edits',
            'high_values': 'Use 0.1-0.2 for conservative edits',
            'symptoms_to_decrease': ['Poor generalization', 'Edit too constrained'],
            'symptoms_to_increase': ['Overgeneralization', 'Model drift']
        },
        'layers': {
            'description': 'Which layers to edit in the model',
            'default': '[4, 5, 6, 7, 8]',
            'notes': [
                'Earlier layers (0-10): More general, affects broader patterns',
                'Middle layers (10-20): Balance of specificity and generalization',
                'Later layers (20-32): More specific, affects particular facts',
                'Try different ranges if current edits fail to generalize'
            ]
        },
        'nullspace_threshold': {
            'description': 'Threshold for null space projection in AlphaEdit',
            'default': 0.02,
            'low_values': 'Use 0.01 for editing larger subspace',
            'high_values': 'Use 0.03-0.05 for editing smaller, more constrained subspace',
            'notes': 'AlphaEdit-specific parameter for preserving model behavior'
        }
    }
    
    # Add specific problematic cases
    failed_cases = [d for d in analysis['diagnoses'] if d['overall_status'] != 'success']
    doc['specific_cases'] = failed_cases[:50]  # Top 50 failures
    
    with open(output_file, 'w') as f:
        json.dump(doc, f, indent=2)
    
    print(f"Recommendations document saved to: {output_file}")
    
    # Also create a human-readable text version
    txt_file = output_file.replace('.json', '.txt')
    with open(txt_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("AlphaEdit Failure Analysis: Recommendations for Improvement\n")
        f.write("="*80 + "\n\n")
        
        f.write("GENERAL RECOMMENDATIONS\n")
        f.write("-"*80 + "\n")
        for rec in doc['general_recommendations']:
            f.write(f"\n{rec['issue']}\n")
            f.write(f"Description: {rec['description']}\n")
            f.write("Solutions:\n")
            for sol in rec['solutions']:
                f.write(f"  - {sol}\n")
        
        f.write("\n\n" + "="*80 + "\n")
        f.write("PARAMETER TUNING GUIDE\n")
        f.write("="*80 + "\n")
        for param, info in doc['parameter_tuning_guide'].items():
            f.write(f"\n{param}:\n")
            f.write(f"  Description: {info['description']}\n")
            if 'default' in info:
                f.write(f"  Default: {info['default']}\n")
            if 'low_values' in info:
                f.write(f"  Low values: {info['low_values']}\n")
            if 'high_values' in info:
                f.write(f"  High values: {info['high_values']}\n")
            if 'symptoms_to_increase' in info:
                f.write(f"  When to increase: {', '.join(info['symptoms_to_increase'])}\n")
            if 'symptoms_to_decrease' in info:
                f.write(f"  When to decrease: {', '.join(info['symptoms_to_decrease'])}\n")
            if 'notes' in info:
                if isinstance(info['notes'], list):
                    f.write(f"  Notes:\n")
                    for note in info['notes']:
                        f.write(f"    - {note}\n")
                else:
                    f.write(f"  Notes: {info['notes']}\n")
    
    print(f"Human-readable recommendations saved to: {txt_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Deep diagnostic tool for understanding edit failures'
    )
    parser.add_argument(
        '--run-dir',
        type=str,
        required=True,
        help='Path to the run directory'
    )
    parser.add_argument(
        '--output-report',
        type=str,
        default='diagnostic_report.json',
        help='Output file for detailed diagnostic report'
    )
    parser.add_argument(
        '--output-recommendations',
        type=str,
        default='recommendations.json',
        help='Output file for recommendations document'
    )
    
    args = parser.parse_args()
    
    print("Analyzing edit failures and generating diagnostics...")
    analysis = analyze_failure_patterns(args.run_dir)
    
    print_diagnostic_summary(analysis)
    export_diagnostic_report(analysis, args.output_report)
    create_recommendations_document(analysis, args.output_recommendations)
    
    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80)


if __name__ == '__main__':
    main()
