#!/usr/bin/env python3
"""
Predictive Model for Edit Success

Goal: Model edit success as a function of:
  - Model properties (architecture, size, training data)
  - Edit properties (relation type, fact entrenchment, linguistic complexity)
  - Dataset properties (temporal/static, paraphrase diversity)

This allows us to:
1. Predict which edits will fail before attempting
2. Understand WHY certain model/dataset combinations work better
3. Guide parameter selection based on model+edit characteristics
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple
import argparse

# Optional ML libraries (install if you want to build full predictive model)
try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler
    import pandas as pd
    HAS_ML = True
except ImportError:
    HAS_ML = False
    print("Warning: sklearn not installed. Install for full ML modeling: pip install scikit-learn pandas")


class EditSuccessPredictor:
    """
    Predict edit success based on model, dataset, and edit features
    """
    
    def __init__(self):
        self.features = []
        self.labels = []
        self.metadata = []
        
    def extract_model_features(self, model_name: str, model_info: Dict) -> Dict[str, float]:
        """
        Extract features about the model architecture and training
        
        Features to extract:
        - Model size (parameters)
        - Architecture family (GPT, Llama, etc.)
        - Training data characteristics (if available)
        - Layer count
        - Hidden dimension size
        """
        features = {}
        
        # Model size category (normalize to 0-1 scale)
        size_map = {
            'gpt2-xl': 1.5e9,
            'gpt-j-6b': 6e9,
            'gpt-j': 6e9,
            'llama3-8b': 8e9,
            'llama-3-8b': 8e9,
        }
        
        model_lower = model_name.lower()
        for key, size in size_map.items():
            if key in model_lower:
                features['model_size_bn'] = size / 1e9
                features['model_log_size'] = np.log10(size)
                break
        else:
            features['model_size_bn'] = 1.0  # Default
            features['model_log_size'] = 9.0
        
        # Architecture family (one-hot encoding)
        features['arch_gpt2'] = 1.0 if 'gpt2' in model_lower else 0.0
        features['arch_gptj'] = 1.0 if 'gpt-j' in model_lower or 'gptj' in model_lower else 0.0
        features['arch_llama'] = 1.0 if 'llama' in model_lower else 0.0
        
        # Training data era (affects what knowledge is baked in)
        # GPT-2: 2019, GPT-J: 2021, Llama3: 2024
        training_year_map = {
            'gpt2': 2019,
            'gpt-j': 2021,
            'llama3': 2024,
        }
        for key, year in training_year_map.items():
            if key in model_lower:
                features['training_year'] = year
                features['training_recency'] = year - 2019  # Normalize from GPT-2
                break
        else:
            features['training_year'] = 2020
            features['training_recency'] = 1
        
        return features
    
    def extract_edit_features(self, case: Dict) -> Dict[str, float]:
        """
        Extract features about the specific edit
        
        Features:
        - Relation type (encoded)
        - Subject frequency/popularity
        - Prompt length and complexity
        - Number of paraphrases
        - Paraphrase diversity
        """
        features = {}
        
        # Relation type (known hard relations)
        relation = case['requested_rewrite'].get('relation_id', '')
        
        # Hard relations get special indicator
        hard_relations = ['P413', 'P103', 'P1412', 'P30', 'P37']
        features['relation_is_hard'] = 1.0 if relation in hard_relations else 0.0
        
        # Specific relation indicators (top failing ones)
        features['relation_P413'] = 1.0 if relation == 'P413' else 0.0
        features['relation_P103'] = 1.0 if relation == 'P103' else 0.0
        features['relation_P1412'] = 1.0 if relation == 'P1412' else 0.0
        
        # Prompt characteristics
        prompt = case['requested_rewrite'].get('prompt', '')
        features['prompt_length'] = len(prompt.split())
        features['prompt_char_length'] = len(prompt)
        
        # Subject/object characteristics
        subject = case['requested_rewrite'].get('subject', '')
        target_new = case['requested_rewrite'].get('target_new', {}).get('str', '')
        target_true = case['requested_rewrite'].get('target_true', {}).get('str', '')
        
        features['subject_length'] = len(subject.split())
        features['target_new_length'] = len(target_new.split())
        features['target_true_length'] = len(target_true.split())
        
        # Paraphrase characteristics
        if 'paraphrase_prompts' in case:
            num_paraphrases = len(case['paraphrase_prompts'])
            features['num_paraphrases'] = num_paraphrases
            
            # Paraphrase length diversity (proxy for semantic diversity)
            para_lengths = [len(p.split()) for p in case['paraphrase_prompts']]
            features['paraphrase_length_std'] = np.std(para_lengths) if para_lengths else 0
            features['paraphrase_length_range'] = max(para_lengths) - min(para_lengths) if para_lengths else 0
        else:
            features['num_paraphrases'] = 0
            features['paraphrase_length_std'] = 0
            features['paraphrase_length_range'] = 0
        
        # Neighborhood characteristics
        if 'neighborhood_prompts' in case:
            features['num_neighbors'] = len(case['neighborhood_prompts'])
        else:
            features['num_neighbors'] = 0
        
        return features
    
    def extract_result_features(self, case: Dict) -> Dict[str, float]:
        """
        Extract features from the edit result (probabilities)
        These can be used as targets or as features for multi-stage prediction
        """
        features = {}
        post = case.get('post', {})
        
        # Rewrite probabilities
        if 'rewrite_prompts_probs' in post:
            probs = post['rewrite_prompts_probs']
            if probs:
                # Shape: [num_prompts, 2] where [:,0] is target_new, [:,1] is target_true
                probs_array = np.array(probs)
                if probs_array.ndim == 2 and probs_array.shape[1] >= 2:
                    target_new_probs = probs_array[:, 0]
                    target_true_probs = probs_array[:, 1]
                    
                    features['rewrite_target_new_mean'] = np.mean(target_new_probs)
                    features['rewrite_target_new_min'] = np.min(target_new_probs)
                    features['rewrite_prob_diff_mean'] = np.mean(target_new_probs - target_true_probs)
                    features['rewrite_prob_diff_min'] = np.min(target_new_probs - target_true_probs)
        
        # Paraphrase probabilities
        if 'paraphrase_prompts_probs' in post:
            probs = post['paraphrase_prompts_probs']
            if probs:
                probs_array = np.array(probs)
                if probs_array.ndim == 2 and probs_array.shape[1] >= 2:
                    target_new_probs = probs_array[:, 0]
                    target_true_probs = probs_array[:, 1]
                    
                    features['paraphrase_target_new_mean'] = np.mean(target_new_probs)
                    features['paraphrase_target_new_min'] = np.min(target_new_probs)
                    features['paraphrase_prob_diff_mean'] = np.mean(target_new_probs - target_true_probs)
                    features['paraphrase_prob_diff_min'] = np.min(target_new_probs - target_true_probs)
        
        # Neighborhood probabilities
        if 'neighborhood_prompts_probs' in post:
            probs = post['neighborhood_prompts_probs']
            if probs:
                probs_array = np.array(probs)
                if probs_array.ndim == 2 and probs_array.shape[1] >= 2:
                    features['neighborhood_prob_mean'] = np.mean(probs_array[:, 0])
                    features['neighborhood_prob_std'] = np.std(probs_array[:, 0])
        
        return features
    
    def extract_dataset_features(self, run_dir: Path) -> Dict[str, float]:
        """
        Extract features about the dataset and run configuration
        
        Features:
        - Dataset type (news vs counterfact)
        - Edit ordering (temporal vs random)
        - Hyperparameters
        """
        features = {}
        
        # Dataset type from path
        run_path_str = str(run_dir)
        features['dataset_news'] = 1.0 if 'news' in run_path_str else 0.0
        features['dataset_counterfact'] = 1.0 if 'news' not in run_path_str else 0.0
        
        # Edit ordering
        features['ordering_temporal'] = 1.0 if 'temporal' in run_path_str else 0.0
        features['ordering_random'] = 1.0 if 'random' in run_path_str else 0.0
        
        # Load hyperparameters if available
        params_file = run_dir / 'params.json'
        if params_file.exists():
            with open(params_file, 'r') as f:
                params = json.load(f)
            
            # Key hyperparameters
            features['mom2_update_weight'] = params.get('mom2_update_weight', 15000)
            features['kl_factor'] = params.get('kl_factor', 0.0625)
            features['v_lr'] = params.get('v_lr', 0.5)
            features['v_num_grad_steps'] = params.get('v_num_grad_steps', 25)
            features['clamp_norm_factor'] = params.get('clamp_norm_factor', 0.75)
            features['nullspace_threshold'] = params.get('nullspace_threshold', 0.02)
        else:
            # Defaults
            features['mom2_update_weight'] = 15000
            features['kl_factor'] = 0.0625
            features['v_lr'] = 0.5
            features['v_num_grad_steps'] = 25
            features['clamp_norm_factor'] = 0.75
            features['nullspace_threshold'] = 0.02
        
        return features
    
    def compute_edit_success(self, case: Dict) -> Dict[str, int]:
        """
        Compute multiple types of success for multi-task prediction
        """
        post = case.get('post', {})
        
        success = {}
        
        # Rewrite success
        if 'rewrite_prompts_correct' in post:
            rewrite_correct = post['rewrite_prompts_correct']
            success['rewrite_success'] = 1 if all(rewrite_correct) else 0
        else:
            success['rewrite_success'] = 0
        
        # Paraphrase success
        if 'paraphrase_prompts_correct' in post:
            para_correct = post['paraphrase_prompts_correct']
            success['paraphrase_success'] = 1 if all(para_correct) else 0
        else:
            success['paraphrase_success'] = 0
        
        # Overall success
        success['overall_success'] = success['rewrite_success'] and success['paraphrase_success']
        
        # Neighborhood preservation
        if 'neighborhood_prompts_correct' in post:
            neigh_correct = post['neighborhood_prompts_correct']
            success['neighborhood_preservation'] = 1 if all(neigh_correct) else 0
            success['neighborhood_score'] = np.mean(neigh_correct) if neigh_correct else 0
        else:
            success['neighborhood_preservation'] = 0
            success['neighborhood_score'] = 0
        
        return success
    
    def load_run_data(self, run_dir: Path, model_name: str = None):
        """
        Load all data from a run directory
        """
        run_dir = Path(run_dir)
        
        # Infer model name from path if not provided
        if model_name is None:
            run_path_str = str(run_dir)
            if 'llama3' in run_path_str.lower() or 'llama-3' in run_path_str.lower():
                model_name = 'llama3-8b'
            elif 'gpt-j' in run_path_str.lower() or 'gptj' in run_path_str.lower():
                model_name = 'gpt-j-6b'
            elif 'gpt2-xl' in run_path_str.lower():
                model_name = 'gpt2-xl'
            else:
                model_name = 'unknown'
        
        print(f"Loading run: {run_dir}")
        print(f"Model: {model_name}")
        
        # Get dataset-level features
        dataset_features = self.extract_dataset_features(run_dir)
        
        # Get model-level features
        model_features = self.extract_model_features(model_name, {})
        
        # Load all cases
        cases_dir = run_dir / 'cases'
        if not cases_dir.exists():
            print(f"Warning: No cases directory in {run_dir}")
            return
        
        case_files = sorted(cases_dir.glob('*.json'))
        print(f"Found {len(case_files)} cases")
        
        for case_file in case_files:
            with open(case_file, 'r') as f:
                case = json.load(f)
            
            # Extract all features
            edit_features = self.extract_edit_features(case)
            result_features = self.extract_result_features(case)
            success = self.compute_edit_success(case)
            
            # Combine all features
            all_features = {
                **model_features,
                **dataset_features,
                **edit_features,
                **result_features,
            }
            
            self.features.append(all_features)
            self.labels.append(success)
            self.metadata.append({
                'run_dir': str(run_dir),
                'case_id': case['case_id'],
                'relation': case['requested_rewrite'].get('relation_id', ''),
                'model': model_name,
            })
    
    def load_multiple_runs(self, run_specs: List[Tuple[str, str]]):
        """
        Load multiple runs with (run_dir, model_name) tuples
        """
        for run_dir, model_name in run_specs:
            self.load_run_data(Path(run_dir), model_name)
    
    def get_dataframe(self) -> 'pd.DataFrame':
        """
        Convert to pandas DataFrame for analysis
        """
        if not HAS_ML:
            print("Error: pandas not installed")
            return None
        
        # Convert features to DataFrame
        df_features = pd.DataFrame(self.features)
        df_labels = pd.DataFrame(self.labels)
        df_metadata = pd.DataFrame(self.metadata)
        
        # Combine
        df = pd.concat([df_metadata, df_features, df_labels], axis=1)
        
        return df
    
    def build_predictor(self, target: str = 'overall_success', feature_type: str = 'predit'):
        """
        Build a predictive model
        
        Args:
            target: What to predict ('overall_success', 'paraphrase_success', etc.)
            feature_type: 'predict' (before edit) or 'diagnose' (after edit, includes probs)
        """
        if not HAS_ML:
            print("Error: scikit-learn not installed. Install with: pip install scikit-learn")
            return None
        
        df = self.get_dataframe()
        
        # Select features based on type
        if feature_type == 'predict':
            # Features available BEFORE editing (for prediction)
            feature_cols = [col for col in df.columns if not any(
                x in col for x in ['_prob', 'rewrite_target', 'paraphrase_target', 'neighborhood_prob',
                                   'success', 'preservation', 'score', 'case_id', 'run_dir', 'relation_id']
            )]
        else:
            # All features including probability results (for diagnosis)
            feature_cols = [col for col in df.columns if not any(
                x in col for x in ['success', 'preservation', 'score', 'case_id', 'run_dir', 'relation_id']
            )]
        
        X = df[feature_cols].fillna(0)
        y = df[target]
        
        print(f"\nBuilding {feature_type} model for {target}")
        print(f"Features: {len(feature_cols)}")
        print(f"Samples: {len(X)}")
        print(f"Positive rate: {y.mean():.1%}")
        
        # Try multiple models
        models = {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42),
        }
        
        results = {}
        for name, model in models.items():
            # Cross-validation
            scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
            results[name] = {
                'mean': scores.mean(),
                'std': scores.std(),
                'scores': scores
            }
            print(f"{name}: {scores.mean():.3f} (+/- {scores.std():.3f})")
        
        # Train best model on all data
        best_model_name = max(results.keys(), key=lambda k: results[k]['mean'])
        best_model = models[best_model_name]
        best_model.fit(X, y)
        
        # Feature importance
        if hasattr(best_model, 'feature_importances_'):
            importances = best_model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            print(f"\nTop 10 features for {target}:")
            for i in range(min(10, len(indices))):
                idx = indices[i]
                print(f"{i+1}. {feature_cols[idx]}: {importances[idx]:.4f}")
        
        return {
            'model': best_model,
            'feature_cols': feature_cols,
            'results': results,
            'X': X,
            'y': y,
        }
    
    def analyze_feature_interactions(self):
        """
        Analyze interactions between model, dataset, and edit features
        """
        df = self.get_dataframe()
        
        print("\n=== Feature Interaction Analysis ===\n")
        
        # 1. Model architecture effects
        print("1. Model Architecture Effects:")
        for model_col in ['arch_gpt2', 'arch_gptj', 'arch_llama']:
            if model_col in df.columns:
                model_data = df[df[model_col] == 1]
                if len(model_data) > 0:
                    print(f"   {model_col}: {model_data['overall_success'].mean():.1%} success ({len(model_data)} edits)")
        
        # 2. Dataset effects
        print("\n2. Dataset Effects:")
        for dataset_col in ['dataset_news', 'dataset_counterfact']:
            if dataset_col in df.columns:
                dataset_data = df[df[dataset_col] == 1]
                if len(dataset_data) > 0:
                    print(f"   {dataset_col}: {dataset_data['overall_success'].mean():.1%} success ({len(dataset_data)} edits)")
        
        # 3. Relation difficulty
        print("\n3. Relation Difficulty:")
        for rel_col in ['relation_P413', 'relation_P103', 'relation_P1412']:
            if rel_col in df.columns:
                rel_data = df[df[rel_col] == 1]
                if len(rel_data) > 0:
                    print(f"   {rel_col}: {rel_data['overall_success'].mean():.1%} success ({len(rel_data)} edits)")
        
        # 4. Hyperparameter effects (correlation with success)
        print("\n4. Hyperparameter Correlations with Success:")
        hyperparam_cols = ['mom2_update_weight', 'kl_factor', 'v_lr', 'v_num_grad_steps', 
                          'clamp_norm_factor', 'nullspace_threshold']
        
        for col in hyperparam_cols:
            if col in df.columns:
                corr = df[col].corr(df['overall_success'])
                print(f"   {col}: {corr:.3f}")
        
        # 5. Model-Dataset interaction
        print("\n5. Model-Dataset Interactions:")
        for model_col in ['arch_gpt2', 'arch_gptj', 'arch_llama']:
            if model_col not in df.columns:
                continue
            model_data = df[df[model_col] == 1]
            if len(model_data) == 0:
                continue
            
            print(f"\n   {model_col}:")
            for dataset_col in ['dataset_news', 'dataset_counterfact']:
                if dataset_col not in model_data.columns:
                    continue
                subset = model_data[model_data[dataset_col] == 1]
                if len(subset) > 0:
                    print(f"      + {dataset_col}: {subset['overall_success'].mean():.1%} ({len(subset)} edits)")
        
        # 6. Model-Relation interaction
        print("\n6. Model-Relation Interactions (Hard Relations):")
        hard_relations = df[df['relation_is_hard'] == 1]
        if len(hard_relations) > 0:
            for model_col in ['arch_gpt2', 'arch_gptj', 'arch_llama']:
                if model_col in hard_relations.columns:
                    model_hard = hard_relations[hard_relations[model_col] == 1]
                    if len(model_hard) > 0:
                        print(f"   {model_col} on hard relations: {model_hard['overall_success'].mean():.1%} ({len(model_hard)} edits)")


def main():
    parser = argparse.ArgumentParser(description='Build predictive model for edit success')
    parser.add_argument('--runs', nargs='+', help='Run directories to analyze')
    parser.add_argument('--output', default='predictive_model_results.json', help='Output file')
    parser.add_argument('--analyze-only', action='store_true', help='Only analyze, don\'t build ML models')
    
    args = parser.parse_args()
    
    # Default runs if none specified
    if not args.runs:
        args.runs = [
            'results/AlphaEdit/run_050:unknown',
            'news_alphaedit_eval/runs/temporal/llama3/run_030:llama3-8b',
            'news_alphaedit_eval/runs/temporal/gpt-j/run_040:gpt-j-6b',
            'news_alphaedit_eval/runs/temporal/gpt2-xl/run_031:gpt2-xl',
            'news_alphaedit_eval/runs/random/gpt2-xl/run_030:gpt2-xl',
        ]
    
    # Parse run specifications
    run_specs = []
    for run_spec in args.runs:
        if ':' in run_spec:
            run_dir, model = run_spec.split(':', 1)
        else:
            run_dir = run_spec
            model = None
        
        if Path(run_dir).exists():
            run_specs.append((run_dir, model))
        else:
            print(f"Warning: Run directory not found: {run_dir}")
    
    if not run_specs:
        print("Error: No valid run directories found")
        return
    
    # Load data
    predictor = EditSuccessPredictor()
    predictor.load_multiple_runs(run_specs)
    
    print(f"\nLoaded {len(predictor.features)} edits from {len(run_specs)} runs")
    
    # Analyze feature interactions
    predictor.analyze_feature_interactions()
    
    if args.analyze_only or not HAS_ML:
        return
    
    # Build predictive models
    print("\n=== Building Predictive Models ===\n")
    
    # 1. Predict overall success (BEFORE editing)
    print("\n--- Predicting Overall Success (Pre-Edit Features) ---")
    predictor.build_predictor(target='overall_success', feature_type='predict')
    
    # 2. Predict paraphrase success (BEFORE editing)
    print("\n--- Predicting Paraphrase Success (Pre-Edit Features) ---")
    predictor.build_predictor(target='paraphrase_success', feature_type='predict')
    
    # 3. Diagnose using post-edit probabilities
    print("\n--- Diagnosing Failure (Post-Edit Features) ---")
    predictor.build_predictor(target='overall_success', feature_type='diagnose')
    
    print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()
