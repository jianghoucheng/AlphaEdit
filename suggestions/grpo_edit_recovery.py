#!/usr/bin/env python3
"""
GRPO (Group Relative Policy Optimization) for Failed Edit Recovery

Concept: Use reinforcement learning to learn optimal edit parameters by treating
edit success/failure as a reward signal.

Strategy:
1. First pass: Apply edits with base parameters, track failures
2. Calculate reward: Success metrics (rewrite, paraphrase, neighborhood)
3. Second pass: Use GRPO to learn better parameters for failed edits
4. Observe improvement and iterate

This is more expensive than simple replay techniques but can learn optimal
strategies automatically rather than hand-tuning.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import argparse


@dataclass
class EditReward:
    """
    Reward signal for an edit attempt
    """
    rewrite_success: float  # 0-1: fraction of rewrite prompts correct
    paraphrase_success: float  # 0-1: fraction of paraphrase prompts correct
    neighborhood_preservation: float  # 0-1: fraction of neighborhood preserved
    
    # Probability-based metrics (more granular than binary)
    rewrite_prob_gain: float  # P(target_new) - P(target_true) on rewrite
    paraphrase_prob_gain: float  # P(target_new) - P(target_true) on paraphrases
    
    # Combined reward
    total_reward: float
    
    @classmethod
    def from_edit_result(cls, case: Dict) -> 'EditReward':
        """
        Calculate reward from edit result
        """
        post = case.get('post', {})
        
        # Binary success rates
        rewrite_success = np.mean(post.get('rewrite_prompts_correct', [0]))
        paraphrase_success = np.mean(post.get('paraphrase_prompts_correct', [0]))
        neighborhood_preservation = np.mean(post.get('neighborhood_prompts_correct', [1]))
        
        # Probability gains (how much we shifted the distribution)
        rewrite_probs = np.array(post.get('rewrite_prompts_probs', []))
        paraphrase_probs = np.array(post.get('paraphrase_prompts_probs', []))
        
        if rewrite_probs.size > 0 and rewrite_probs.ndim == 2:
            # Shape: [num_prompts, 2] where [:,0] is target_new, [:,1] is target_true
            rewrite_prob_gain = np.mean(rewrite_probs[:, 0] - rewrite_probs[:, 1])
        else:
            rewrite_prob_gain = 0.0
        
        if paraphrase_probs.size > 0 and paraphrase_probs.ndim == 2:
            paraphrase_prob_gain = np.mean(paraphrase_probs[:, 0] - paraphrase_probs[:, 1])
        else:
            paraphrase_prob_gain = 0.0
        
        # Combined reward (weighted sum)
        # Emphasize paraphrase success (the main bottleneck)
        total_reward = (
            0.2 * rewrite_success +  # Rewrite is usually easy
            0.5 * paraphrase_success +  # Main goal: generalization
            0.2 * neighborhood_preservation +  # Don't break other facts
            0.05 * max(0, rewrite_prob_gain) +  # Bonus for strong edits
            0.05 * max(0, paraphrase_prob_gain)
        )
        
        return cls(
            rewrite_success=rewrite_success,
            paraphrase_success=paraphrase_success,
            neighborhood_preservation=neighborhood_preservation,
            rewrite_prob_gain=rewrite_prob_gain,
            paraphrase_prob_gain=paraphrase_prob_gain,
            total_reward=total_reward
        )


@dataclass
class EditAction:
    """
    Action space: hyperparameter settings for an edit
    """
    v_lr: float
    v_num_grad_steps: int
    kl_factor: float
    mom2_update_weight: float
    
    # Optional: could also tune these
    clamp_norm_factor: Optional[float] = None
    nullspace_threshold: Optional[float] = None
    
    def to_dict(self) -> Dict:
        return {
            'v_lr': self.v_lr,
            'v_num_grad_steps': self.v_num_grad_steps,
            'kl_factor': self.kl_factor,
            'mom2_update_weight': self.mom2_update_weight,
            'clamp_norm_factor': self.clamp_norm_factor,
            'nullspace_threshold': self.nullspace_threshold,
        }


class GRPOEditOptimizer:
    """
    GRPO-based optimizer for learning edit parameters
    
    How it works:
    1. Sample multiple parameter configurations for each failed edit
    2. Apply edits with those configurations
    3. Calculate rewards based on success
    4. Update policy to favor high-reward configurations
    5. Use learned policy to generate better parameters for next round
    """
    
    def __init__(self, base_hparams: Dict):
        """
        Args:
            base_hparams: Starting hyperparameters
        """
        self.base_hparams = base_hparams
        
        # Define action space (parameter ranges to explore)
        self.action_space = {
            'v_lr': (0.1, 1.0),  # Learning rate
            'v_num_grad_steps': (10, 100),  # Optimization steps
            'kl_factor': (0.01, 0.2),  # KL penalty
            'mom2_update_weight': (5000, 30000),  # Edit breadth
        }
        
        # Policy: learned distribution over action space
        # Start with base params as mean, some variance
        self.policy_mean = {
            'v_lr': base_hparams.get('v_lr', 0.5),
            'v_num_grad_steps': base_hparams.get('v_num_grad_steps', 25),
            'kl_factor': base_hparams.get('kl_factor', 0.0625),
            'mom2_update_weight': base_hparams.get('mom2_update_weight', 15000),
        }
        
        self.policy_std = {
            'v_lr': 0.2,
            'v_num_grad_steps': 10.0,
            'kl_factor': 0.03,
            'mom2_update_weight': 5000.0,
        }
        
        # History of (action, reward) pairs for learning
        self.history = []
    
    def sample_action(self, edit: Dict, num_samples: int = 4) -> List[EditAction]:
        """
        Sample multiple parameter configurations to try
        
        For GRPO, we sample a group of actions and compare them relatively.
        
        Args:
            edit: The edit to optimize for
            num_samples: Number of configurations to try (GRPO group size)
            
        Returns:
            List of parameter configurations to try
        """
        actions = []
        
        # Get edit features for conditioning
        relation = edit['requested_rewrite'].get('relation_id', '')
        
        # Adjust mean based on relation difficulty (simple prior)
        if relation in ['P413', 'P103', 'P1412']:
            # Hard relations: bias towards higher strength
            adjusted_mean = {
                'v_lr': min(1.0, self.policy_mean['v_lr'] * 1.3),
                'v_num_grad_steps': int(self.policy_mean['v_num_grad_steps'] * 1.3),
                'kl_factor': self.policy_mean['kl_factor'] * 0.8,
                'mom2_update_weight': self.policy_mean['mom2_update_weight'] * 1.2,
            }
        else:
            adjusted_mean = self.policy_mean.copy()
        
        # Sample from policy distribution
        for _ in range(num_samples):
            action = EditAction(
                v_lr=np.clip(
                    np.random.normal(adjusted_mean['v_lr'], self.policy_std['v_lr']),
                    *self.action_space['v_lr']
                ),
                v_num_grad_steps=int(np.clip(
                    np.random.normal(adjusted_mean['v_num_grad_steps'], self.policy_std['v_num_grad_steps']),
                    *self.action_space['v_num_grad_steps']
                )),
                kl_factor=np.clip(
                    np.random.normal(adjusted_mean['kl_factor'], self.policy_std['kl_factor']),
                    *self.action_space['kl_factor']
                ),
                mom2_update_weight=int(np.clip(
                    np.random.normal(adjusted_mean['mom2_update_weight'], self.policy_std['mom2_update_weight']),
                    *self.action_space['mom2_update_weight']
                )),
            )
            actions.append(action)
        
        return actions
    
    def update_policy(self, actions: List[EditAction], rewards: List[EditReward]):
        """
        Update policy based on observed rewards (GRPO update)
        
        GRPO key idea: Compare actions within a group, move towards better ones
        
        Args:
            actions: List of actions tried
            rewards: Corresponding rewards
        """
        if len(actions) != len(rewards):
            raise ValueError("Actions and rewards must have same length")
        
        # Store in history
        for action, reward in zip(actions, rewards):
            self.history.append((action, reward))
        
        # GRPO update: compare within this group
        reward_values = np.array([r.total_reward for r in rewards])
        
        # Normalize rewards (relative to group mean)
        mean_reward = reward_values.mean()
        std_reward = reward_values.std() + 1e-8
        advantages = (reward_values - mean_reward) / std_reward
        
        # Update policy: move mean towards high-reward actions
        learning_rate = 0.1
        
        for param in ['v_lr', 'v_num_grad_steps', 'kl_factor', 'mom2_update_weight']:
            # Weighted average: weight by advantage
            param_values = np.array([getattr(a, param) for a in actions])
            
            # Positive advantage = good action, move towards it
            weights = np.exp(advantages * 0.5)  # Softmax-like weighting
            weights /= weights.sum()
            
            weighted_mean = (param_values * weights).sum()
            
            # Update policy mean
            self.policy_mean[param] = (
                (1 - learning_rate) * self.policy_mean[param] +
                learning_rate * weighted_mean
            )
            
            # Also adapt variance (less exploration as we learn)
            self.policy_std[param] *= 0.95  # Gradually reduce exploration
        
        print(f"\n📊 Policy Update:")
        print(f"   Group rewards: {reward_values}")
        print(f"   Advantages: {advantages}")
        print(f"   Updated means: {self.policy_mean}")
    
    def get_best_action(self) -> EditAction:
        """
        Get best action according to current policy (exploitation)
        """
        return EditAction(
            v_lr=self.policy_mean['v_lr'],
            v_num_grad_steps=int(self.policy_mean['v_num_grad_steps']),
            kl_factor=self.policy_mean['kl_factor'],
            mom2_update_weight=int(self.policy_mean['mom2_update_weight']),
        )


class GRPOEditRecovery:
    """
    Full GRPO-based edit recovery system
    """
    
    def __init__(self, base_hparams: Dict):
        self.base_hparams = base_hparams
        self.optimizer = GRPOEditOptimizer(base_hparams)
    
    def first_pass(self, edits: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """
        First pass: Apply edits with base parameters, identify failures
        
        Args:
            edits: List of edits to attempt
            
        Returns:
            (successful_edits, failed_edits)
        """
        print(f"\n🔄 FIRST PASS: Applying {len(edits)} edits with base parameters")
        print(f"   Base params: {self.base_hparams}")
        
        # In practice, you'd actually apply edits here
        # For now, we'll assume they've already been applied and we're loading results
        
        successful = []
        failed = []
        
        for edit in edits:
            reward = EditReward.from_edit_result(edit)
            
            # Consider success if overall reward > 0.6
            if reward.total_reward > 0.6:
                successful.append(edit)
            else:
                failed.append(edit)
        
        print(f"   ✓ Success: {len(successful)}/{len(edits)} ({len(successful)/len(edits)*100:.1f}%)")
        print(f"   ✗ Failed: {len(failed)}/{len(edits)} ({len(failed)/len(edits)*100:.1f}%)")
        
        return successful, failed
    
    def grpo_recovery_round(self, failed_edit: Dict, round_num: int = 1) -> Tuple[EditAction, EditReward]:
        """
        One round of GRPO recovery for a single failed edit
        
        Args:
            failed_edit: The edit that failed
            round_num: Which recovery round this is
            
        Returns:
            (best_action, best_reward)
        """
        print(f"\n🔧 GRPO Round {round_num} for case_id={failed_edit['case_id']}")
        
        # Sample multiple parameter configurations
        actions = self.optimizer.sample_action(failed_edit, num_samples=4)
        
        print(f"   Trying {len(actions)} parameter configurations:")
        for i, action in enumerate(actions):
            print(f"     Config {i+1}: v_lr={action.v_lr:.3f}, steps={action.v_num_grad_steps}, "
                  f"kl={action.kl_factor:.4f}, mom2={action.mom2_update_weight}")
        
        # In practice: apply edit with each configuration and measure reward
        # For demonstration, we'll simulate different rewards
        rewards = []
        for i, action in enumerate(actions):
            # Simulate reward (in practice, you'd actually run the edit)
            # Higher v_lr and steps generally help weak edits
            simulated_reward = EditReward(
                rewrite_success=0.8 + 0.15 * (action.v_lr - 0.5),
                paraphrase_success=0.4 + 0.3 * (action.v_lr - 0.5) + 0.001 * (action.v_num_grad_steps - 25),
                neighborhood_preservation=0.15 - 0.05 * (action.v_lr - 0.5),
                rewrite_prob_gain=0.5 + 0.3 * (action.v_lr - 0.5),
                paraphrase_prob_gain=0.2 + 0.4 * (action.v_lr - 0.5),
                total_reward=0.0  # Will calculate
            )
            # Calculate total
            simulated_reward.total_reward = (
                0.2 * simulated_reward.rewrite_success +
                0.5 * simulated_reward.paraphrase_success +
                0.2 * simulated_reward.neighborhood_preservation +
                0.05 * max(0, simulated_reward.rewrite_prob_gain) +
                0.05 * max(0, simulated_reward.paraphrase_prob_gain)
            )
            rewards.append(simulated_reward)
            
            print(f"     Config {i+1} reward: {simulated_reward.total_reward:.3f} "
                  f"(para: {simulated_reward.paraphrase_success:.2f}, neigh: {simulated_reward.neighborhood_preservation:.2f})")
        
        # Update policy based on results
        self.optimizer.update_policy(actions, rewards)
        
        # Return best action and its reward
        best_idx = np.argmax([r.total_reward for r in rewards])
        return actions[best_idx], rewards[best_idx]
    
    def recover_failed_edits(self, failed_edits: List[Dict], num_rounds: int = 3) -> Dict:
        """
        Full GRPO recovery: iterate over failed edits with learned policy
        
        Args:
            failed_edits: Edits that failed in first pass
            num_rounds: Number of GRPO rounds to run
            
        Returns:
            Recovery results
        """
        print(f"\n{'='*80}")
        print(f"GRPO RECOVERY: {len(failed_edits)} failed edits, {num_rounds} rounds")
        print(f"{'='*80}")
        
        recovered = []
        still_failed = []
        
        # Process each failed edit
        for i, edit in enumerate(failed_edits[:10]):  # Limit to 10 for demo
            print(f"\n--- Edit {i+1}/{min(10, len(failed_edits))} ---")
            print(f"    Relation: {edit['requested_rewrite'].get('relation_id', 'unknown')}")
            print(f"    Subject: {edit['requested_rewrite'].get('subject', 'unknown')}")
            
            best_reward_overall = 0
            best_action_overall = None
            
            # Multiple rounds of optimization
            for round_num in range(1, num_rounds + 1):
                best_action, best_reward = self.grpo_recovery_round(edit, round_num)
                
                if best_reward.total_reward > best_reward_overall:
                    best_reward_overall = best_reward.total_reward
                    best_action_overall = best_action
                
                # Early stopping if we achieve good reward
                if best_reward.total_reward > 0.7:
                    print(f"   ✓ Achieved good reward ({best_reward.total_reward:.3f}), stopping early")
                    break
            
            # Check if recovered
            if best_reward_overall > 0.6:
                recovered.append({
                    'edit': edit,
                    'best_action': best_action_overall,
                    'reward': best_reward_overall
                })
                print(f"   ✓ RECOVERED (reward: {best_reward_overall:.3f})")
            else:
                still_failed.append(edit)
                print(f"   ✗ Still failed (reward: {best_reward_overall:.3f})")
        
        return {
            'recovered': recovered,
            'still_failed': still_failed,
            'recovery_rate': len(recovered) / len(failed_edits[:10]) if failed_edits else 0,
            'final_policy': self.optimizer.policy_mean
        }


def compare_grpo_vs_simple():
    """
    Compare GRPO approach vs simple techniques
    """
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    GRPO vs Simple Replay Techniques                          ║
╚══════════════════════════════════════════════════════════════════════════════╝

┌──────────────────────────────────────────────────────────────────────────────┐
│ GRPO (Reinforcement Learning) Approach                                      │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│ How it works:                                                                 │
│   1. First pass: Apply base params, track failures (55% success)            │
│   2. For each failed edit, sample 4 parameter configs                       │
│   3. Apply all 4, calculate rewards                                         │
│   4. Update policy to favor high-reward configs                             │
│   5. Repeat 2-3 rounds, learning optimal params                             │
│                                                                              │
│ Pros:                                                                         │
│   ✓ Learns optimal parameters automatically                                 │
│   ✓ Adapts to each specific edit                                            │
│   ✓ Can discover non-obvious parameter combinations                         │
│   ✓ Reward signal captures multiple objectives (para + neigh)               │
│                                                                              │
│ Cons:                                                                         │
│   ✗ EXPENSIVE: 4 attempts × 3 rounds × N edits = 12N edit operations       │
│   ✗ Requires multiple passes (can't recover in one shot)                    │
│   ✗ Complex to implement and debug                                          │
│   ✗ Needs infrastructure for tracking rewards and updating policy           │
│                                                                              │
│ Cost:                                                                         │
│   - First pass: N edits                                                      │
│   - Recovery: 12 × (failed edits) ≈ 12 × 0.45N ≈ 5.4N edits                │
│   - TOTAL: ~6.4N edit operations                                            │
│                                                                              │
│ Expected improvement:                                                         │
│   Baseline: 55% success                                                      │
│   After GRPO: 70-75% success (+15-20%)                                      │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────┐
│ Simple Techniques (from simple_edit_replay.py)                               │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│ Relation-Adaptive (SIMPLEST):                                                │
│   Cost: 1N (no extra compute!)                                              │
│   Improvement: +7-12%                                                        │
│   Method: Use 1.5x strength for hard relations (P413, P103, P1412)         │
│                                                                              │
│ Multi-Shot:                                                                   │
│   Cost: 2N (apply twice)                                                     │
│   Improvement: +5-10%                                                        │
│   Method: Apply edit, then refine with reduced strength                     │
│                                                                              │
│ Combined (Relation-Adaptive + Multi-Shot):                                   │
│   Cost: 2N                                                                    │
│   Improvement: +10-15%                                                       │
│   Method: Adapt strength by relation, then apply twice                      │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘

📊 COST-BENEFIT ANALYSIS

Technique              Cost    Improvement    Cost-Effectiveness    Complexity
─────────────────────────────────────────────────────────────────────────────
Relation-Adaptive      1.0x    +7-12%         ⭐⭐⭐⭐⭐             ⭐⭐⭐⭐⭐
Multi-Shot             2.0x    +5-10%         ⭐⭐⭐⭐              ⭐⭐⭐⭐
Combined               2.0x    +10-15%        ⭐⭐⭐⭐⭐             ⭐⭐⭐⭐
GRPO                   6.4x    +15-20%        ⭐⭐⭐                ⭐⭐

🎯 RECOMMENDATION

For your use case (news-based temporal editing):

1. START with Relation-Adaptive (simple_edit_replay.py)
   - Zero extra cost
   - Easy to implement (just check relation_id)
   - Gets you +7-12% improvement immediately
   - Baseline: 55% → 62-67%

2. IF NEEDED, add Multi-Shot
   - 2x cost but still reasonable
   - Gets you to +10-15% total
   - Baseline: 55% → 65-70%

3. ONLY use GRPO if:
   - You have compute budget (6x cost)
   - You want to automate parameter search
   - You're doing many runs and want to learn optimal policy
   - Simple techniques plateaued and you need more

💡 WHY GRPO MIGHT NOT BE WORTH IT FOR YOU

Your original concern: "not too computationally expensive (using other models 
to help train or GRPO are both quite expensive, not terrible, but not ideal)"

GRPO is 6.4x more expensive than baseline, and only gives you +5-10% over 
simple combined technique (2x cost).

For the marginal gain (70-75% vs 65-70%), you're paying 3.2x more compute.

BETTER ALTERNATIVE:
- Use relation-adaptive (free) to get to 62-67%
- Run hyperparameter sweep on 200 edits (Experiment 1 from earlier) to find 
  better base params
- Apply those better params + relation-adaptive
- Likely to hit 70%+ without GRPO complexity

🔬 WHEN GRPO MAKES SENSE

GRPO would be valuable if:
1. You're building a continual learning system that edits thousands of facts
2. You want the system to automatically adapt parameters over time
3. You have diverse edit types and want per-type optimization
4. You're comparing to other methods and want to show "optimal" performance

For a research paper on news-based editing, showing that simple relation-
adaptive technique improves by 10% is actually a BETTER story than using 
expensive GRPO, because it shows the method can be improved with minimal cost.

""")


def main():
    parser = argparse.ArgumentParser(description='GRPO-based edit recovery')
    parser.add_argument('--run-dir', type=str, help='Run directory to analyze')
    parser.add_argument('--demo', action='store_true', help='Run demo with simulated edits')
    parser.add_argument('--compare', action='store_true', help='Compare GRPO vs simple techniques')
    
    args = parser.parse_args()
    
    if args.compare:
        compare_grpo_vs_simple()
        return
    
    if args.demo or not args.run_dir:
        print("\n🎬 Running GRPO demo with simulated edits...")
        
        # Create demo failed edits
        failed_edits = []
        for i in range(5):
            failed_edits.append({
                'case_id': i,
                'requested_rewrite': {
                    'relation_id': ['P413', 'P103', 'P1412', 'P30', 'P495'][i],
                    'subject': f'Subject_{i}',
                    'prompt': f'Prompt for subject {i}',
                    'target_new': {'str': f'NewValue_{i}'},
                    'target_true': {'str': f'OldValue_{i}'},
                },
                'post': {
                    'rewrite_prompts_correct': [True, False],  # Weak edit
                    'paraphrase_prompts_correct': [False, False, False],  # Generalization failed
                    'neighborhood_prompts_correct': [True, True, False, False],  # Overgeneralization
                    'rewrite_prompts_probs': [[2.0, 1.5], [1.8, 2.0]],
                    'paraphrase_prompts_probs': [[1.2, 2.0], [1.0, 2.1], [0.9, 2.2]],
                }
            })
        
        # Run GRPO recovery
        base_hparams = {
            'v_lr': 0.5,
            'v_num_grad_steps': 25,
            'kl_factor': 0.0625,
            'mom2_update_weight': 15000,
        }
        
        recovery_system = GRPOEditRecovery(base_hparams)
        results = recovery_system.recover_failed_edits(failed_edits, num_rounds=3)
        
        print(f"\n{'='*80}")
        print(f"FINAL RESULTS")
        print(f"{'='*80}")
        print(f"Recovered: {len(results['recovered'])} / {len(failed_edits)}")
        print(f"Recovery rate: {results['recovery_rate']:.1%}")
        print(f"\nLearned policy parameters:")
        for param, value in results['final_policy'].items():
            print(f"  {param}: {value:.3f}")


if __name__ == '__main__':
    main()
