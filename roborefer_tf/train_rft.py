import argparse
import json
import logging
from typing import Dict, List, Any, Tuple, Union

import tensorflow as tf
import numpy as np
from transformers import AutoTokenizer

from .config import RoboReferConfig
from .model import RoboRefer
from .dataset import make_dataset
from .reward import RewardCalculator

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GRPOTrainer:
    """Group‑Relative Policy Optimisation (§C.4) with REINFORCE baseline.
    
    Fixed version with proper TensorFlow integration and error handling.
    """

    def __init__(self, model: RoboRefer, n_samples: int = 8, learning_rate: float = 1e-5):
        self.model = model
        self.n = n_samples
        self.learning_rate = learning_rate
        
        # Optimizer with gradient clipping
        self.opt = tf.keras.optimizers.Adam(
            learning_rate=learning_rate, 
            global_clipnorm=1.0,
            epsilon=1e-8
        )
        
        # Reward calculator
        self.reward_fn = RewardCalculator()
        
        # Load matching tokenizer so we can decode outputs
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-4B")
        
        # Ensure pad token exists (critical for generation)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            
        # Moving‑average baseline (EMA) for variance reduction
        self.baseline = tf.Variable(0.0, trainable=False, dtype=tf.float32, name="baseline")
        
        # Training metrics
        self.step_count = tf.Variable(0, trainable=False, dtype=tf.int64, name="step_count")
        self.total_reward_sum = tf.Variable(0.0, trainable=False, dtype=tf.float32, name="total_reward")
        
        logger.info(f"Initialized GRPO trainer with {n_samples} samples per batch")

    # ---------------------------------------------------------------------
    # Core helpers
    # ---------------------------------------------------------------------
    def _validate_inputs(self, rgb: tf.Tensor, depth: tf.Tensor, prompt_ids: tf.Tensor) -> None:
        """Validate input tensors."""
        if tf.reduce_any(tf.equal(tf.shape(prompt_ids)[0], 0)):
            raise ValueError("Empty prompt_ids batch")
        if tf.reduce_any(tf.equal(tf.shape(rgb)[0], 0)):
            raise ValueError("Empty rgb batch")
        if tf.reduce_any(tf.equal(tf.shape(depth)[0], 0)):
            raise ValueError("Empty depth batch")

    def _expand_batch(self, tensor: tf.Tensor) -> tf.Tensor:
        """Tile a `(batch_size, …)` tensor `n` times on batch dimension.
        
        Args:
            tensor: Input tensor of shape (batch_size, ...)
            
        Returns:
            Expanded tensor of shape (batch_size * n, ...)
        """
        # Create tile pattern: [n, 1, 1, ...] for remaining dimensions
        tile_pattern = tf.concat([[self.n], tf.ones(tf.rank(tensor) - 1, tf.int32)], 0)
        return tf.tile(tensor, tile_pattern)

    def _sample_completions(self, rgb: tf.Tensor, depth: tf.Tensor, prompt_ids: tf.Tensor) -> Dict[str, Any]:
        """Generate `n` candidate completions and compute log‑probs for each.
        
        Args:
            rgb: RGB vision input (batch_size, H, W, 3)
            depth: Depth vision input (batch_size, H, W, 1)
            prompt_ids: Token IDs for prompt (batch_size, seq_len)
            
        Returns:
            Dictionary with 'outputs', 'log_probs', and 'generated_ids'
        """
        # Validate inputs
        self._validate_inputs(rgb, depth, prompt_ids)
        
        # Repeat vision + prompt to shape (n * batch_size, …)
        rgb_expanded = self._expand_batch(rgb)
        depth_expanded = self._expand_batch(depth)
        prompt_expanded = self._expand_batch(prompt_ids)
        
        # Generate completions using HuggingFace
        try:
            generated_ids = self.model.llm.generate(
                input_ids=prompt_expanded,
                max_new_tokens=64,
                min_new_tokens=1,
                temperature=1.0,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                return_dict_in_generate=False,
                output_scores=False,
            )
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise RuntimeError(f"HuggingFace generation failed: {e}")
        
        # Compute log probabilities for generated tokens
        logits = self.model(rgb_expanded, depth_expanded, prompt_expanded, training=False).logits
        
        # Extract dimensions
        seq_len = tf.shape(generated_ids)[1]
        prompt_len = tf.shape(prompt_ids)[1]
        
        # Extract only newly generated tokens and corresponding logits
        new_token_ids = generated_ids[:, prompt_len:]  # (n * batch_size, new_seq_len)
        # Shift logits left for autoregressive prediction
        new_logits = logits[:, prompt_len - 1 : seq_len - 1]  # (n * batch_size, new_seq_len, vocab)
        
        # Ensure shapes match
        new_seq_len = tf.shape(new_token_ids)[1]
        new_logits = new_logits[:, :new_seq_len]  # Truncate if needed
        
        # Compute per-token log probabilities
        log_probs_per_token = -tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=new_token_ids,
            logits=new_logits
        )
        
        # Handle padding tokens (set their log prob to 0)
        pad_mask = tf.not_equal(new_token_ids, self.tokenizer.pad_token_id)
        log_probs_per_token = tf.where(pad_mask, log_probs_per_token, 0.0)
        
        # Sum log‑probs across generated tokens for each sequence
        log_probs_seq = tf.reduce_sum(log_probs_per_token, axis=1)  # (n * batch_size,)
        
        # Decode outputs for reward computation
        try:
            decoded_outputs = [
                self.tokenizer.decode(seq, skip_special_tokens=True) 
                for seq in generated_ids.numpy()
            ]
        except Exception as e:
            logger.error(f"Decoding failed: {e}")
            # Fallback: create dummy outputs
            decoded_outputs = ["[DECODE_ERROR]"] * len(generated_ids)
        
        return {
            "outputs": decoded_outputs,
            "log_probs": log_probs_seq,
            "generated_ids": generated_ids,
            "new_token_ids": new_token_ids
        }

    def _compute_rewards(self, outputs: List[str], gt_data: Dict[str, Any]) -> tf.Tensor:
        """Compute rewards for generated outputs.
        
        Args:
            outputs: List of decoded output strings
            gt_data: Ground truth data dictionary
            
        Returns:
            Tensor of rewards with shape (n * batch_size,)
        """
        try:
            # Call reward function
            if hasattr(self.reward_fn, '__call__'):
                rewards_raw = self.reward_fn(outputs, gt_data)
            else:
                # Fallback: assume it's a method
                rewards_raw = self.reward_fn.calculate_reward(outputs, gt_data)
            
            # Convert to tensor
            if isinstance(rewards_raw, (list, tuple, np.ndarray)):
                rewards = tf.constant(rewards_raw, dtype=tf.float32)
            elif isinstance(rewards_raw, tf.Tensor):
                rewards = tf.cast(rewards_raw, tf.float32)
            else:
                # Single value - expand to match batch
                rewards = tf.fill([len(outputs)], tf.cast(rewards_raw, tf.float32))
                
            # Validate shape
            expected_shape = [len(outputs)]
            if rewards.shape.as_list() != expected_shape:
                logger.warning(f"Reward shape mismatch: got {rewards.shape}, expected {expected_shape}")
                rewards = tf.reshape(rewards, expected_shape)
                
            return rewards
            
        except Exception as e:
            logger.error(f"Reward computation failed: {e}")
            # Fallback: return zero rewards
            return tf.zeros([len(outputs)], dtype=tf.float32)

    def _extract_ground_truth(self, batch: Dict[str, tf.Tensor]) -> Dict[str, Any]:
        """Extract ground truth data from batch in a TF-function safe way.
        
        Args:
            batch: Training batch dictionary
            
        Returns:
            Ground truth data dictionary
        """
        gt_data = {}
        
        # Extract coordinates if available
        if "gt_x" in batch and "gt_y" in batch:
            gt_x = tf.squeeze(batch["gt_x"])
            gt_y = tf.squeeze(batch["gt_y"])
            # Keep as tensors for now, convert later if needed
            gt_data["point_tensor"] = tf.stack([gt_x, gt_y])
            # For reward function, we'll need Python values
            gt_data["point"] = [gt_x, gt_y]  # Will be converted in eager mode
        else:
            gt_data["point_tensor"] = tf.constant([0.0, 0.0])
            gt_data["point"] = [0.0, 0.0]
            
        # Add other ground truth data as needed
        for key in ["instruction", "target", "label"]:
            if key in batch:
                gt_data[key] = batch[key]
                
        return gt_data

    def train_step_eager(self, batch: Dict[str, tf.Tensor]) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, Dict[str, Any]]:
        """Training step in eager execution mode (recommended for development).
        
        Args:
            batch: Training batch dictionary
            
        Returns:
            Tuple of (loss, mean_reward, baseline, metrics)
        """
        rgb = batch["rgb"]
        depth = batch["depth"] 
        prompt_ids = batch["prompt"]
        
        # Extract ground truth data
        gt_data = self._extract_ground_truth(batch)
        
        # Convert tensor coordinates to Python values for reward function
        if "point_tensor" in gt_data:
            point_tensor = gt_data["point_tensor"]
            if hasattr(point_tensor, 'numpy'):
                gt_data["point"] = point_tensor.numpy().tolist()
            else:
                gt_data["point"] = [float(point_tensor[0]), float(point_tensor[1])]

        # Sample completions
        sample_results = self._sample_completions(rgb, depth, prompt_ids)
        log_probs = sample_results["log_probs"]  # (n,)
        
        # Compute rewards
        rewards = self._compute_rewards(sample_results["outputs"], gt_data)  # (n,)
        
        # Training step with gradients
        with tf.GradientTape() as tape:
            tape.watch(self.model.trainable_variables)
            
            # Compute advantage (rewards - baseline)
            advantage = rewards - self.baseline
            
            # GRPO loss: maximize expected advantage-weighted log probability
            loss = -tf.reduce_mean(advantage * log_probs)
            
            # Optional: add entropy regularization to encourage exploration
            # entropy_bonus = 0.01 * tf.reduce_mean(log_probs)
            # loss = loss - entropy_bonus

        # Compute and apply gradients
        gradients = tape.gradient(loss, self.model.trainable_variables)
        
        # Check for NaN gradients
        finite_grads = []
        for grad in gradients:
            if grad is not None:
                finite_grads.append(tf.reduce_all(tf.math.is_finite(grad)))
            else:
                finite_grads.append(tf.constant(True))
                
        if not tf.reduce_all(finite_grads):
            logger.warning("NaN or Inf gradients detected, skipping update")
            return loss, tf.reduce_mean(rewards), self.baseline, {"grad_norm": float('inf')}
        
        self.opt.apply_gradients(zip(gradients, self.model.trainable_variables))

        # Update moving baseline (EMA with momentum 0.9)
        mean_reward = tf.reduce_mean(rewards)
        self.baseline.assign(self.baseline * 0.9 + mean_reward * 0.1)
        
        # Update step counter and running stats
        self.step_count.assign_add(1)
        self.total_reward_sum.assign_add(mean_reward)
        
        # Compute metrics
        grad_norm = tf.linalg.global_norm(gradients)
        metrics = {
            "grad_norm": float(grad_norm),
            "advantage_mean": float(tf.reduce_mean(advantage)),
            "advantage_std": float(tf.math.reduce_std(advantage)),
            "reward_min": float(tf.reduce_min(rewards)),
            "reward_max": float(tf.reduce_max(rewards)),
            "log_prob_mean": float(tf.reduce_mean(log_probs)),
            "baseline_value": float(self.baseline),
            "step_count": int(self.step_count)
        }
        
        return loss, mean_reward, self.baseline, metrics

    @tf.function
    def train_step_graph(self, batch: Dict[str, tf.Tensor]) -> Tuple[tf.Tensor, tf.Tensor]:
        """Training step in graph mode (faster but less flexible).
        
        Note: This is more complex due to HuggingFace compatibility issues.
        Use eager mode for development.
        """
        rgb = batch["rgb"]
        depth = batch["depth"]
        prompt_ids = batch["prompt"]
        
        # Extract ground truth (keeping as tensors)
        gt_data = self._extract_ground_truth(batch)
        
        # Use py_function to handle HuggingFace generation
        def sample_and_reward_fn():
            # This will be called in eager mode even within tf.function
            sample_results = self._sample_completions(rgb, depth, prompt_ids)
            rewards = self._compute_rewards(sample_results["outputs"], gt_data)
            return sample_results["log_probs"], rewards
        
        log_probs, rewards = tf.py_function(
            func=sample_and_reward_fn,
            inp=[],
            Tout=[tf.float32, tf.float32]
        )
        
        # Set shapes explicitly (required for tf.function)
        log_probs.set_shape([None])
        rewards.set_shape([None])
        
        with tf.GradientTape() as tape:
            tape.watch(self.model.trainable_variables)
            advantage = rewards - self.baseline
            loss = -tf.reduce_mean(advantage * log_probs)

        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.opt.apply_gradients(zip(gradients, self.model.trainable_variables))

        # Update baseline
        mean_reward = tf.reduce_mean(rewards)
        self.baseline.assign(self.baseline * 0.9 + mean_reward * 0.1)
        
        return loss, mean_reward

    def save_checkpoint(self, filepath: str, include_optimizer: bool = True) -> None:
        """Save training checkpoint."""
        checkpoint_data = {
            "model_weights": self.model.get_weights(),
            "baseline": float(self.baseline.numpy()),
            "step_count": int(self.step_count.numpy()),
            "total_reward_sum": float(self.total_reward_sum.numpy()),
        }
        
        if include_optimizer:
            checkpoint_data["optimizer_weights"] = self.opt.get_weights()
            
        # Save as numpy arrays for compatibility
        np.savez_compressed(filepath, **checkpoint_data)
        logger.info(f"Saved checkpoint to {filepath}")

    def load_checkpoint(self, filepath: str, load_optimizer: bool = True) -> None:
        """Load training checkpoint."""
        try:
            checkpoint_data = np.load(filepath, allow_pickle=True)
            
            # Restore model weights
            self.model.set_weights(checkpoint_data["model_weights"])
            
            # Restore training state
            self.baseline.assign(float(checkpoint_data["baseline"]))
            self.step_count.assign(int(checkpoint_data["step_count"]))
            self.total_reward_sum.assign(float(checkpoint_data["total_reward_sum"]))
            
            # Restore optimizer state if available
            if load_optimizer and "optimizer_weights" in checkpoint_data:
                self.opt.set_weights(checkpoint_data["optimizer_weights"])
                
            logger.info(f"Loaded checkpoint from {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            raise


def main():
    """Main training script."""
    ap = argparse.ArgumentParser(description="GRPO fine-tuning for RoboRefer")
    ap.add_argument("--data", required=True, help="Path to training data")
    ap.add_argument("--batch", type=int, default=1, help="Batch size")
    ap.add_argument("--ckpt", help="Optional pretrained SFT weights")
    ap.add_argument("--resume", help="Resume from checkpoint")
    ap.add_argument("--steps", type=int, default=10000, help="Training steps")
    ap.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    ap.add_argument("--samples", type=int, default=8, help="Samples per batch")
    ap.add_argument("--eager", action="store_true", help="Use eager mode (recommended)")
    ap.add_argument("--log-interval", type=int, default=50, help="Logging interval")
    ap.add_argument("--save-interval", type=int, default=1000, help="Checkpoint save interval")
    ap.add_argument("--output-dir", default="./checkpoints", help="Output directory")
    args = ap.parse_args()

    # Setup
    tf.config.run_functions_eagerly(args.eager)
    
    cfg = RoboReferConfig()
    cfg.rft_samples = args.samples
    
    model = RoboRefer(cfg)
    
    # Load initial weights
    if args.ckpt:
        model.load_weights(args.ckpt)
        logger.info(f"Loaded pretrained weights from {args.ckpt}")

    # Create trainer
    trainer = GRPOTrainer(model, n_samples=args.samples, learning_rate=args.lr)
    
    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
        logger.info(f"Resumed training from {args.resume}")

    # Load dataset
    ds = make_dataset(args.data, args.batch)
    
    # Create output directory
    import os
    os.makedirs(args.output_dir, exist_ok=True)

    logger.info(f"Starting GRPO training ({'eager' if args.eager else 'graph'} mode)")
    logger.info(f"Samples per batch: {args.samples}, Learning rate: {args.lr}")
    
    # Training loop
    try:
        for step, batch in enumerate(ds):
            if step >= args.steps:
                break
                
            if args.eager:
                # Use eager mode (recommended)
                loss, mean_reward, baseline, metrics = trainer.train_step_eager(batch)
                
                if step % args.log_interval == 0:
                    logger.info(
                        f"Step {step:5d} | Loss: {loss.numpy():.4f} | "
                        f"Reward: {mean_reward.numpy():.3f} | "
                        f"Baseline: {baseline.numpy():.3f} | "
                        f"GradNorm: {metrics['grad_norm']:.3f} | "
                        f"AdvStd: {metrics['advantage_std']:.3f}"
                    )
            else:
                # Use graph mode
                loss, mean_reward = trainer.train_step_graph(batch)
                
                if step % args.log_interval == 0:
                    logger.info(
                        f"Step {step:5d} | Loss: {loss.numpy():.4f} | "
                        f"Reward: {mean_reward.numpy():.3f} | "
                        f"Baseline: {trainer.baseline.numpy():.3f}"
                    )
            
            # Save checkpoint periodically
            if step > 0 and step % args.save_interval == 0:
                ckpt_path = os.path.join(args.output_dir, f"checkpoint_step_{step}.npz")
                trainer.save_checkpoint(ckpt_path)
    
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
    finally:
        # Save final model
        final_weights_path = os.path.join(args.output_dir, "roborefer_rft_final.h5")
        model.save_weights(final_weights_path)
        logger.info(f"Saved final weights to {final_weights_path}")
        
        # Save final checkpoint
        final_ckpt_path = os.path.join(args.output_dir, "final_checkpoint.npz")
        trainer.save_checkpoint(final_ckpt_path)
        logger.info(f"Saved final checkpoint to {final_ckpt_path}")
        
        # Print final statistics
        total_steps = int(trainer.step_count.numpy())
        avg_reward = float(trainer.total_reward_sum.numpy()) / max(total_steps, 1)
        logger.info(f"Training completed: {total_steps} steps, avg reward: {avg_reward:.3f}")


if __name__ == "__main__":
    main()