
import argparse, tensorflow as tf
from .config import RoboReferConfig
from .model import RoboRefer
from .dataset import make_dataset
from .reward import RewardCalculator

class GRPOTrainer:
    """
    Minimal Group Relative Policy Optimisation loop (§C.4).
    Sampling uses greedy‑temperature 1 until further work.
    """
    def __init__(self, model, n_samples=8):
        self.model = model
        self.n = n_samples
        self.opt = tf.keras.optimizers.Adam(1e-5)
        self.reward_fn = RewardCalculator()

    @tf.function
    def policy_loss(self, log_probs, rewards, baseline):
        return -tf.reduce_mean((rewards - baseline) * log_probs)

    def train_step(self, batch):
        # TODO: implement batched sampling with model.generate once TF‑HF supports it
        outputs = ["<think>dummy</think><answer>[0.0, 0.0]</answer>"] * self.n
        log_probs = tf.zeros((self.n,))  # placeholder
        rewards = self.reward_fn(outputs, {"point": [0.0, 0.0]})
        baseline = tf.reduce_mean(rewards)
        with tf.GradientTape() as tape:
            loss = self.policy_loss(log_probs, rewards, baseline)
        # Currently no trainable contribution – sampling not wired.
        return loss.numpy()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--batch", type=int, default=1)
    args = ap.parse_args()

    cfg = RoboReferConfig()
    model = RoboRefer(cfg)
    ds = make_dataset(args.data, args.batch)
    trainer = GRPOTrainer(model, cfg.rft_samples)

    for step, batch in enumerate(ds):
        loss = trainer.train_step(batch)
        if step % 100 == 0:
            print(f"RFT step {step}  loss={loss:.4f}")

if __name__ == "__main__":
    main()
