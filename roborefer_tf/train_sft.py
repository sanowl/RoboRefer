
import argparse, tensorflow as tf
from .config import RoboReferConfig
from .model import RoboRefer
from .dataset import make_dataset
from .losses import sft_loss

def train_depth_alignment(model, ds, epochs):
    """
    Stage‑1 (§4.1): align depth tokens with frozen RGB tokens.
    Uses MSE between projected RGB & Depth features.
    """
    opt = tf.keras.optimizers.Adam(1e-4)
    mse = tf.keras.losses.MeanSquaredError()
    for ep in range(epochs):
        for batch in ds:
            with tf.GradientTape() as tape:
                rgb_feat = model.rgb_proj(model.rgb_enc(batch["rgb"]))
                depth_feat = model.depth_proj(model.depth_enc(batch["depth"]))
                loss = mse(rgb_feat, depth_feat)
            vars_ = model.depth_enc.trainable_variables + model.depth_proj.trainable_variables
            grads = tape.gradient(loss, vars_)
            opt.apply_gradients(zip(grads, vars_))
        print(f"Depth‑align epoch {ep}  loss={loss.numpy():.4f}")

def train_spatial(model, ds, epochs):
    """Stage‑2 (§4.1): supervised captioning SFT."""
    opt = tf.keras.optimizers.Adam(5e-5)
    for ep in range(epochs):
        for batch in ds:
            with tf.GradientTape() as tape:
                logits = model(batch["rgb"], batch["depth"], batch["prompt"], training=True).logits
                loss = sft_loss(batch["answer"], logits)
            grads = tape.gradient(loss, model.trainable_variables)
            opt.apply_gradients(zip(grads, model.trainable_variables))
        print(f"Spatial SFT epoch {ep}  loss={loss.numpy():.4f}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--depth_epochs", type=int, default=1)
    ap.add_argument("--spatial_epochs", type=int, default=2)
    ap.add_argument("--ckpt_out", default="roborefer_sft.h5")
    args = ap.parse_args()

    cfg = RoboReferConfig()
    model = RoboRefer(cfg)
    ds = make_dataset(args.data, args.batch)

    train_depth_alignment(model, ds, args.depth_epochs)
    train_spatial(model, ds, args.spatial_epochs)
    model.save_weights(args.ckpt_out)
    print("Saved SFT weights to", args.ckpt_out)

if __name__ == "__main__":
    main()
