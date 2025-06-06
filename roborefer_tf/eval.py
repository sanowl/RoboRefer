
import argparse, tensorflow as tf
from .config import RoboReferConfig
from .model import RoboRefer
from .dataset import make_dataset

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--batch", type=int, default=1)
    args = ap.parse_args()

    model = RoboRefer(RoboReferConfig())
    model.load_weights(args.ckpt)

    ds = make_dataset(args.data, args.batch, shuffle=False)
    # TODO: implement mask‑IoU metric (§4.2)
    print("Evaluation scaffold loaded – implement mask‑IoU for RefSpatial‑Bench.")

if __name__ == "__main__":
    main()
