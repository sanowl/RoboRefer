
import argparse, importlib, sys

def main():
    parser = argparse.ArgumentParser(description="Unified CLI dispatcher")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("train_sft")
    sub.add_parser("train_rft")
    sub.add_parser("eval")
    args, rem = parser.parse_known_args()

    if args.cmd == "train_sft":
        import roborefer_tf.train_sft as _mod; _mod.main()
    elif args.cmd == "train_rft":
        import roborefer_tf.train_rft as _mod; _mod.main()
    elif args.cmd == "eval":
        import roborefer_tf.eval as _mod; _mod.main()

if __name__ == "__main__":
    main()
