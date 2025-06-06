
import tensorflow as tf, json, re
from typing import List, Dict, Any

class RewardCalculator:
    """
    Implements ROF, RP, RPF, RAcc (Appendix C.4).
    RPF / RAcc are pattern‑matching placeholders – fill in with key‑step
    annotations once available.
    """
    def __init__(self, tol_px: int = 50):
        self.tol = tol_px

    def __call__(self, outputs: List[str], gts: Dict[str, Any]) -> tf.Tensor:
        rewards = []
        for out in outputs:
            rof = int("<answer>" in out and "<think>" in out)
            try:
                pt_str = re.findall(r"<answer>(.*?)</answer>", out)[0]
                x, y = json.loads(pt_str.strip())
                gt_x, gt_y = gts["point"]
                rp = int(abs(x - gt_x) + abs(y - gt_y) < self.tol)
            except Exception:
                rp = 0
            rpf = racc = 0  # TODO: implement once step annotations are in data
            rewards.append(rof + rp + 0.25 * (rpf + racc))
        return tf.convert_to_tensor(rewards, dtype=tf.float32)
