
from dataclasses import dataclass

@dataclass
class RoboReferConfig:
    """
    Configuration mirrors Appendix C.1 (Zhou et al., 2025).
    Adjust the defaults if you swap backbones or change training strategy.
    """
    image_res: int = 448
    vision_width: int = 1024
    llm_hidden: int = 4096
    proj_dim: int = 2048
    depth_alignment_epochs: int = 1   # Stage‑1 (§4.1)
    spatial_epochs: int = 2           # Stage‑2 (§4.1)
    rft_samples: int = 8              # N rollouts per prompt (§C.4)
