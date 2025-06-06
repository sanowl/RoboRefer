
import tensorflow as tf
from transformers import TFAutoModelForCausalLM, AutoConfig
from .encoders import build_rgb_encoder, build_depth_encoder, build_mlp_proj
from .config import RoboReferConfig

def load_llm(model_name: str = "Qwen/Qwen2-4B"):
    """Loads HF checkpoint in TensorFlow (Appendix C.1 ‘Language backbone’)."""
    cfg = AutoConfig.from_pretrained(model_name)
    return TFAutoModelForCausalLM.from_pretrained(model_name, from_pt=True)

class RoboRefer(tf.keras.Model):
    """
    Full multimodal architecture (§3.2) – two encoders, two projectors, Qwen LLM.
    """
    def __init__(self, cfg: RoboReferConfig):
        super().__init__()
        self.cfg = cfg
        self.rgb_enc = build_rgb_encoder(cfg.image_res)
        self.depth_enc = build_depth_encoder(self.rgb_enc)
        self.rgb_proj = build_mlp_proj(cfg.vision_width, cfg.llm_hidden, "rgb_proj")
        self.depth_proj = build_mlp_proj(cfg.vision_width, cfg.llm_hidden, "depth_proj")
        self.llm = load_llm()

    def call(self, rgb, depth, prompt_ids, training=False):
        rgb_tok = self.rgb_proj(self.rgb_enc(rgb))
        depth_tok = self.depth_proj(self.depth_enc(depth))
        vision_tokens = tf.concat([rgb_tok, depth_tok], axis=1)

        batch = tf.shape(prompt_ids)[0]
        vis_len = tf.shape(vision_tokens)[1]
        pad_tok = tf.zeros((batch, vis_len), dtype=prompt_ids.dtype)
        lm_input = tf.concat([pad_tok, prompt_ids], axis=1)

        return self.llm(inputs_embeds=vision_tokens,
                        input_ids=lm_input,
                        training=training)
