
import tensorflow as tf
import tensorflow_hub as hub

def build_mlp_proj(hidden_dim: int, output_dim: int, name: str):
    """Linear projector used for both RGB and Depth tokens (§3.2)."""
    inp = tf.keras.Input(shape=(None, hidden_dim))
    out = tf.keras.layers.Dense(output_dim, name=f"{name}_fc")(inp)
    return tf.keras.Model(inp, out, name=name)

def build_rgb_encoder(image_res: int = 448):
    """
    Loads SigLIP‑SO400M‑P14 (TensorFlow‑Hub) as RGB backbone
    (Appendix C.1 ‘Vision encoder’).
    """
    url = "https://tfhub.dev/google/siglip/so400m-patch14-448/1"
    base = hub.KerasLayer(url, trainable=True, name="siglip_rgb")
    inp = tf.keras.Input(shape=(image_res, image_res, 3), dtype=tf.float32)
    tokens = base(inp)["pooled_output"]
    return tf.keras.Model(inp, tokens, name="rgb_encoder")

def build_depth_encoder(rgb_encoder: tf.keras.Model):
    """
    Depth encoder shares weights with RGB then updates separately (§3.2).
    """
    depth_inp = tf.keras.Input(shape=rgb_encoder.input_shape[1:], dtype=tf.float32)
    x = depth_inp
    if x.shape[-1] == 1:
        x = tf.repeat(x, 3, axis=-1)
    tokens = rgb_encoder(x)
    depth_enc = tf.keras.Model(depth_inp, tokens, name="depth_encoder")
    depth_enc.set_weights(rgb_encoder.get_weights())  # clone weights for init
    return depth_enc
