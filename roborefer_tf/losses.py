
import tensorflow as tf

def sft_loss(labels, logits):
    """
    Masked cross‑entropy (§4.1 Stage‑2).
    Padding token id = 0.
    """
    loss = tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)
    mask = tf.cast(tf.not_equal(labels, 0), tf.float32)
    return tf.reduce_sum(loss * mask) / tf.reduce_sum(mask)
