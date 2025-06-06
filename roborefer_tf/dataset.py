
import tensorflow as tf

def parse_refspatial_example(example):
    features = {
        "rgb": tf.io.FixedLenFeature([], tf.string),
        "depth": tf.io.FixedLenFeature([], tf.string),
        "prompt_ids": tf.io.VarLenFeature(tf.int64),
        "answer_ids": tf.io.VarLenFeature(tf.int64),
    }
    ex = tf.io.parse_single_example(example, features)
    rgb = tf.image.decode_jpeg(ex["rgb"])
    depth = tf.image.decode_png(ex["depth"], dtype=tf.uint16)
    depth = tf.image.convert_image_dtype(depth, tf.float32) / 65535.0
    prompt = tf.sparse.to_dense(ex["prompt_ids"])
    answer = tf.sparse.to_dense(ex["answer_ids"])
    return {"rgb": rgb, "depth": depth, "prompt": prompt, "answer": answer}

def make_dataset(tfrecord_glob, batch, shuffle=True):
    files = tf.io.gfile.glob(tfrecord_glob)
    ds = tf.data.TFRecordDataset(files, num_parallel_reads=tf.data.AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(2048)
    ds = ds.map(parse_refspatial_example, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch).prefetch(tf.data.AUTOTUNE)
