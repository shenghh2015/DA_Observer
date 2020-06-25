import tensorflow as tf

def fp32(*values):
    if len(values) == 1 and isinstance(values[0], tuple):
        values = values[0]
    values = tuple(tf.cast(v, tf.float32) for v in values)
    return values if len(values) >= 2 else values[0]

def lerp(a, b, t):
    with tf.name_scope('Lerp'):
        return a + (b - a) * t

def lerp_clip(a, b, t):
    with tf.name_scope('LerpClip'):
        return a + (b - a) * tf.clip_by_value(t, 0.0, 1.0)