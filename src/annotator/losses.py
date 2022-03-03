
import tensorflow as tf


@tf.function
def sum_cross_entropy(y_true, y_pred):
    xentropy = tf.nn.softmax_cross_entropy_with_logits(y_true, y_pred)
    comulative_error = tf.reduce_sum(xentropy, axis=-1)
    return tf.reduce_sum(comulative_error)/tf.size(comulative_error, out_type=tf.float32)


def weighted_cross_entropy(w = [1.0, 1.0, 10.0, 10.0]):
    @tf.function
    def weighted_cross_entropy_loss(y_true, y_pred):


        weights = tf.reduce_sum(tf.constant(w) * y_true, axis=-1)
        
        unweighted_losses = tf.nn.softmax_cross_entropy_with_logits(y_true, y_pred)
        
        weighted_losses = unweighted_losses * weights

        return tf.reduce_mean(weighted_losses)
        #return tf.reduce_sum(comulative_error)/tf.size(comulative_error, out_type=tf.float32)
        
    return weighted_cross_entropy_loss


def sample_weighted_cross_entropy():
    x_entropy = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    
    def sample_weighted_cross_entropy_loss(y_true, y_pred):
        
        per_batch = tf.reduce_sum(y_true * [0,0,1,1], axis=[-2,-1])
        sample_weight = tf.math.log(per_batch+1)
        #sample_weight = sample_weight/tf.reduce_sum(sample_weight)
        return x_entropy(y_true, y_pred, sample_weight=sample_weight)
    
    return sample_weighted_cross_entropy_loss