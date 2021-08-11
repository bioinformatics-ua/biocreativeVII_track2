import logging
import os
import sys
import tensorflow as tf
import numpy as np
import random
import inspect
from tensorflow.data import Dataset
log = logging.getLogger(__name__)
from logging.handlers import TimedRotatingFileHandler

FORMATTER = logging.Formatter("%(asctime)s — %(name)s — %(levelname)s — %(message)s")


LOG_FILE = os.path.join("logs", "logsbiocreative.log")

def dsc(y_true, y_pred):
    smooth = 1.
    y_true_f = tf.reshape(y_true, (-1,))
    y_pred_f = tf.reshape(y_pred, (-1,))
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
    return score

@tf.function
def dice_loss(y_true, y_pred):
    print(y_true)
    print(y_pred)
    #https://stackoverflow.com/questions/51973856/how-is-the-smooth-dice-loss-differentiable
    return (1 - dsc(y_true, y_pred))

@tf.function
def sum_cross_entropy(y_true, y_pred):
    xentropy = tf.nn.softmax_cross_entropy_with_logits(y_true, y_pred)
    comulative_error = tf.reduce_sum(xentropy, axis=-1)
    return tf.reduce_sum(comulative_error)/tf.size(comulative_error, out_type=tf.float32)

class BaseLogger:
    def __init__(self):
        """
        From: https://www.toptal.com/python/in-depth-python-logging
        """
        super().__init__()

        self.logger = logging.getLogger(self.__class__.__name__)
        
        if not self.logger.hasHandlers():
            self.logger.setLevel(logging.DEBUG)

            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(FORMATTER)
            self.logger.addHandler(console_handler)

            if not os.path.exists('logs'):
                os.makedirs('logs')

            file_handler = TimedRotatingFileHandler(LOG_FILE, when='midnight', encoding='utf-8')
            file_handler.setFormatter(FORMATTER)
            self.logger.addHandler(file_handler)

            self.logger.propagate = False
        
def concate_tf_dataset(tf_datasets):
    n_samples = sum([x.data_loader.n_samples for x in tf_datasets])

    def new_generator():
        for tf_dataset in tf_datasets:
            for data in tf_dataset:
                yield data

    dtypes, shapes = find_dtype_and_shapes(new_generator())

    return Dataset.from_generator(new_generator, 
                                  output_types= dtypes, 
                                  output_shapes= shapes), n_samples
        
def set_random_seed(seed_value=42):
    tf.random.set_seed(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)

def merge_dicts(*list_of_dicts):
    # fast merge according to https://stackoverflow.com/questions/1781571/how-to-concatenate-two-dictionaries-to-create-a-new-one-in-python
    
    temp = dict(list_of_dicts[0], **list_of_dicts[1])
    
    for i in range(2, len(list_of_dicts)):
        temp.update(list_of_dicts[i])
        
    return temp


def find_dtype_and_shapes(data_generator):
    """
    Automatically gets the dtype and shapes of samples
    
    """
    # get one sample
    sample = next(iter(data_generator))

    if isinstance(sample, dict):
        dtypes = {}
        shapes = {}
        for key in sample.keys():
            tf_value = tf.constant(sample[key])
            dtypes[key] = tf_value.dtype
            shapes[key] = tf_value.shape
    elif isinstance(sample, tuple):
        dtypes = []
        shapes = []
        for e in sample:
            tf_value = tf.constant(e)
            dtypes.append(tf_value.dtype)
            shapes.append(tf_value.shape)
        dtypes = tuple(dtypes)
        shapes = tuple(shapes)
    elif isinstance(sample, list):
        dtypes = []
        shapes = []
        for e in sample:
            tf_value = tf.constant(e)
            dtypes.append(tf_value.dtype)
            shapes.append(tf_value.shape)

    else:
        raise ValueError(f"The find_dtype_and_shapes only supports when the sample from generator are dict or tuples or list, but found {type(sample)}")
    
    return dtypes, shapes


"""
FUNCTIONS FOR DEBUG AND TEST
"""
from timeit import default_timer as timer

def acc_on_test(model, test_ds):
    # simple evaluation
    y_hat = []
    y_true = []
    
    s = timer()
    for x, y in test_ds:
        #y_hat.extend(np.argmax(model.predict(x), axis=-1).tolist())
        y_hat.extend(model.inference(x).numpy().tolist())
        y_true.extend(y.numpy().tolist())
    print("time: ", timer()-s)
    acc = sum([y_hat[i]==y_true[i] for i in range(len(y_true))])/len(y_true)
    return acc
    
def load_dataset(batch_size):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    x_train, x_test = x_train / 255.0, x_test / 255.0

    
    def transform_data_train(x, y):
        x = tf.cast(x, dtype=tf.float32)
        return tf.expand_dims(x, axis=-1), tf.one_hot(y, 10)

    train_ds = tf.data.Dataset.from_tensor_slices((x_train,y_train))\
                                            .map(transform_data_train, num_parallel_calls=tf.data.experimental.AUTOTUNE, deterministic=False)\
                                            .cache("afdasdfja")\
                                            .shuffle(60000, reshuffle_each_iteration=True)\
                                            .batch(batch_size, drop_remainder=True)\
                                            .prefetch(tf.data.experimental.AUTOTUNE)
    
    def transform_data_test(x, y):
        x = tf.cast(x, dtype=tf.float32)
        return tf.expand_dims(x, axis=-1), tf.cast(y, dtype=tf.int32)
    
    test_ds =  tf.data.Dataset.from_tensor_slices((x_test,y_test))\
                                            .map(transform_data_test, num_parallel_calls=tf.data.experimental.AUTOTUNE, deterministic=False)\
                                            .cache()\
                                            .batch(batch_size)\
                                            .prefetch(tf.data.experimental.AUTOTUNE)
    print("Train:", train_ds.element_spec)
    print()
    print("Test:", test_ds.element_spec)
    
    return train_ds, test_ds