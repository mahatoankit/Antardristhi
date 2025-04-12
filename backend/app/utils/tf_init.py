import os
import logging
import tensorflow as tf
import warnings

# Suppress TensorFlow warnings and errors
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # 0=DEBUG, 1=INFO, 2=WARNING, 3=ERROR
tf.get_logger().setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

# Disable TensorFlow's auto-tuning
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Disable TensorFlow's GPU usage if causing issues
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
