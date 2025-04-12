import os
import logging
import warnings

# Suppress warnings and configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

# TensorFlow environment variables
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # 0=DEBUG, 1=INFO, 2=WARNING, 3=ERROR
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU usage

# Try to import TensorFlow, but gracefully handle failures
try:
    import tensorflow as tf
    tf.get_logger().setLevel(logging.ERROR)
    TF_AVAILABLE = True
    logger.info("TensorFlow successfully imported")
except ImportError as e:
    TF_AVAILABLE = False
    logger.warning(f"TensorFlow import failed: {str(e)}")
    logger.warning("Continuing without TensorFlow. Some advanced features may be limited.")
except Exception as e:
    TF_AVAILABLE = False
    logger.warning(f"TensorFlow initialization error: {str(e)}")
    logger.warning("Continuing without TensorFlow. Some advanced features may be limited.")
