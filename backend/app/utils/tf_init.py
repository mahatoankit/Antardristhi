import os
import logging
import warnings

# Configure logging first
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress all warnings
warnings.filterwarnings("ignore")

# TensorFlow environment variables - set these BEFORE importing TensorFlow
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # 0=DEBUG, 1=INFO, 2=WARNING, 3=ERROR
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU usage

# Suppress absl logging before TF import (handles the WARNING messages)
try:
    import absl.logging

    absl.logging.set_verbosity(absl.logging.ERROR)
    # Disable absl INFO and WARNING logs
    absl.logging._warn_preinit_stderr = False
except ImportError:
    pass

# Try to import TensorFlow, but gracefully handle failures
try:
    # Import and configure TensorFlow
    import tensorflow as tf

    # Disable all TF logging
    tf.get_logger().setLevel(logging.ERROR)

    # Disable eager execution to prevent duplicate registrations
    tf.compat.v1.disable_eager_execution()

    # Set TF logging to errors only
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    # For extra safety, disable message printing
    os.environ["TF_CPP_MIN_VLOG_LEVEL"] = "3"

    TF_AVAILABLE = True
    logger.info("TensorFlow successfully imported")
except ImportError as e:
    TF_AVAILABLE = False
    logger.warning(f"TensorFlow import failed: {str(e)}")
    logger.warning(
        "Continuing without TensorFlow. Some advanced features may be limited."
    )
except Exception as e:
    TF_AVAILABLE = False
    logger.warning(f"TensorFlow initialization error: {str(e)}")
    logger.warning(
        "Continuing without TensorFlow. Some advanced features may be limited."
    )
