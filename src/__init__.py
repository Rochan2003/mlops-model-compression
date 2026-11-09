"""Sets the legacy Keras flag before anything imports TensorFlow.

tensorflow_model_optimization predates Keras 3 and still expects the Keras 2
API. TensorFlow 2.16 and later ship Keras 3 by default, so without this the
pruning and QAT wrappers fail. It has to be set before TensorFlow is imported,
which is why it lives here rather than in a script.
"""

import os

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
# Keeps the TensorFlow startup banner out of the logs.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
