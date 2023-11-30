import os
import logging
import tensorflow as tf

tf.compat.v1.app.flags.DEFINE_integer("epochs", 10, "Epochs to train.")
tf.compat.v1.app.flags.DEFINE_integer("batch_size", 100, "Training batch size")
tf.compat.v1.app.flags.DEFINE_string("save_dir", "", "Directory to save checkpoint and logs.")
tf.compat.v1.app.flags.DEFINE_bool("use_ckpt", False, "Whether to start from scratch of start from checkpoint.")
tf.compat.v1.app.flags.DEFINE_string("ckpt_dir", "", "Checkpoint dir to use.")
tf.compat.v1.app.flags.DEFINE_float("base_lr", 1e-3, "Base learning rate.")
tf.compat.v1.app.flags.DEFINE_bool("anneal_lr", True, "Whether to anneal lr after 0.75 of total epochs.")
tf.compat.v1.app.flags.DEFINE_string("optimizer", "rmsprop", "Optimizer to use.")
tf.compat.v1.app.flags.DEFINE_integer("save_every_n_epochs", 5, "Epochs between checkpoint saves.")
tf.compat.v1.app.flags.DEFINE_integer("eval_every_n_epochs", 1, "Epochs between validation run.")
tf.compat.v1.app.flags.DEFINE_integer("print_interval", 10, "Print train metrics every n mini-batches.")
tf.compat.v1.app.flags.DEFINE_bool("debug", False, "If true, eval is not ran before training.")
tf.compat.v1.app.flags.DEFINE_bool("test_mode", False, "If true, only run test set.")

logger = logging.getLogger("tf")
logger.setLevel(logging.DEBUG)
# create console handler
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)
