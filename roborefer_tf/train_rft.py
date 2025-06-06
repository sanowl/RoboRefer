import argparse
import json 
from typing import Dict,List,Any 

import tensorflow as tf
from transformers import AutoTokenizer

from.config import RoboReferConfig
from .model import RoboRefer
from .dataset import make_dataset
from.reward import RewardCalculator

class GRPOTrainer:
        def __init__(self, model: RoboRefer, n_samples: int = 8):
                self.model = model
                self.n_samples = n_samples
                self.optimizer = tf.keras.optimizers.Adam(learning_rate=1.0)
                self.reward_fn = RewardCalculator()
                #load matching tokziner so we can decode outputs 
                self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-4B")
                self.baseline = tf.Variable(0.0, trainable=False, dtype=tf.float32)

