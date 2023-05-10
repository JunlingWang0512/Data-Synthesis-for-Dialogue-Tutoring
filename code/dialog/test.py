# from code.dialog.methods.dialogue_inpainting_method import DialogueInpaintingMethod
import dataclasses
import itertools
import json
import logging
import os
import subprocess
# from dialog.methods.dialogue_inpainting import DialogueInpaintingMethod
from methods.dialogue_inpainting_method import DialogueInpaintingMethod

os.environ["TRANSFORMERS_CACHE"] = os.getenv("TRANSFORMERS_CACHE", "/cluster/scratch/wangjun/tf_cache")
os.environ["HF_HOME"] = os.getenv("HF_HOME", "/cluster/scratch/wangjun/hf_cache")
# os.environ["TRANSFORMERS_CACHE"] = os.getenv("TRANSFORMERS_CACHE")
# os.environ["HF_HOME"] = os.getenv("HF_HOME")
import sys
import time
from enum import Enum

import torch
import transformers
transformers.set_seed(42)
import bert_score
from sacrebleu.metrics import BLEU
from transformers import (
    AutoConfig,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
    Trainer,
    Seq2SeqTrainingArguments,
    TrainerCallback,
    TrainerState,
    TrainerControl,
)
from transformers.trainer_utils import is_main_process, PredictionOutput, get_last_checkpoint

from dialog.arguments import *
from dialog.methods.base import Method, TaskType
from dialog.methods.generation import ResponseGenerationMethod, DocumentGroundedGenerationMethod, \
    Seq2SeqMethod, FisherApproximationForDocumentGroundedGenerationMethod, DocumentGroundedGenerationWithCTRLMethod, \
    FisherApproximationForDocumentGroundedGenerationWithCTRLMethod, DocumentGroundedGenerationWithDensityRatioMethod, \
    ResponseGenerationMethod, ChannelModelMethod, NoisyChannelModelMethod
from dialog.models.dexperts import DensityRatioMethodConfig
from dialog.models.noisy_channel import NoisyChannelConfig
from utils import NumpyEncoder


method_classes = [
    Seq2SeqMethod,
    DocumentGroundedGenerationMethod,
    ResponseGenerationMethod,
    FisherApproximationForDocumentGroundedGenerationMethod,
    DocumentGroundedGenerationWithCTRLMethod,
    FisherApproximationForDocumentGroundedGenerationWithCTRLMethod,
    DocumentGroundedGenerationWithDensityRatioMethod,
    ResponseGenerationMethod,
    ChannelModelMethod,
    NoisyChannelModelMethod,
    DialogueInpaintingMethod,  # Add this line
]



