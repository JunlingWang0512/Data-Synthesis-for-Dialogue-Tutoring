import dataclasses
import itertools
import json
from uuid import uuid4
import logging
import os
import subprocess

from typing import List, Tuple
import itertools
from datasets import Dataset, load_dataset, concatenate_datasets
import time
import random

os.environ["TRANSFORMERS_CACHE"] = os.getenv("TRANSFORMERS_CACHE", "/cluster/scratch/wangjun/dialogue_inpainting5_10_both/tf_cache")
os.environ["HF_HOME"] = os.getenv("HF_HOME", "/cluster/scratch/wangjun/dialogue_inpainting5_10_both/hf_cache")
# os.environ["TRANSFORMERS_CACHE"] = os.getenv("TRANSFORMERS_CACHE")
# os.environ["HF_HOME"] = os.getenv("HF_HOME")
import sys
import time
from enum import Enum

import numpy as np
# import openai
import nltk
nltk.download('punkt')

import pandas
import re
import csv


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

from dialog.methods.dialogue_inpainting_method import DialogueInpaintingMethod #junling modify
from dialog.arguments import *
from dialog.methods.base import Method, TaskType
from dialog.methods.generation import ResponseGenerationMethod, DocumentGroundedGenerationMethod, \
    Seq2SeqMethod, FisherApproximationForDocumentGroundedGenerationMethod, DocumentGroundedGenerationWithCTRLMethod, \
    FisherApproximationForDocumentGroundedGenerationWithCTRLMethod, DocumentGroundedGenerationWithDensityRatioMethod, \
    ResponseGenerationMethod, ChannelModelMethod, NoisyChannelModelMethod
from dialog.models.dexperts import DensityRatioMethodConfig
from dialog.models.noisy_channel import NoisyChannelConfig
from utils import NumpyEncoder

logging.basicConfig(stream=sys.stdout, level=logging.NOTSET)
logger = logging.getLogger(__name__)
os.environ["WANDB_DISABLED"] = "true"


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
    DialogueInpaintingMethod,  # junling modify
]


class GPUMemoryCallback(TrainerCallback):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prediction_step = 0

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        if torch.cuda.is_available():
            max_gpu_allocated = torch.cuda.max_memory_allocated() / 10 ** 9
            logging.info(f"Maximum allocated GPU memory: {max_gpu_allocated:.3f} GB")
            state.log_history[-1]['gpu_memory'] = torch.cuda.max_memory_allocated()

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if state.global_step in [1, 8, 64, 512]:
            if torch.cuda.is_available():
                max_gpu_allocated = torch.cuda.max_memory_allocated() / 10 ** 9
                logging.info(
                    f"Maximum allocated GPU memory: {max_gpu_allocated:.3f} GB")
        super().on_step_end(args, state, control, **kwargs)

    def on_prediction_step(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        self.prediction_step += 1
        if self.prediction_step in [1, 8, 64, 512]:
            if torch.cuda.is_available():
                max_gpu_allocated = torch.cuda.max_memory_allocated() / 10 ** 9
                logging.info(
                    f"Maximum allocated GPU memory: {max_gpu_allocated:.3f} GB")
        super().on_prediction_step(args, state, control, **kwargs)


def _setup_logging(training_args: TrainingArguments):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
    )
    logger.setLevel(logging.INFO if is_main_process(
        training_args.local_rank) else logging.WARN)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
    logger.info("Training/evaluation parameters %s", training_args)


def get_config_class(model_args):
    if "density_ratio" in model_args.method:
        return DensityRatioMethodConfig
    elif "noisy_channel" in model_args.method:
        return NoisyChannelConfig
    else:
        return AutoConfig #I use autoconfig


def get_tokenizer_name(config, model_args):
    # with open('/cluster/scratch/wangjun/temp3/model_args.txt', 'w') as f:
    #     f.write(str(model_args))
    if "density_ratio" in model_args.method:
        return config.language_model_tokenizer_name_or_path
    elif "noisy_channel" in model_args.method:
        return config.direct_model_tokenizer_name_or_path
    elif model_args.tokenizer_name:
        return model_args.tokenizer_name
    else:
        return model_args.model_name_or_path


class RunMode(Enum):
    TRAIN = 1
    PREDICT = 2

#__function for dialog inpainting__
tokenizer = AutoTokenizer.from_pretrained(  #can be merged into main function if needed later
        'google/flan-t5-base',
        cache_dir='/cluster/scratch/wangjun/dialogue_inpainting5_10_both/cache',
        use_fast=True,
        revision="main",
        use_auth_token=None,
    )
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    
def _truncate_to_max_length(context):
        context_len = 0
        truncated_context = []
        history_max_utterances = 999
        history_max_tokens = 384

        context = context[-history_max_utterances:]
        for turn in context[::-1]:
            if context_len + len(turn) < history_max_tokens:
                truncated_context.append(turn)
                context_len += len(turn)
            else:
                break

        return truncated_context[::-1]
def _tokenize_with_special_tokens(turn):
        user_token = "<user>"
        system_token ="<system>"
        dialog_act = tokenizer(turn["dialog_act"], add_special_tokens=False)["input_ids"]
        user_tag = user_token if turn["user"] == "user" else system_token
        user_tag = tokenizer(user_tag, add_special_tokens=False)["input_ids"]
        text = tokenizer(turn["text"], add_special_tokens=False)["input_ids"]
        return user_tag + dialog_act + text
def _process_dialog_context(context):
        context = [_tokenize_with_special_tokens(turn) for turn in context]
        context = _truncate_to_max_length(context)
        return list(itertools.chain.from_iterable(context))
def _process_response(response):
        response = tokenizer(response)["input_ids"]
        if len(response) > 512:
            response = response[:511] + [response[-1]]
        return response
    
def generate_partial_dialog(sentences: List[str], document_title: str) -> Tuple[List[dict], str]:
    sequences, labels = [], []
    
    if len(sentences) % 2 == 0:
        raise ValueError("The input 'sentences' must have an odd number of elements.")

    introduction = f"Hello, I am an automated assistant and can answer questions about {document_title}"
    dialog = [{'dialog_act': '', 'text': introduction, 'user': 'user'}]

    for i, text in enumerate(sentences[:-1]):
        user = 'system' if i % 2 == 0 else 'user'
        dialog.append({'dialog_act': '', 'text': text, 'user': user})

    dialog.append({'dialog_act': '', 'text': '<extra_id_0>', 'user': 'system'})
    dialog.append({'dialog_act': '', 'text': sentences[-1], 'user': 'user'})

    dialog_act = ''
    
    
    unique_id = f"{int(time.time() * 1000)}-{random.randint(1000, 9999)}"
    new_dict = {
    'id': [unique_id],
    'context': [dialog],
    'dataset_id':['dialog_inpainting'],
    'dialog_act':[''],
    'knowledge':[[]],
    'response': [''],
    
    }
    
    context = _process_dialog_context(dialog)
    
    dialog_act = tokenizer(dialog_act, add_special_tokens=False)["input_ids"] #tokenizer在main里面应该有
    # label = _process_response('')
    bos_token_needed = tokenizer.bos_token is not None
    full_sequence = [[tokenizer.bos_token_id]] if bos_token_needed else []

    full_sequence += [
        dialog_act,
        context,
        [tokenizer.eos_token_id]
    ]

    full_sequence = list(itertools.chain.from_iterable(full_sequence))

    sequences.append(full_sequence)
    # labels.append(label)
    
    # input_ids = sequences

    return_dict = {
    "input_ids": full_sequence,
    }
    
    dataset = Dataset.from_dict(new_dict)
    

    updated_dataset = dataset.map(
    lambda example: return_dict,
    batched=False,
    load_from_cache_file=False
    )
    return updated_dataset

#__function for dialog inpainting__
def main(run_mode: RunMode):
    training_args_class = Seq2SeqTrainingArguments
    parser_arguments = (ModelArguments, DataTrainingArguments if run_mode ==
                                                                 RunMode.TRAIN else DataPredictionArguments,
                        training_args_class)
    parser = HfArgumentParser(parser_arguments)

    raw_args = sys.argv[1:]
    json_index = -1 if raw_args[-1].endswith(".json") and (len(
        raw_args) == 1 or not raw_args[-2].startswith('-') or '=' in raw_args[-2]) else 0
    if len(raw_args) > 0 and raw_args[json_index].endswith(".json"):
        with open(raw_args[json_index]) as fp:
            json_args_dict = json.load(fp)
        del raw_args[json_index]

        if run_mode == RunMode.TRAIN:
            train_parser = HfArgumentParser(training_args_class)
            training_args_dict = vars(train_parser.parse_args(
                raw_args + ['--output_dir', json_args_dict['output_dir']]))
            training_args_dict.update(json_args_dict)
            json_args_dict = training_args_dict

        model_args, data_args, training_args = parser.parse_dict(
            json_args_dict, allow_extra_keys=True)
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    logging.info(data_args)

    logging.info(
        f"My rank is {training_args.local_rank} with {torch.cuda.device_count()} GPUs.")
    if training_args.local_rank != -1:
        torch.cuda.set_device(training_args.local_rank)

    if (
            os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty."
            "Use --overwrite_output_dir to overcome."
        )

    _setup_logging(training_args)

    config_class = get_config_class(model_args)

    config = config_class.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=model_args.use_auth_token,
    )
    if model_args.num_labels is not None:
        config.num_labels = model_args.num_labels

    tokenizer = AutoTokenizer.from_pretrained(
        get_tokenizer_name(config, model_args),
        cache_dir=model_args.cache_dir,
        use_fast=True,
        revision=model_args.model_revision,
        use_auth_token=model_args.use_auth_token,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    method_class = next(
        (m for m in method_classes if m.name == model_args.method), None)
    if method_class is None:
        raise Exception(f"No method class for name {model_args.method}.")
    method_definition: Method = method_class(
        model_args, data_args, config, tokenizer)
#junling need modify
    # Set seed before initializing model.
    set_seed(training_args.seed)

    model = method_definition.get_model(run_mode, config)
    model.config.keys_to_ignore_at_inference = [
        "decoder_attentions"
    ]
    model.config.num_beams = model_args.generation_beam_size
    model.config.max_length = model_args.generation_max_len
    model.config.do_sample = model_args.generation_do_sample
    model.config.length_penalty = model_args.generation_length_penalty
    model.config.no_repeat_ngram_size = model_args.generation_no_repeat_ngram_size
    model.config.uid_regularization = model_args.generation_uid_regularization
    model.config.num_return_sequences = model_args.num_return_sequences

    if run_mode == RunMode.TRAIN:
        extra_trainer_args = {
            'train_dataset': method_definition.get_train_dataset(),
            'eval_dataset': method_definition.get_validation_dataset(),
        }
    else:
        extra_trainer_args = {}

    data_collator = method_definition.get_data_collator()
    trainer_class = method_definition.get_trainer_class()

    trainer: Trainer = trainer_class(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=method_definition.compute_metrics,
        **extra_trainer_args,
    )
    trainer.add_callback(GPUMemoryCallback())


    if run_mode == RunMode.TRAIN:
        # Check for existing checkpoint to continue the training
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        resume_from_checkpoint = last_checkpoint if last_checkpoint is not None else None
        # Start training
        train_result = trainer.train(
            resume_from_checkpoint=resume_from_checkpoint)

        output_train_file = os.path.join(
            training_args.output_dir, "train_results.txt")
        if trainer.is_world_process_zero():
            with open(output_train_file, "w") as writer:
                logger.info("***** Train results *****")
                for key, value in sorted(train_result.metrics.items()):
                    logger.info(f"  {key} = {value}")
                    writer.write(f"{key} = {value}\n")

            # Need to save the state, since Trainer.save_model saves only the tokenizer with the model
            trainer.state.save_to_json(os.path.join(
                training_args.output_dir, "trainer_state.json"))
    
    elif run_mode == RunMode.PREDICT:
        # Input data
        
        with open('/cluster/scratch/wangjun/local_data/book_dataset_v4/business/business_ethics.json') as f:
            dataset = json.load(f)

        count_section = 0
        for key in dataset:
            
            if key not in ('book_statistics','chapter_concepts','chapter_questions'):
                count_section += 1
                # if(count_section == 1 or count_section == 2):
                #     continue
                # if(count_section == 5):
                #     break
                # print(dataset[key]['content'])
                section = dataset[key]['content']
                count = 0
                result = []
                document_title = str(key)
                for paragraph in section:
                    
                    # if len(paragraph) > 69:
                        # divide paragraph into sentences, and then generate response for each sentence
                    sentences = nltk.sent_tokenize(paragraph)
                    if(len(sentences) == 1):
                        document_title = str(sentences[0])
                        
                    elif(len(sentences) > 1):
                        # print('document_title',document_title)
                        # print('sentences',sentences)
                        dialog = []  # Initialize an empty dialog
                        author_num = []
                        test_datasets = []

                        # Generate dialog inpainting
                        for idx, sentence in enumerate(sentences):
                            if idx == 0:
                                test_dataset = generate_partial_dialog([sentence], document_title)
                            else:
                                test_dataset = generate_partial_dialog(dialog + [sentence], document_title)
                            
                            results = trainer.predict(test_dataset)
                            for example in test_dataset:
                                test_datasets.append(example)
                            results = method_definition.postprocess_predictions(results, test_dataset)

                            results_to_keep = []
                            for j in range(int(len(results) / model.config.num_return_sequences)):
                                lower_bound = j * model.config.num_return_sequences
                                upper_bound = j * model.config.num_return_sequences + model_args.num_sequences_to_keep
                                results_to_keep.extend(results[lower_bound:upper_bound])
                            results = results_to_keep

                            generated_sentence = results[0]

                            dialog.append(generated_sentence)  # Add the generated sentence as the first element
                            dialog.append(sentence)  # Add the current input sentence
                            author_num.append(0)
                            author_num.append(1)
                            
                        # Save the targeted content
                        output_data = {
                            "title": document_title,
                            "pid": str(uuid4()),
                            "passage": " ".join(sentences),
                            "sentences": sentences,
                            "author_num": author_num,
                            "utterances": dialog
                        }
                        if data_args.prediction_output_file is not None:
                            with open(data_args.prediction_output_file, 'a') as f:
                                json.dump(output_data, f, cls=NumpyEncoder)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        # sentences = [
        #     "Lesmahagow F.C.",
        #     "Lesmahagow Football Club is a Scottish football club, based in the town of Lesmahagow, South Lanarkshire.",
        #     "They were formed in 1885 and play at Craighead Park.",
        #     "Currently playing in the West of Scotland League Central District First Division.",
        #     "they wear Red And White Hooped Shirts, White Shorts, Red Socks and away colours are Blue Shirt White Trim, Blue Shorts, Blue Socks white trim.",
        #     "The club are sponsored by The Black Bull Inn, Lesmahagow."
        # ]

        # document_title = "Lesmahagow F.C."
        # dialog = []  # Initialize an empty dialog
        # author_num = []
        # test_datasets = []

        # # Generate dialog inpainting
        # for idx, sentence in enumerate(sentences):
        #     if idx == 0:
        #         test_dataset = generate_partial_dialog([sentence], document_title)
        #     else:
        #         test_dataset = generate_partial_dialog(dialog + [sentence], document_title)
            
        #     results = trainer.predict(test_dataset)
        #     for example in test_dataset:
        #         test_datasets.append(example)
        #     results = method_definition.postprocess_predictions(results, test_dataset)

        #     results_to_keep = []
        #     for j in range(int(len(results) / model.config.num_return_sequences)):
        #         lower_bound = j * model.config.num_return_sequences
        #         upper_bound = j * model.config.num_return_sequences + model_args.num_sequences_to_keep
        #         results_to_keep.extend(results[lower_bound:upper_bound])
        #     results = results_to_keep

        #     generated_sentence = results[0]

        #     dialog.append(generated_sentence)  # Add the generated sentence as the first element
        #     dialog.append(sentence)  # Add the current input sentence
        #     author_num.append(0)
        #     author_num.append(1)
            
        # # Save the targeted content
        # output_data = {
        #     "title": document_title,
        #     "pid": str(uuid4()),
        #     "passage": " ".join(sentences),
        #     "sentences": sentences,
        #     "author_num": author_num,
        #     "utterances": dialog
        # }
        
        
        # # with open('/cluster/scratch/wangjun/temp4/main_test_datasets.txt', 'w') as f:
        # #     f.write(str(test_datasets))
        # if data_args.prediction_output_file is not None:
        #     with open(data_args.prediction_output_file, 'wt') as f:
        #         json.dump(output_data, f, cls=NumpyEncoder)


    # elif run_mode == RunMode.PREDICT:
    #     # test_dataset = method_definition.get_test_dataset()
    #     #__junling modify__
    #     sentences = [
    #     "Lesmahagow F.C."
    #     ]
        
    #     document_title = "Lesmahagow F.C."

    #     test_dataset = generate_partial_dialog(sentences, document_title)
        
    #     # with open('/cluster/scratch/wangjun/temp4/test_dataset_attribute.txt', 'w') as f:
    #     #     f.write(str(test_dataset))
    #     with open('/cluster/scratch/wangjun/temp4/test_dataset_dialog_inpainting.txt', 'w') as f:
    #         for example in test_dataset:
    #             f.write(str(example) + '\n')
    #     # import pickle
    #     # obj2 = "123"
    #     # with open('/cluster/scratch/wangjun/temp4/test_dataset.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
    #     #     pickle.dump([test_dataset, obj2], f)
    #     # print('updated_dataset',updated_dataset)
    #     # for example in updated_dataset:
    #     #     print(str(example))
    #     #__junling modify__

    #     results = trainer.predict(test_dataset)
        
    #     with open('/cluster/scratch/wangjun/temp4/results_trainer_predict.txt', 'w') as f:
    #             f.write(str(results))
    #     # scores = results.scores #junling modify
    #     results = method_definition.postprocess_predictions(
    #         results,
    #         test_dataset
    #     )
    #     with open('/cluster/scratch/wangjun/temp4/results_postprocess_predictions.txt', 'w') as f:
    #             f.write(str(results))
    #     results_to_keep = []
    #     for idx in range(int(len(results) / model.config.num_return_sequences)):
    #         lower_bound = idx * model.config.num_return_sequences
    #         upper_bound = idx * model.config.num_return_sequences + model_args.num_sequences_to_keep
    #         results_to_keep.extend(results[lower_bound:upper_bound])
    #     results = results_to_keep
    #     with open('/cluster/scratch/wangjun/temp4/results_to_keep.txt', 'w') as f:
    #             f.write(str(results))
    #     if data_args.prediction_output_file is not None:
    #         with open(data_args.prediction_output_file, 'wt') as f:
    #             json.dump(
    #                 dataclasses.asdict(results) if type(
    #                     results) == PredictionOutput else results,
    #                 f,
    #                 cls=NumpyEncoder
    #             )
        #--junling modify--
        # for sample in test_dataset:
        #     with open('/cluster/scratch/wangjun/temp3/4_27_sample.txt', 'w') as f:
        #         f.write(str(sample))
        #--junling modify--
        # refs = [[sample["response"] for sample in test_dataset]] #junling modify

        # other metrics can be calculated with further jobs
        
        # bleu = BLEU()
        # score = bleu.corpus_score(results, refs)

        # scores = {
        #     "sacrebleu": score.score,
        #     "sacrebleu_signature": str(bleu.get_signature()),
        # }

        # if data_args.metric_output_file is not None:
        #     with open(data_args.metric_output_file, "wt") as f:
        #         json.dump(
        #             scores,
        #             f
        #         )
