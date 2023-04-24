
# import itertools

# from dialog.methods.preprocessing.base import Preprocessor
import time
import random
import numpy as np
# def get_tokenizer_name(config, model_args):
#     if "density_ratio" in model_args.method:
#         return config.language_model_tokenizer_name_or_path
#     elif "noisy_channel" in model_args.method:
#         return config.direct_model_tokenizer_name_or_path
#     elif model_args.tokenizer_name:
#         return model_args.tokenizer_name
#     else:
#         return model_args.model_name_or_path
# config = {
#         "method": "document_grounded_generation",
#         "model_name_or_path": "google/flan-t5-base",
#         "model_description": "flan_t5_base_baseline",
#         "train_dataset": "wow_DI",
#         "test_datasets": ["wow_DI"],
#         "train_dataset_config_name": "response_generation",
#         "test_dataset_config_name": "response_generation",
#         "expert_dataset_name": "wow",
#         "anti_expert_dataset_name": "faithdial",
#         "expert_dataset_config_name": "cape_expert",
#         "anti_expert_dataset_config_name": "hallucinated_response",
#         "dataset_train_split": "train",
#         "dataset_val_split": "validation",
#         "dataset_test_split": "test",
#         "per_device_train_batch_size": 4,
#         "gradient_accumulation_steps": 8,
#         "per_device_eval_batch_size": 8,
#         "gpu_mem_train": 12,
#         "gpu_mem_test": 10,
#         "num_epochs": 10,
#         "num_expert_epochs": 5,
#         "gpu_mem_fisher": 12
#     }
# training_args_class = Seq2SeqTrainingArguments
# parser_arguments = (ModelArguments, DataTrainingArguments if run_mode ==
#                                                                 RunMode.TRAIN else DataPredictionArguments,
#                     training_args_class)
# parser = HfArgumentParser(parser_arguments)

# raw_args = sys.argv[1:]
# json_index = -1 if raw_args[-1].endswith(".json") and (len(
#     raw_args) == 1 or not raw_args[-2].startswith('-') or '=' in raw_args[-2]) else 0
# if len(raw_args) > 0 and raw_args[json_index].endswith(".json"):
#     with open(raw_args[json_index]) as fp:
#         json_args_dict = json.load(fp)
#     del raw_args[json_index]

#     if run_mode == RunMode.TRAIN:
#         train_parser = HfArgumentParser(training_args_class)
#         training_args_dict = vars(train_parser.parse_args(
#             raw_args + ['--output_dir', json_args_dict['output_dir']]))
#         training_args_dict.update(json_args_dict)
#         json_args_dict = training_args_dict

#     model_args, data_args, training_args = parser.parse_dict(
#         json_args_dict, allow_extra_keys=True)
# else:
#     model_args, data_args, training_args = parser.parse_args_into_dataclasses()
# tokenizer = AutoTokenizer.from_pretrained(
#         get_tokenizer_name(config, model_args),
#         cache_dir=model_args.cache_dir,
#         use_fast=True,
#         revision=model_args.model_revision,
#         use_auth_token=model_args.use_auth_token,
#     )
# if tokenizer.pad_token is None:
#     tokenizer.pad_token = tokenizer.eos_token
# def _process_dialog_context(self, context):
#         context = [self._tokenize_with_special_tokens(turn) for turn in context]
#         context = self._truncate_to_max_length(context)
#         return list(itertools.chain.from_iterable(context))

# def _process_knowledge(self, knowledge):
#         knowledge = self.tokenizer(
#             self.model_args.knowledge_sep_token.join([f"{k['text']}" for k in knowledge]),
#             add_special_tokens=False
#         )["input_ids"]
#         knowledge = self._truncate_knowledge(knowledge)
#         return knowledge

# def _truncate_knowledge(self, knowledge):
#         if self.model_args.knowledge_truncation_strategy == "right":
#             knowledge = knowledge[:self.model_args.knowledge_max_tokens]
#         return knowledge
def preprocess(features):#junling modify
#     with open('/cluster/scratch/wangjun/dialogue_inpainting4_14/features_preprocess_input.txt', 'w') as f:
#             f.write(str(features))
    sequences, labels = [], []
    filename = 'test_generation_output4.txt'
    count = 1
    mark_ratio=0.8
    index_to_mask = 0 #initialize
    with open(filename, 'w') as f:
        for context, dialog_act, response in zip(
                features["context"],
                features["dialog_act"],
                # features["knowledge"],
                features["response"]
        ):
            # Add response to the context
            context.append({'text': response, 'user': 'system', 'dialog_act': ''})
            mask_content = ''
            
            #--VERSION2 MASK UTTERANCE AT RANDOM--
            if len(context) > 0:
                # random.seed(int(time.time()))
                # Generate a random index to choose from context
                previous_index = index_to_mask
                index_to_mask = np.random.randint(0, len(context))
                # index_to_mask = random.randint(0, len(context) - 1)
                if(index_to_mask==previous_index):
                    index_to_mask = np.random.randint(0, len(context))
                    # index_to_mask = random.randint(0, len(context) - 1)
                # Mask a 'text' in the context
                mask_content = context[index_to_mask]['text']
                context[index_to_mask]['text'] = '<extra_id_0>'
            # context = _process_dialog_context(context)
            # response = _process_response(response)
            # knowledge = _process_knowledge(knowledge)
            # dialog_act = tokenizer(dialog_act, add_special_tokens=False)["input_ids"]
            # label = _process_response(mark_content)      

            # bos_token_needed = self.tokenizer.bos_token is not None
            # full_sequence = [[self.tokenizer.bos_token_id]] if bos_token_needed else []

            # full_sequence += [
            #     dialog_act,
            #     [self.tokenizer.convert_tokens_to_ids(self.model_args.knowledge_tag_token)],
            #     knowledge,
            #     context,
            #     [self.tokenizer.eos_token_id]
            # ]

            # full_sequence = list(itertools.chain.from_iterable(full_sequence))

            # sequences.append(full_sequence)
            # labels.append(label)
            
            f.write(f'count: {count}\n')
            count += 1
            f.write(f'context: {context}\n')
            f.write(f'dialog_act: {dialog_act}\n')
            # f.write(f'knowledge: {knowledge}\n')
            # f.write(f'label: {label}\n')
            f.write(f'mask_content: {mask_content}\n')
            f.write(f'index_to_mask: {index_to_mask}\n')
            f.write(f'len(context): {len(context)}\n')
            f.write('\n')  # add a blank line between iterations


       

        # bos_token_needed = self.tokenizer.bos_token is not None
        # full_sequence = [[self.tokenizer.bos_token_id]] if bos_token_needed else []

        # full_sequence += [
        #     dialog_act,
        #     [self.tokenizer.convert_tokens_to_ids(self.model_args.knowledge_tag_token)],
        #     knowledge,
        #     context,
        #     [self.tokenizer.eos_token_id]
        # ]

        # full_sequence = list(itertools.chain.from_iterable(full_sequence))

        # sequences.append(full_sequence)
        # labels.append(response)
    # with open('/cluster/scratch/wangjun/dialogue_inpainting4_14/sequences_preprocess_output.txt', 'w') as f:
    #     f.write(str(features))
    # with open('/cluster/scratch/wangjun/dialogue_inpainting4_14/labels_preprocess_output.txt', 'w') as f:
    #     f.write(str(features))
    return "xxx"
with open('features_preprocess_input.txt', 'r') as f:
    content = f.read()
    features = eval(content)
preprocess(features)