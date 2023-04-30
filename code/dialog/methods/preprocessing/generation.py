import itertools
import random
from dialog.methods.preprocessing.base import Preprocessor
import time
import numpy as np
class Seq2SeqPreprocessor(Preprocessor):

    def preprocess(self, features):
        sequences, labels = [], []
        for source, target in zip(features["input"], features["output"]):
            tokenized_source = self.tokenizer(
                source, 
                max_length=self.model_args.max_input_length,
                truncation=True
            )["input_ids"]
            tokenized_target = self.tokenizer(
                target, 
                max_length=self.model_args.max_output_length,
                truncation=True
            )["input_ids"]
            sequences.append(tokenized_source)
            labels.append(tokenized_target)

        return sequences, labels


class DocumentGroundedPreprocessor(Preprocessor):

    def _process_knowledge(self, knowledge):
        knowledge = self.tokenizer(
            self.model_args.knowledge_sep_token.join([f"{k['text']}" for k in knowledge]),
            add_special_tokens=False
        )["input_ids"]
        knowledge = self._truncate_knowledge(knowledge)
        return knowledge

    def _truncate_knowledge(self, knowledge):
        if self.model_args.knowledge_truncation_strategy == "right":
            knowledge = knowledge[:self.model_args.knowledge_max_tokens]
        return knowledge

    # def preprocess(self, features):#original code
    #     # with open('/cluster/scratch/feiclu/dialogue_inpainting4_14/features_preprocess_input.txt', 'w') as f:
    #     #         f.write(str(features))
    #     sequences, labels = [], []
    #     for context, dialog_act, knowledge, response in zip(
    #             features["context"],
    #             features["dialog_act"],
    #             features["knowledge"],
    #             features["response"]
    #     ):
    #         context = self._process_dialog_context(context)
    #         response = self._process_response(response)
    #         knowledge = self._process_knowledge(knowledge)
    #         dialog_act = self.tokenizer(dialog_act, add_special_tokens=False)["input_ids"]

    #         bos_token_needed = self.tokenizer.bos_token is not None
    #         full_sequence = [[self.tokenizer.bos_token_id]] if bos_token_needed else []

    #         full_sequence += [
    #             dialog_act,
    #             [self.tokenizer.convert_tokens_to_ids(self.model_args.knowledge_tag_token)],
    #             knowledge,
    #             context,
    #             [self.tokenizer.eos_token_id]
    #         ]

    #         full_sequence = list(itertools.chain.from_iterable(full_sequence))

    #         sequences.append(full_sequence)
    #         labels.append(response)
    #     # with open('/cluster/scratch/feiclu/dialogue_inpainting4_14/sequences_preprocess_output.txt', 'w') as f:
    #     #     f.write(str(features))
    #     # with open('/cluster/scratch/feiclu/dialogue_inpainting4_14/labels_preprocess_output.txt', 'w') as f:
    #     #     f.write(str(features))
    #     return sequences, labels
    # def preprocess(self, features):#no knowledge
    #     # with open('/cluster/scratch/feiclu/dialogue_inpainting4_14/features_preprocess_input.txt', 'w') as f:
    #     #         f.write(str(features))
    #     sequences, labels = [], []
    #     for context, dialog_act, response in zip(
    #             features["context"],
    #             features["dialog_act"],
    #             # features["knowledge"],
    #             features["response"]
    #     ):
    #         context = self._process_dialog_context(context)
    #         response = self._process_response(response)
    #         # knowledge = self._process_knowledge(knowledge)
    #         dialog_act = self.tokenizer(dialog_act, add_special_tokens=False)["input_ids"]

    #         bos_token_needed = self.tokenizer.bos_token is not None
    #         full_sequence = [[self.tokenizer.bos_token_id]] if bos_token_needed else []

    #         full_sequence += [
    #             dialog_act,
    #             # [self.tokenizer.convert_tokens_to_ids(self.model_args.knowledge_tag_token)],
    #             # knowledge,
    #             context,
    #             [self.tokenizer.eos_token_id]
    #         ]

    #         full_sequence = list(itertools.chain.from_iterable(full_sequence))

    #         sequences.append(full_sequence)
    #         labels.append(response)
    #     # with open('/cluster/scratch/feiclu/dialogue_inpainting4_14/sequences_preprocess_output.txt', 'w') as f:
    #     #     f.write(str(features))
    #     # with open('/cluster/scratch/feiclu/dialogue_inpainting4_14/labels_preprocess_output.txt', 'w') as f:
    #     #     f.write(str(features))
    #     return sequences, labels
    def preprocess(self, features): #dialogue inpainting
        sequences, labels = [], []
        mask_contents = []
        responses = []
        index_to_mask = 0 #initialize
        with open('/cluster/scratch/feiclu/temp3/features_len.txt', 'w') as f:
                f.write(str(len(features)))
        
        for context, dialog_act, response in zip(
                features["context"],
                features["dialog_act"],
                # features["knowledge"],
                features["response"]
        ):
            # Add response to the context
            context.append({'text': response, 'user': 'system', 'dialog_act': ''})
            mask_content = ''
            responses.append(response)
            #--VERSION2 MASK UTTERANCE AT RANDOM--
            if len(context) > 0:
                previous_index = index_to_mask
                # Generate a random index to choose from context
                index_to_mask = np.random.randint(0, len(context))
                # random.seed(int(time.time()))
                # index_to_mask = random.randint(0, len(context) - 1)
                if(index_to_mask==previous_index):
                    index_to_mask = np.random.randint(0, len(context))
                
                

                # Mask a 'text' in the context
                mask_content = context[index_to_mask]['text']
                context[index_to_mask]['text'] = '<extra_id_0>'
            mask_contents.append(mask_content)
            
            #--VERSION2 MASK UTTERANCE AT RANDOM--

            context = self._process_dialog_context(context)
            # response = self._process_response(response)
            # response = self._process_response(mask_content) #junling modify
            dialog_act = self.tokenizer(dialog_act, add_special_tokens=False)["input_ids"]
            label = self._process_response(mask_content)

            bos_token_needed = self.tokenizer.bos_token is not None
            full_sequence = [[self.tokenizer.bos_token_id]] if bos_token_needed else []

            full_sequence += [
                dialog_act,
                context,
                [self.tokenizer.eos_token_id]
            ]

            full_sequence = list(itertools.chain.from_iterable(full_sequence))

            sequences.append(full_sequence)
            labels.append(label)
        import pickle
        # Save mask_contents to a pickle file
        with open("/cluster/scratch/feiclu/temp3/mask_contents.pkl", "wb") as f:
            pickle.dump(mask_contents, f)
        with open("/cluster/scratch/feiclu/temp3/responses.pkl", "wb") as f:
            pickle.dump(responses, f)
        # return sequences, labels
        return sequences, labels, mask_contents #junling modify


class DocumentGroundedPreprocessorWithKnowledgeLabels(DocumentGroundedPreprocessor):

    def preprocess(self, features):
        knowledge_labels = []
        sequences, labels = super().preprocess(features)

        for sample in features["knowledge"]:
            knowledge_labels.append(self.tokenizer(sample[0]["text"])["input_ids"])

        return sequences, labels, knowledge_labels

class DocumentGroundedPreprocessorWithCTRLTokens(DocumentGroundedPreprocessor):

    def _process_ctrl_tokens(self, ctrl_tokens):
        ctrl_tokens = self.tokenizer(
            ctrl_tokens,
            add_special_tokens=False
        )["input_ids"]
        ctrl_tokens = self._truncate_knowledge(ctrl_tokens)
        return ctrl_tokens

    def preprocess(self, features):
        sequences, labels = [], []
        for context, dialog_act, knowledge, ctrl_tokens, response in zip(
                features["context"],
                features["dialog_act"],
                features["knowledge"],
                features["ctrl_tokens"],
                features["response"]
        ):
            context = self._process_dialog_context(context)
            response = self._process_response(response)
            knowledge = self._process_knowledge(knowledge)
            if not self.data_args.is_training:
                ctrl_tokens = "<high-prec> <entailed>"
            ctrl_tokens = self._process_ctrl_tokens(ctrl_tokens)
            dialog_act = self.tokenizer(dialog_act, add_special_tokens=False)["input_ids"]

            bos_token_needed = self.tokenizer.bos_token is not None
            full_sequence = [[self.tokenizer.bos_token_id]] if bos_token_needed else []

            full_sequence += [
                dialog_act,
                ctrl_tokens,
                knowledge,
                context,
                [self.tokenizer.eos_token_id]
            ]

            full_sequence = list(itertools.chain.from_iterable(full_sequence))

            sequences.append(full_sequence)
            labels.append(response)

        return sequences, labels

class DocumentGroundedPreprocessorForDensityRatio(DocumentGroundedPreprocessorWithCTRLTokens):

    def preprocess(self, features):
        sequences, sequences_ctrl, labels = [], [], []
        for idx, (context, dialog_act, knowledge, response) in enumerate(zip(
                features["context"],
                features["dialog_act"],
                features["knowledge"],
                features["response"]
        )):
            context = self._process_dialog_context(context)
            response = self._process_response(response)
            knowledge = self._process_knowledge(knowledge)
            if "ctrl_tokens" in features:
                if not self.data_args.is_training:
                    ctrl_tokens = "<high-prec> <entailed>"
                ctrl_tokens = self._process_ctrl_tokens(ctrl_tokens)
            else: 
                ctrl_tokens = []
            dialog_act = self.tokenizer(dialog_act, add_special_tokens=False)["input_ids"]

            bos_token_needed = self.tokenizer.bos_token is not None
            full_sequence = [[self.tokenizer.bos_token_id]] if bos_token_needed else []
            full_sequence_ctrl = [[self.tokenizer.bos_token_id]] if bos_token_needed else []

            full_sequence += [
                dialog_act,
                knowledge,
                context,
                [self.tokenizer.eos_token_id]
            ]

            full_sequence_ctrl += [
                dialog_act,
                ctrl_tokens,
                knowledge,
                context,
                [self.tokenizer.eos_token_id]
            ]

            full_sequence = list(itertools.chain.from_iterable(full_sequence))
            full_sequence_ctrl = list(itertools.chain.from_iterable(full_sequence_ctrl))

            sequences.append(full_sequence)
            sequences_ctrl.append(full_sequence_ctrl)
            labels.append(response)

        return sequences, sequences_ctrl, labels

class DocumentGroundedPreprocessorForNoisyChannel(DocumentGroundedPreprocessor):

    def preprocess(self, features):
        sequences, sequences_lm, labels = [], [], []
        for idx, (context, dialog_act, knowledge, response) in enumerate(zip(
                features["context"],
                features["dialog_act"],
                features["knowledge"],
                features["response"]
        )):
            context = self._process_dialog_context(context)
            response = self._process_response(response)
            knowledge = self._process_knowledge(knowledge)
            no_knowledge = self._process_knowledge([])
            dialog_act = self.tokenizer(dialog_act, add_special_tokens=False)["input_ids"]

            bos_token_needed = self.tokenizer.bos_token is not None
            full_sequence = [[self.tokenizer.bos_token_id]] if bos_token_needed else []
            full_sequence_lm = [[self.tokenizer.bos_token_id]] if bos_token_needed else []

            full_sequence += [
                dialog_act,
                knowledge,
                context,
                [self.tokenizer.eos_token_id]
            ]

            full_sequence_lm += [
                dialog_act,
                no_knowledge,
                context,
                [self.tokenizer.eos_token_id]
            ]

            full_sequence = list(itertools.chain.from_iterable(full_sequence))
            full_sequence_lm = list(itertools.chain.from_iterable(full_sequence_lm))

            sequences.append(full_sequence)
            sequences_lm.append(full_sequence_lm)
            labels.append(response)

        return sequences, sequences_lm, labels