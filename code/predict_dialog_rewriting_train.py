#dialog rewriting train
import json
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
import peft
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training, TaskType
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
# Load Data
with open('/cluster/scratch/wangjun/keyword+post_flan-t5-xl_result/combined_gpt3.5_generated_train.json', 'r') as file:
    data = json.load(file)

source_texts = [entry['source'] for entry in data]
target_texts = [entry['rewritten'] for entry in data]

# Tokenization
# PROMPT = "Rewrite the following dialogue, so that the teacher ask questions deeper and deeper, help to guide the student to learn the knowledge. Dialogue: "
PROMPT = "Rewrite the dialogue provided below. The teacher should ask questions deeper and deeper, guiding the student towards a clearer understanding of the topic. The rewritten dialogue should contain the same or a greater number of conversational turns compared to the original dialogue, ensuring it flows naturally. Dialogue:"
source_texts_with_prompt = [PROMPT + text for text in source_texts]




model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-xl",  device_map='auto')
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-xl")

 # Define LoRA Config
lora_config = LoraConfig(
r=10,
lora_alpha=32,
target_modules=["q", "v"],
lora_dropout=0.05,
bias="none",
task_type=peft.TaskType.SEQ_2_SEQ_LM
)
# prepare int-8 model for training
model = prepare_model_for_int8_training(model)
# add LoRA adaptor
model = get_peft_model(model, lora_config)
#--junling modify--








# tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xl")
source_encodings = tokenizer(source_texts_with_prompt, padding=True, truncation=True, return_tensors="pt")
target_encodings = tokenizer(target_texts, padding=True, truncation=True, return_tensors="pt")

# Model Initialization
# model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-xl")

# Dataset Preparation
class DialogDataset(torch.utils.data.Dataset):
    def __init__(self, source_encodings, target_encodings):
        self.source_encodings = source_encodings
        self.target_encodings = target_encodings

    def __getitem__(self, idx):
        return {
            "input_ids": self.source_encodings["input_ids"][idx],
            "attention_mask": self.source_encodings["attention_mask"][idx],
            "labels": self.target_encodings["input_ids"][idx]
        }

    def __len__(self):
        return len(self.source_encodings.input_ids)

dataset = DialogDataset(source_encodings, target_encodings)

# Training Setup
training_args = TrainingArguments(
    per_device_train_batch_size=1,
    num_train_epochs=10,
    logging_dir="/cluster/scratch/wangjun/dialog_rewritting/log",
    output_dir="/cluster/scratch/wangjun/dialog_rewritting/output",
    learning_rate=6.25e-5,
    gradient_accumulation_steps=4,
    fp16=True

)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)

trainer.train()

# Save the trained model and tokenizer
model_save_path = "/cluster/scratch/wangjun/dialog_rewritting/model/model_3/"
# trainer.save_model(model_save_path)
# tokenizer.save_pretrained(model_save_path)

# peft_model_id= str(training_args.output_dir).replace("checkpoints", "models/epoch-best")
trainer.model.save_pretrained(model_save_path)
tokenizer.save_pretrained(model_save_path)
#load model and tokenizer:
# Load the model and tokenizer
# model = T5ForConditionalGeneration.from_pretrained(model_save_path)
# tokenizer = T5Tokenizer.from_pretrained(model_save_path)


#inference
# model = AutoModelForSeq2SeqLM.from_pretrained(model_save_path,  device_map='auto')
# tokenizer = AutoTokenizer.from_pretrained(model_save_path)
# # Inference Function
# def generate_dialogue(model, input_text):
#     input_with_prompt = PROMPT + input_text
#     input_encoding = tokenizer(input_with_prompt, return_tensors="pt", truncation=True, padding=True)
#     output_ids = model.generate(input_encoding["input_ids"])
#     return tokenizer.decode(output_ids[0], skip_special_tokens=True)

# sample_source = "[Teacher]What's the topic for today? [Student]We're discussing AI."
# print(generate_dialogue(model, sample_source))


#memory exeed的主要问题是数据的sequence过长，尝试用其他encoding方案或者缩短句子长度，不然很难放入模型中，效果也未必好