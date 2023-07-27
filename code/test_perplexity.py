import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

model.eval()
if torch.cuda.is_available():
    model.to('cuda')

def calculate_perplexity(sentences):
    sentence1, sentence2 = sentences
    sentence = sentence1 + " " + sentence2

    inputs = tokenizer(sentence, return_tensors='pt')

    if torch.cuda.is_available():
        inputs = inputs.to('cuda')

    with torch.no_grad():
        outputs = model(**inputs, labels=inputs['input_ids'])
        loss = outputs.loss

    perplexity = torch.exp(loss)

    return perplexity.item()
