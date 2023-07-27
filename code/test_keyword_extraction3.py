from transformers import BartTokenizer, BartForConditionalGeneration

def summarize_text_bart(text):
    model_name = 'facebook/bart-large-cnn'
    model = BartForConditionalGeneration.from_pretrained(model_name)
    tokenizer = BartTokenizer.from_pretrained(model_name)

    inputs = tokenizer([text], max_length=1024, return_tensors='pt', truncation=True)
    summary_ids = model.generate(inputs['input_ids'], num_beams=4, max_length=200, early_stopping=True)
    summary = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids]

    return summary[0]

text = "Identify the amplitude, $$\\(|A|.\\)$$  Identify the period, $$\\(P = \\frac{2\\pi},{|B|}.\\)$$ Start at the origin, with the function increasing to the right if $$\\(A\\)$$ is positive or decreasing if $$\\(A\\)$$ is negative. At $$\\(x = \\frac{\\pi},{2|B|}\\)$$ there is a local maximum for $$\\(A > 0\\)$$ or a minimum for $$\\(A < 0,\\)$$ with $$\\(y = A.\\)$$ The curve returns to the x-axis at $$\\(x = \\frac{\\pi},{|B|}.\\)$$ There is a local minimum for $$\\(A > 0\\)$$ (maximum for $$\\(A < 0\\)$$ ) at $$\\(x = \\frac{3\\pi},{2| B|}\\)$$ with $$\\(y = â€“A.\\)$$ The curve returns again to the x-axis at $$\\(x = \\frac{2\\pi},{|B|}.\\)$$"
print(summarize_text_bart(text))
