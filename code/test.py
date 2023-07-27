from transformers import T5Tokenizer, T5ForConditionalGeneration

tokenizer = T5Tokenizer.from_pretrained('google/flan-t5-large')
model = T5ForConditionalGeneration.from_pretrained('google/flan-t5-large')

# Example student answer
student_answer = "We hold that \u201can ethical person\u201d conveys the same sense as \u201ca moral person,\u201d and we do not regard religious belief as a requirement for acting ethically in business and the professions."

input_text = f"Translate this student's formal answer into casual, conversational language, but keep the original meaning the same: '{student_answer}'"
input_ids = tokenizer.encode(input_text, return_tensors="pt")

outputs = model.generate(input_ids, max_length=200, num_return_sequences=1)

decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(decoded)
