import openai
import json
import time
# Step 1: Extract utterances

def extract_utterances(data):
    utterances = data['utterances']
    dialog = []

    for idx, utterance in enumerate(utterances):
        role = "Teacher: " if idx % 2 == 0 else "Student: "
        dialog.append(f"{role} {utterance.replace(role + ': ', '')}")
    
    return " ".join(dialog)

# Step 2: Use GPT-3.5-turbo API for rewriting
def generate_response0(prompt, model):
    messeage_content = prompt
    print(messeage_content)
    
    while True:  # Loop indefinitely until successful
        try:
            completion = openai.ChatCompletion.create(
                model=model,
                messages=[{"role": "user", "content": messeage_content}]
            )
            return completion
        except:  # Catch any exception
            print("Error occurred while generating response. Retrying in 2 seconds...")
            time.sleep(2)  # Wait for 2 seconds before retrying
def rewrite_dialog(api_key, dialog):
    openai.api_key = api_key

    prompt = (f"Task: rewrite the following dialog so that both student and teacher can ask questions instead of only teacher asking questions, each question should be answered by the other person. The answer from student can be wrong and the teacher will correct it in the next turn.\n"
              "Instructions:\n"
              "1.You can flip some roles, If the teacher asks a question and the student answers, reformat it so that the student asks a related or clarifying question, and the teacher answers. But make sure teacher should ask at least half of questions."
              "2.You can make the conversations longer. For instance, after a student answers, make the teacher provide additional information or ask a follow-up question. But the dialog shouldn't contain information that wasn't in the original dialog"
              "3.Instead of students giving direct answers, you can make them sound more uncertain, prompting the teacher to provide further clarity."
              "4.Try to make the dialog more natural and coherent."
              f"\nmy current dialog is:\n{dialog}\n")
            #   "please output dialog in the same format as my input dialog, do not include newline in the output, use normal symbols so I can paste into json file. Use double \ instead of single \ in your output.")
    

    result = generate_response0(prompt,'gpt-3.5-turbo')
    prediction = result['choices'][0]['message']['content']
    
    return prediction

# Main Function
def generate_dataset(input_file, api_key):
    count = 0
    data = []
    with open(input_file, 'r') as f:
        decoder = json.JSONDecoder()
        text = f.read()
        while text:
            obj, idx = decoder.raw_decode(text)
            data.append(obj)
            text = text[idx:].lstrip()
    
    output_data = []

    for item in data:
        dialog = extract_utterances(item)
        rewritten = rewrite_dialog(api_key, dialog)
        
        output_data.append({
            "source": dialog,
            "rewritten": rewritten
        })
        count += 1
        print('count=',count)

    with open('/cluster/scratch/wangjun/keyword+post_flan-t5-xl_result/combined_gpt3.5_generated_train.json', 'w') as f:
        json.dump(output_data, f, indent=4)

# Example
input_file = '/cluster/scratch/wangjun/keyword+post_flan-t5-xl_result/combined.json'
api_key = ''  # Replace with your API key
# generate_dataset(input_file, api_key)
dialog = "Teacher: What can I say that will help us? Student: Properly defining an angle first requires that we define a ray. Teacher: What is a directed line segment? Student: A ray is a directed line segment. Teacher: What is a line segment? Student: It consists of one point on a line and all points extending in one direction from that point. Teacher: What is the first point of a ray? Student: The first point is called the endpoint of the ray. Teacher: What is another way we can refer to a specific ray? Student: We can refer to a specific ray by stating its endpoint and any other point on it. Teacher: What is the name of the ray in Figure 1? Student: The ray in Figure 1 can be named as ray EF, or in symbol form $$\\(\\overset{\\longrightarrow}{EF}.\\)$$"
prediction = rewrite_dialog(api_key,dialog)
print('prediction',prediction)

# prediction Teacher: What can I say that will help us understand angles better?
# Student: Properly defining an angle first requires that we define a ray.
# Teacher: That's right. So, what is a directed line segment?
# Student: Could you please explain what a directed line segment is?
# Teacher: A directed line segment is a line segment with a specific direction. It starts at one point and goes in a particular direction.
# Student: I see. And what is a line segment exactly?
# Teacher: A line segment is a part of a line with two endpoints. It consists of all the points on the line between those two endpoints.
# Student: I understand now. So, what is the first point of a ray called?
# Teacher: What would you call the first point of a ray?
# Student: Is the first point of a ray called the endpoint?
# Teacher: Yes, that's correct. The first point of a ray is called the endpoint.
# Student: Okay, got it. Can we refer to a specific ray in any other way?
# Teacher: How else can we identify or refer to a specific ray?
# Student: Is it possible to refer to a specific ray by stating its endpoint and any other point on it?
# Teacher: Absolutely! We can refer to a specific ray by mentioning its endpoint and any other point that lies on the ray.
# Student: Thank you for clarifying that. By the way, what is the name of the ray shown in Figure 1?
# Teacher: Do you know what the ray in Figure 1 is called?
# Student: Could you please tell me the name of the ray in Figure 1?
# Teacher: The ray in Figure 1 can be named as ray EF, or in symbol form $\overset{\longrightarrow}{EF}$.


# prompt
# Task: rewrite the following dialog so that both student and teacher can ask questions instead of only teacher asking questions, each question should be answered by the other person
# Instructions:
# 1.You can flip some roles, If the teacher asks a question and the student answers, reformat it so that the student asks a related or clarifying question, and the teacher answers. But make sure teacher should ask at least half of questions.
# 2.you can make the conversations longer. For instance, after a student answers, make the teacher provide additional information or ask a follow-up question. But the dialog shouldn't contain information that wasn't in the original dialog
# 3.Instead of students giving direct answers, you can make them sound more uncertain, prompting the teacher to provide further clarity.
# 4.Try to make the dialog more natural and coherent
# my dialog is:
# Teacher: What can I say that will help us?
# Student: Properly defining an angle first requires that we define a ray.
# Teacher: What is a directed line segment?
# Student: A ray is a directed line segment.
# Teacher: What is a line segment?
# Student: It consists of one point on a line and all points extending in one direction from that point.
# Teacher: What is the first point of a ray?
# Student: The first point is called the endpoint of the ray.
# Teacher: What is another way we can refer to a specific ray?
# Student: We can refer to a specific ray by stating its endpoint and any other point on it.
# Teacher: What is the name of the ray in Figure 1?
# Student: The ray in Figure 1 can be named as ray EF, or in symbol form $$\\(\\overset{\\longrightarrow}{EF}.\\)$$


# Student: What can I say that will help us?
# Teacher: Properly defining an angle first requires that we define a ray. Can you explain what a ray is?
# Student: A ray is a directed line segment, right?
# Teacher: Not quite. Let me clarify. A ray is actually a part of a line that starts at a fixed point, called the endpoint, and extends infinitely in one direction. It's important to remember that a ray has only one endpoint. Does that make sense?
# Student: Ah, I see now. So, a ray is like a part of a line that starts at a specific point and goes on forever in one direction. Thanks for clarifying that! 
# Teacher: You're welcome! Now, let's talk about line segments. Can you explain what a line segment consists of?
# Student: Sure! A line segment consists of one point on a line and all points extending in one direction from that point.
# Teacher: Not exactly. A line segment is actually a part of a line that has two endpoints. It's a finite section of a line with a definite length. Does that make sense?
# Student: Oh, I didn't get that part. So, a line segment is like a part of a line that has two specific endpoints and it has a finite length. Thanks for explaining!
# Teacher: Exactly! Now, let's dive deeper into rays. What is the first point of a ray called?
# Student: Um, the first point of a ray is called the endpoint, right?
# Teacher: That's correct! The endpoint is indeed the first point of a ray. Good job!
# Student: Thank you! I'm starting to understand this better. Now, can we refer to a specific ray in another way?
# Teacher: Absolutely! We can refer to a specific ray by stating its endpoint and any other point on it. This helps to clearly identify the ray we're talking about. Does that make sense?
# Student: Yes, it does! So, we can name a ray by mentioning its endpoint and any other point that lies on the ray. Thanks for clarifying!
# Teacher: You're welcome! Now, let's apply what we've learned. Can you tell me the name of the ray in Figure 1?
# Student: Um, the ray in Figure 1 can be named as ray EF, or in symbol form $$\(\overset{\longrightarrow}{EF}.\)$$ Is that correct?
# Teacher: Perfect! You named it correctly. It's indeed ray EF or symbolically written as $$\(\overset{\longrightarrow}{EF}.\)$$ Well done!