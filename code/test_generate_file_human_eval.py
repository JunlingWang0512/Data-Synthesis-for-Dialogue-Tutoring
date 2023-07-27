import json
import pandas as pd
import ast
def search_json(json_file, context, target_utterance):
    # file_path = '/cluster/scratch/wangjun/GPT_results/GPT_3.5/business_search_output.json'

    data = []
    with open(json_file, 'r') as f:
        decoder = json.JSONDecoder()
        text = f.read()
        while text:
            obj, idx = decoder.raw_decode(text)
            data.append(obj)
            text = text[idx:].lstrip()

    # with open(json_file) as f:
    #     data = json.load(f)
    # print("target_utterances",target_utterance)
    for item in data:
        # if context == item["sentences"]:
        for i in range(1, len(item["utterances"])):
            # print('item["utterances"][i]',item["utterances"][i])
            if item["utterances"][i] == target_utterance:
                return [item["utterances"][i-1], target_utterance]
    return None

def main():
    df = pd.read_excel('/cluster/scratch/wangjun/local_data/human_eval/human_eval_dataset_7_25.xlsx')
    new_rows = []

    for _, row in df.iterrows():
        context = row['context']
        tuple_ = row['tuple']
        category = row['category']
        answer = row['answer']
        # print("tuple_",tuple_)
        # tuple2 = ast.literal_eval(tuple_)

        json_file = f'/cluster/scratch/wangjun/GPT_results/GPT_3.5/{category}_search_output.json'
        new_tuple = search_json(json_file, context, answer)
        
        if new_tuple is not None:
            new_row = {
                "context": context,
                "tuple": new_tuple,
                "category": category,
                "human_relevance": "",
                "human_factual_consistency":"",
                "strategy": "GPT-3.5",
                "question": new_tuple[0],
                "answer": new_tuple[1]
            }
            new_rows.append(new_row)
        else:
            new_row = {
                "context": context,
                "tuple": "",
                "category": category,
                "human_relevance": "",
                "human_factual_consistency":"",
                "strategy": "GPT-3.5",
                "question": "",
                "answer": ""
            }
            new_rows.append(new_row)


    new_df = pd.DataFrame(new_rows, columns=["context","tuple","category","human_relevance","human_factual_consistency","strategy","question","answer"])
    new_df.to_excel('new_eval_file.xlsx', index=False)

if __name__ == "__main__":
    main()
