import pandas as pd
from scipy.stats import pearsonr

# load your data
# df = pd.read_excel(file_path)

# # select the columns
# # columns = ['human_factual_consistency', 'human_relevance', 'Q_A_eval3']

# # pairwise correlation
# for i in range(len(columns)):
#     for j in range(i+1, len(columns)):
#         col1 = columns[i]
#         col2 = columns[j]
        
#         # compute correlation
#         corr, p_value = pearsonr(df[col1], df[col2])

#         print(f'Correlation between {col1} and {col2}:')
#         print(f'Pearson Correlation Coefficient: {corr}')
#         print(f'p-value: {p_value}\n')


import pandas as pd
from scipy.stats import spearmanr

def calculate_correlation(file_path, columns):
    df = pd.read_excel(file_path)

    # pairwise correlation
    for i in range(len(columns)):
        for j in range(i+1, len(columns)):
            col1 = columns[i]
            col2 = columns[j]
            # compute pearson correlation
            corr, p_value = pearsonr(df[col1], df[col2])
            print('____________pearson_________________')
            print(f'Correlation between {col1} and {col2}:')
            print(f'Pearson Correlation Coefficient: {corr}')
            print(f'p-value: {p_value}\n')
            
    for i in range(len(columns)):
        for j in range(i+1, len(columns)):
            col1 = columns[i]
            col2 = columns[j]
            # compute spearman correlation
            corr, p_value = spearmanr(df[col1], df[col2])
            print('____________spearman_________________')
            print(f'Correlation between {col1} and {col2}:')
            print(f'Spearman Correlation Coefficient: {corr}')
            print(f'p-value: {p_value}\n')


# Usage
columns = ['human_factual_consistency', 'human_relevance', 'QFactScore']
file_path = '/cluster/home/wangjun/dialog_inpainting/dialog_inpainting_implementation/code/qfactscore_key+post_human_eval_8_14_100.xlsx'

calculate_correlation(file_path, columns)

# keyword+post roberta == deberta
# Correlation between human_factual_consistency and human_relevance:
# Pearson Correlation Coefficient: 0.8626899735508381
# p-value: 8.292622865179489e-13

# Correlation between human_factual_consistency and Q_A_eval3:
# Pearson Correlation Coefficient: 0.4785062244317342
# p-value: 0.0017887040812288794

# Correlation between human_relevance and Q_A_eval3:
# Pearson Correlation Coefficient: 0.38219343715119947
# p-value: 0.01494346122276556

#gpt-3.5
# Correlation between human_factual_consistency and human_relevance:
# Pearson Correlation Coefficient: 0.6783671839134456
# p-value: 1.5082469549583251e-06

# Correlation between human_factual_consistency and Q_A_eval3:
# Pearson Correlation Coefficient: 0.3611111971350307
# p-value: 0.022061766180491296

# Correlation between human_relevance and Q_A_eval3:
# Pearson Correlation Coefficient: 0.3292510756746062
# p-value: 0.038026034520366014

#################################################################################
# entailment score

#keyword+post
# Correlation between human_factual_consistency and human_relevance:
# Pearson Correlation Coefficient: 0.8626899735508381
# p-value: 8.292622865179489e-13

# Correlation between human_factual_consistency and Q_A_eval3:
# Pearson Correlation Coefficient: 0.42913463443141014
# p-value: 0.005725014407852834

# Correlation between human_relevance and Q_A_eval3:
# Pearson Correlation Coefficient: 0.4267234314173179
# p-value: 0.006034084537741772


# gpt-3.5
# Correlation between human_factual_consistency and human_relevance:
# Pearson Correlation Coefficient: 0.6783671839134456
# p-value: 1.5082469549583251e-06

# Correlation between human_factual_consistency and Q_A_eval3:
# Pearson Correlation Coefficient: 0.43768784462341603
# p-value: 0.004736451205148509

# Correlation between human_relevance and Q_A_eval3:
# Pearson Correlation Coefficient: 0.53428841669725
# p-value: 0.00038360720378060616

#############################################################
# text span binary
# key+post
# Correlation between human_factual_consistency and human_relevance:
# Pearson Correlation Coefficient: 0.8626899735508381
# p-value: 8.292622865179489e-13

# Correlation between human_factual_consistency and Q_A_eval3:
# Pearson Correlation Coefficient: 0.29210549119955725
# p-value: 0.06740348577107932

# Correlation between human_relevance and Q_A_eval3:
# Pearson Correlation Coefficient: 0.3057762147373362
# p-value: 0.05500777760550515

#gpt-3.5
# Correlation between human_factual_consistency and human_relevance:
# Pearson Correlation Coefficient: 0.6783671839134456
# p-value: 1.5082469549583251e-06

# Correlation between human_factual_consistency and Q_A_eval3:
# Pearson Correlation Coefficient: 0.38682798996704
# p-value: 0.013671815919492668

# Correlation between human_relevance and Q_A_eval3:
# Pearson Correlation Coefficient: 0.389819383765292
# p-value: 0.012900723752366556


################################################
# generative model
# key+post
# Correlation between human_factual_consistency and human_relevance:
# Pearson Correlation Coefficient: 0.8626899735508381
# p-value: 8.292622865179489e-13

# Correlation between human_factual_consistency and Q_A_eval3:
# Pearson Correlation Coefficient: 0.4017244950317497
# p-value: 0.010186996072813481

# Correlation between human_relevance and Q_A_eval3:
# Pearson Correlation Coefficient: 0.36966552562711047
# p-value: 0.018891102610392722

#gpt-3.5
# Correlation between human_factual_consistency and human_relevance:
# Pearson Correlation Coefficient: 0.6783671839134456
# p-value: 1.5082469549583251e-06

# Correlation between human_factual_consistency and Q_A_eval3:
# Pearson Correlation Coefficient: 0.13504102400164977
# p-value: 0.4060825492380581

# Correlation between human_relevance and Q_A_eval3:
# Pearson Correlation Coefficient: 0.09932673654557606
# p-value: 0.5420020594149506

#####################################
#question score

# key+post
# ____________pearson_________________
# Correlation between human_factual_consistency and Q_A_eval3:
# Pearson Correlation Coefficient: 0.1822851665772441
# p-value: 0.26025918281529087

# ____________spearman_________________
# Correlation between human_factual_consistency and Q_A_eval3:
# Spearman Correlation Coefficient: 0.23274395919827784
# p-value: 0.14838540147774984

# ____________pearson_________________
# Correlation between human_relevance and Q_A_eval3:
# Pearson Correlation Coefficient: 0.09378310387093378
# p-value: 0.5648861838818389

# ____________spearman_________________
# Correlation between human_relevance and Q_A_eval3:
# Spearman Correlation Coefficient: 0.16396116599667238
# p-value: 0.31203493947445

# gpt3.5
# ____________pearson_________________
# Correlation between human_factual_consistency and human_relevance:
# Pearson Correlation Coefficient: 0.6783671839134456
# p-value: 1.5082469549583251e-06

# ____________spearman_________________
# Correlation between human_factual_consistency and human_relevance:
# Spearman Correlation Coefficient: 0.6842714554540208
# p-value: 1.1250224470872582e-06

# ____________pearson_________________
# Correlation between human_factual_consistency and Q_A_eval3:
# Pearson Correlation Coefficient: 0.06410459064768938
# p-value: 0.6943332229736943

# ____________spearman_________________
# Correlation between human_factual_consistency and Q_A_eval3:
# Spearman Correlation Coefficient: 0.050498558912887445
# p-value: 0.7569785854915623

# ____________pearson_________________
# Correlation between human_relevance and Q_A_eval3:
# Pearson Correlation Coefficient: -0.09229041931730245
# p-value: 0.5711232758939229

# ____________spearman_________________
# Correlation between human_relevance and Q_A_eval3:
# Spearman Correlation Coefficient: -0.0908295560377141
# p-value: 0.5772576899055579

#the best among all: entailment score