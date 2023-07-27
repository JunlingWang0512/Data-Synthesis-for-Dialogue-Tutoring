import pandas as pd
from scipy.stats import pearsonr

# load your data
df = pd.read_excel('/cluster/home/wangjun/dialog_inpainting/dialog_inpainting_implementation/code/7_27_qa_huamn_version6_gpt3.5.xlsx')

# select the columns
columns = ['human_factual_consistency', 'human_relevance', 'Q_A_eval3']

# pairwise correlation
for i in range(len(columns)):
    for j in range(i+1, len(columns)):
        col1 = columns[i]
        col2 = columns[j]
        
        # compute correlation
        corr, p_value = pearsonr(df[col1], df[col2])

        print(f'Correlation between {col1} and {col2}:')
        print(f'Pearson Correlation Coefficient: {corr}')
        print(f'p-value: {p_value}\n')


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

