import pandas as pd
import ast
import numpy as np

url = r"D:\df_for_MRR_2.csv"
df = pd.read_csv(url, header=0)
# convert strings representing lists in the df to real list
df['pred_order'] = df['pred_order'].apply(lambda x: ast.literal_eval(x))
df['gt_order'] = df['gt_order'].apply(lambda x: ast.literal_eval(x))
df['relevant'] = df['relevant'].apply(lambda x: ast.literal_eval(x))
df['indexes'] = df.apply(lambda x: [int(x.pred_order.index(value)) for value in x.gt_order], axis=1)
df['pred_relevant'] = df.apply(lambda x: [x.relevant[i] for i in x.indexes], axis=1)
df['recip_rank'] = df['pred_relevant'].apply(lambda x: 1 / (x.index(1) + 1) if 1 in x else -np.inf)
print(df['recip_rank'][df['recip_rank'] != -np.inf].mean())
