# Beam Search
import ast
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
import re
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from deepctr_torch.inputs import SparseFeat, DenseFeat, get_feature_names
from deepctr_torch.models import *

columns = ["uid", "w1_duration", "w1_num_likes", "w1_num_comments", "w1_watched_time", "w1_like", "w2_duration", "w2_num_likes", "w2_num_comments", "w2_watched_time", "w2_like", "w3_duration", "w3_num_likes", "w3_num_comments", "w3_watched_time", "w3_like", "w4_duration", "w4_num_likes", "w4_num_comments", "w4_watched_time", "w4_like", "w5_duration", "w5_num_likes", "w5_num_comments",
           "w5_watched_time", "w5_like", "w6_duration", "w6_num_likes", "w6_num_comments", "w6_watched_time", "w6_like", "w7_duration", "w7_num_likes", "w7_num_comments", "w7_watched_time", "w7_like", "w8_duration", "w8_num_likes", "w8_num_comments", "w8_watched_time", "w8_like", "w9_duration", "w9_num_likes", "w9_num_comments", "w9_watched_time", "w9_like", "w10_duration",
           "w10_num_likes", "w10_num_comments", "w10_watched_time", "w10_like", "c1_duration", "c1_num_likes", "c1_num_comments", "c2_duration", "c2_num_likes", "c2_num_comments", "c3_duration", "c3_num_likes", "c3_num_comments", "c4_duration", "c4_num_likes", "c4_num_comments", "t1_duration", "t1_num_likes", "t1_num_comments", "p_like", "p_has_next", "p_effective_view"]

df_train = pd.read_csv('dataset/final_input/final_train.csv', header=0)
df_test = pd.read_csv('dataset/final_input/final_test', header=0)

df_train = df_train.drop(['uid'], axis=1)
df_test = df_test.drop(['uid'], axis=1)
columns.remove('uid')

sparse_features = [f'w{i}_like' for i in range(1, 11)]
target = ["p_like", "p_has_next", "p_effective_view"]
dense_features = [feature for feature in columns if feature in list(set(columns) - set(sparse_features) - set(target))]

for column in columns:
    df_train[column] = df_train[column].astype('float')
    df_test[column] = df_test[column].astype('float')

# Label Encoding for sparse features,and do simple Transformation for dense features
for feat in sparse_features:
    lbe = LabelEncoder()
    df_train[feat] = lbe.fit_transform(df_train[feat])
    df_test[feat] = lbe.fit_transform(df_test[feat])

mms = MinMaxScaler(feature_range=(0, 1))
df_train[dense_features] = mms.fit_transform(df_train[dense_features])
df_test[dense_features] = mms.fit_transform(df_test[dense_features])

# Count #unique features for each sparse field,and record dense feature field name
fixlen_feature_columns = [SparseFeat(feat, vocabulary_size=df_train[feat].max() + 1, embedding_dim=8)
                          for feat in sparse_features] + [DenseFeat(feat, 1, )
                                                          for feat in dense_features]

dnn_feature_columns = fixlen_feature_columns
linear_feature_columns = fixlen_feature_columns

feature_names = get_feature_names(
        linear_feature_columns + dnn_feature_columns)

# Generate final_input data for model

train_model_input = {name: df_train[name] for name in feature_names}
test_model_input = {name: df_test[name] for name in feature_names}

# Define Model,train,predict and evaluate
device = 'cpu'
use_cuda = False
if use_cuda and torch.cuda.is_available():
    print('cuda ready...')
    device = 'cuda:0'

# No autodis
model = MMOE(dnn_feature_columns, num_experts=16, task_types=['binary', 'binary', 'binary'], l2_reg_embedding=1e-5, task_names=target, device=device)
# Autodis
# model = MMOE(dnn_feature_columns, task_types=['binary', 'binary', 'binary'], l2_reg_embedding=1e-5, task_names=target, device=device, use_autodis=True)
# Autodis + Transformers
# model = MMOE(dnn_feature_columns, task_types=['binary', 'binary', 'binary'], l2_reg_embedding=1e-5, task_names=target, device=device, use_autodis=True, use_transformers=True)

model.compile("adam", loss=["binary_crossentropy", "binary_crossentropy", "binary_crossentropy"], metrics=['accuracy', 'accuracy', 'accuracy'])

history = model.fit(train_model_input, df_train[target].values, batch_size=64, epochs=100, verbose=2, validation_split=0.1, shuffle=True)
pred_ans = model.predict(test_model_input, 256)
for i, target_name in enumerate(target):
    print("%s test AUC" % target_name, round(roc_auc_score(df_test[target[i]].values, pred_ans[:, i]), 4))

# Save the model
torch.save(model, 'model.pth')
