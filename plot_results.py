import sys

sys.path.append(r'/models/Multitask-Recommendation-Library')

import pandas as pd
from models.omoe import OMoEModel
from models.mmoe import MMoEModel
from models.ple import PLEModel
from models.aitm import AITMModel
from models.metaheac import MetaHeacModel

from datasets.aliexpress import AliExpressDataset
from datasets.my_dataset import MyDataset


def get_dataset(name, df):
    return MyDataset(df)


import tqdm
import torch
from torch.utils.data import DataLoader

df_test = ...
df_test = pd.read_csv('dataset/final_input/final_test.csv')
test_dataset = get_dataset('', df_test)
test_data_loader = DataLoader(test_dataset, batch_size=256, num_workers=4, shuffle=False)

field_dims = test_dataset.field_dims
numerical_num = test_dataset.numerical_num

device = torch.device('cpu')

MODEL_PATH = r'/models/Multitask-Recommendation-Library/results/my_dataset_metaheac.pt'


def load_model(name, dataset, path):
    """
    Hyperparameters are empirically determined, not opitmized.
    """

    categorical_field_dims = dataset.field_dims
    numerical_num = dataset.numerical_num
    expert_num = 8
    task_num = 3
    embed_dim = 128
    if name == 'omoe':
        print("Model: OMoE")
        model = OMoEModel(categorical_field_dims, numerical_num, embed_dim=embed_dim, bottom_mlp_dims=(512, 256), tower_mlp_dims=(128, 64), task_num=task_num, expert_num=expert_num, dropout=0.2)
    elif name == 'mmoe':
        print("Model: MMoE")
        model = MMoEModel(categorical_field_dims, numerical_num, embed_dim=embed_dim, bottom_mlp_dims=(512, 256), tower_mlp_dims=(128, 64), task_num=task_num, expert_num=expert_num, dropout=0.2)
    elif name == 'ple':
        print("Model: PLE")
        model = PLEModel(categorical_field_dims, numerical_num, embed_dim=embed_dim, bottom_mlp_dims=(512, 256), tower_mlp_dims=(128, 64), task_num=task_num, shared_expert_num=int(expert_num / 2), specific_expert_num=int(expert_num / 2), dropout=0.2)
    elif name == 'aitm':
        print("Model: AITM")
        model = AITMModel(categorical_field_dims, numerical_num, embed_dim=embed_dim, bottom_mlp_dims=(512, 256), tower_mlp_dims=(128, 64), task_num=task_num, dropout=0.2)
    elif name == 'metaheac':
        print("Model: MetaHeac")
        model = MetaHeacModel(categorical_field_dims, numerical_num, embed_dim=embed_dim, bottom_mlp_dims=(512, 256), tower_mlp_dims=(128, 64), task_num=task_num, expert_num=expert_num, critic_num=5, dropout=0.2)
    else:
        model = torch.load(path)
        return model

    model.load_state_dict(torch.load(path))
    return model


model_name = 'metaheac'
model = load_model(model_name, test_dataset, MODEL_PATH).to(device)


def get_predictions(model, df_model_input):
    """

    :param model:
    :param df_model_input:
    :return: list of lists
    """

    test_dataset = get_dataset('', df_model_input)
    test_data_loader = DataLoader(test_dataset, batch_size=256, num_workers=4, shuffle=False)

    model.eval()
    predicts_dict = {}
    task_num = 3
    for i in range(task_num):
        predicts_dict[i] = list()
    with torch.no_grad():
        for categorical_fields, numerical_fields, labels in tqdm.tqdm(test_data_loader, smoothing=0, mininterval=1.0):
            categorical_fields, numerical_fields, labels = categorical_fields.to(device), numerical_fields.to(device), labels.to(device)
            y = model(categorical_fields, numerical_fields)
            for i in range(task_num):
                predicts_dict[i].extend(y[i].tolist())

    pred_ans = list(zip(predicts_dict[0], predicts_dict[1], predicts_dict[2]))
    # Convert list of tuples => list of lists
    pred_ans = [list(item) for item in pred_ans]
    return pred_ans


hi = get_predictions(model, df_test)
