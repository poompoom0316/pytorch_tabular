from sklearn.model_selection import train_test_split, StratifiedKFold
import pandas as pd
from pytorch_tabular import TabularModel
from pytorch_tabular.config import DataConfig, OptimizerConfig, TrainerConfig
from pytorch_tabular.models import TabNetModelConfig, TabNetModel, NodeConfig
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, StratifiedKFold
from pytorch_tabular.utils import get_class_weighted_cross_entropy

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score
import random
import numpy as np
import os

from pytorch_tabular.utils import get_class_weighted_cross_entropy
from pytorch_tabular import TabularModel
from pytorch_tabular.models import CategoryEmbeddingModelConfig
from pytorch_tabular.config import DataConfig, OptimizerConfig, TrainerConfig, ExperimentConfig
from pytorch_tabnet.tab_model import TabNetClassifier
import torch
from pytorch_tabnet.pretraining import TabNetPretrainer
from sklearn.preprocessing import LabelEncoder


def data_load():
    train_path = "Data/train.csv"
    test_path = "Data/test.csv"

    df_train = pd.read_csv(train_path, index_col=0)
    df_test = pd.read_csv(test_path, index_col=0)
    # data["target"] = data.target.astype(int)

    CAT_COLS = list(df_train.filter(like="cat").columns)
    NUM_COLS = list(df_train.filter(like="cont").columns)

    LOW_FREQ_THRESH = 50

    encoders = {}
    # Categorical features need to be LabelEncoded
    for cat_col in CAT_COLS:
        label_enc = LabelEncoder()

        # Group low frequencies into one value
        value_counts = df_train[cat_col].value_counts()
        is_low_frequency = value_counts < LOW_FREQ_THRESH
        low_freq_values = value_counts.index[is_low_frequency]
        if len(low_freq_values) > 0:
            df_train.loc[df_train[cat_col].isin(low_freq_values), cat_col] = "low_frequency"
            # update test set as well
            df_test.loc[df_test[cat_col].isin(low_freq_values), cat_col] = "low_frequency"

        df_train[cat_col] = label_enc.fit_transform(df_train[cat_col])
        encoders[cat_col] = label_enc

    # Encode test set
    for cat_col in CAT_COLS:
        label_enc = encoders[cat_col]
        le_dict = dict(zip(label_enc.classes_, label_enc.transform(label_enc.classes_)))
        # Replace unknown values by the most common value
        # Changing this to another value might make more sense
        if le_dict.get("low_frequency") is not None:
            default_val = le_dict["low_frequency"]
        else:
            default_val = df_train[cat_col].mode().values[0]
        df_test[cat_col] = df_test[cat_col].apply(lambda x: le_dict.get(x, default_val))

    # Clip numerical features in test set to match training set
    for num_col in NUM_COLS:
        df_test[num_col] = np.clip(df_test[num_col], df_train[num_col].min(), df_train[num_col].max())

        # Taken from https://www.kaggle.com/siavrez/kerasembeddings
        df_train[f'q_{num_col}'], bins_ = pd.qcut(df_train[num_col], 25, retbins=True, labels=[i for i in range(25)])
        df_test[f'q_{num_col}'] = pd.cut(df_test[num_col], bins=bins_, labels=False, include_lowest=True)
        CAT_COLS.append(f'q_{num_col}')

    return df_train, df_test, CAT_COLS, NUM_COLS


def print_metrics(y_true, y_pred, tag):
    if isinstance(y_true, pd.DataFrame) or isinstance(y_true, pd.Series):
        y_true = y_true.values
    if isinstance(y_pred, pd.DataFrame) or isinstance(y_pred, pd.Series):
        y_pred = y_pred.values
    if y_true.ndim > 1:
        y_true = y_true.ravel()
    if y_pred.ndim > 1:
        y_pred = y_pred.ravel()
    val_acc = accuracy_score(y_true, y_pred)
    val_f1 = f1_score(y_true, y_pred)
    print(f"{tag} Acc: {val_acc} | {tag} F1: {val_f1}")


def main():
    # Generate Synthetic Data
    data, test_data, cat_col_names, num_col_names = data_load()
    cat_dims = data[cat_col_names].nunique().to_list()
    cat_idxs = [(cat_col_names+num_col_names).index(cat_col) for cat_col in cat_col_names]
    cat_emb_dims = np.ceil(np.log(cat_dims)).astype(np.int).tolist()
    cat_emb_dims = np.ceil(np.clip((np.array(cat_dims)) / 2, a_min=1, a_max=50)).astype(np.int).tolist()
    FEATURES = cat_col_names+num_col_names
    df_sub = pd.read_csv('Data/sample_submission.csv')

    bsize = 2500 * 2

    # ##########Define the Configs############
    N_D = 16
    N_A = 16
    N_INDEP = 2
    N_SHARED = 2
    N_STEPS = 1  # 2
    MASK_TYPE = "sparsemax"
    GAMMA = 1.5
    BS = 512
    MAX_EPOCH = 21  # 20
    PRETRAIN = True

    X = data[FEATURES].values
    y = data["target"].values

    X_test = test_data[FEATURES].values

    if PRETRAIN:
        pretrain_params = dict(n_d=N_D, n_a=N_A, n_steps=N_STEPS,  # 0.2,
                               n_independent=N_INDEP, n_shared=N_SHARED,
                               cat_idxs=cat_idxs,
                               cat_dims=cat_dims,
                               cat_emb_dim=cat_emb_dims,
                               gamma=GAMMA,
                               lambda_sparse=0., optimizer_fn=torch.optim.Adam,
                               optimizer_params=dict(lr=2e-2),
                               mask_type=MASK_TYPE,
                               scheduler_params=dict(mode="min",
                                                     patience=3,
                                                     min_lr=1e-5,
                                                     factor=0.5, ),
                               scheduler_fn=torch.optim.lr_scheduler.ReduceLROnPlateau,
                               verbose=1,
                               )

        pretrainer = TabNetPretrainer(**pretrain_params)

        pretrainer.fit(X_train=X_test,
                       eval_set=[X],
                       max_epochs=MAX_EPOCH,
                       patience=25, batch_size=BS, virtual_batch_size=BS,  # 128,
                       num_workers=0, drop_last=True,
                       pretraining_ratio=0.5  # The bigger your pretraining_ratio the harder it is to reconstruct
                       )
    # Training the Model
    # tabular_mode.fit(train=train, validation=val)
    # # Evaluating the Model
    # # #Loss and Metrics on New Data¶
    # result = tabular_mode.evaluate(test)

    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=777)

    BS = 2048
    MAX_EPOCH = 20
    # skf = StratifiedKFold(n_splits=5, random_state=2021, shuffle=True)

    data['oof_preds'] = np.nan

    for fold_nb, (train_index, valid_index) in enumerate(cv.split(X, y)):
        X_train, X_valid = X[train_index], X[valid_index]
        y_train, y_valid = y[train_index], y[valid_index]

        tabnet_params = dict(n_d=N_D,
                             n_a=N_A,
                             n_steps=N_STEPS, gamma=GAMMA,
                             n_independent=N_INDEP, n_shared=N_SHARED,
                             lambda_sparse=1e-5,
                             seed=0,
                             clip_value=2,
                             cat_idxs=cat_idxs,
                             cat_dims=cat_dims,
                             cat_emb_dim=cat_emb_dims,
                             mask_type=MASK_TYPE,
                             device_name='auto',
                             optimizer_fn=torch.optim.Adam,
                             optimizer_params=dict(lr=5e-2, weight_decay=1e-5),
                             scheduler_params=dict(max_lr=5e-2,
                                                   steps_per_epoch=int(X_train.shape[0] / BS),
                                                   epochs=MAX_EPOCH,
                                                   # final_div_factor=100,
                                                   is_batch_level=True),
                             scheduler_fn=torch.optim.lr_scheduler.OneCycleLR,
                             #                               scheduler_params=dict(mode='max',
                             #                                                     factor=0.5,
                             #                                                     patience=5,
                             #                                                     is_batch_level=False,),
                             #                               scheduler_fn=torch.optim.lr_scheduler.ReduceLROnPlateau,
                             verbose=1)
        # Defining TabNet model
        model = TabNetClassifier(**tabnet_params)

        model.fit(X_train=X_train, y_train=y_train,
                  from_unsupervised=pretrainer if PRETRAIN else None,
                  eval_set=[(X_train, y_train), (X_valid, y_valid)],
                  eval_name=["train", "valid"],
                  eval_metric=["auc"],
                  batch_size=BS,
                  virtual_batch_size=256,
                  max_epochs=MAX_EPOCH,
                  drop_last=True,
                  pin_memory=True,
                  patience=10,
                  )

        val_preds = model.predict_proba(X_valid)[:, -1]
        print('auc:', roc_auc_score(y_true=y_valid, y_score=val_preds))

        data['oof_preds'].iloc[valid_index] = val_preds

        test_preds = model.predict_proba(X_test)[:, -1]
        df_sub[f"fold_{fold_nb+1}"] = test_preds


    df_sub["target"] = df_sub.filter(like="fold_").mean(axis=1).values

    df_sub.to_csv("Analysis/submission_5_tabnet.csv", index=False)

    df_sub = pd.read_csv("Analysis/submission_5_tabnet.csv")

    # df_sub.target = df_sub.target.map(lambda x: 0 if x<=0.5 else 1)
    df_sub.loc[:, ["id", "target"]].to_csv("Analysis/submission_5_2_tabnet.csv", index=False)



if __name__ == '__main__':
    main()
