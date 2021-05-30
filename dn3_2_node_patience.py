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

        # df_train[cat_col] = label_enc.fit_transform(df_train[cat_col])
        # df_train[cat_col] = df_train[cat_col].astype("category")
        # encoders[cat_col] = label_enc

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
        # df_test[cat_col] = df_test[cat_col].apply(lambda x: le_dict.get(x, default_val)).astype(int).astype(str)
        # df_test[cat_col] = df_test[cat_col].astype("category")

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

    bsize = 2500 * 2

    # ##########Define the Configs############
    data_config = DataConfig(
        target=["target"],
        continuous_cols=num_col_names,
        categorical_cols=cat_col_names,
        num_workers=4
    )
    trainer_config = TrainerConfig(
        auto_lr_find=True,
        batch_size=bsize,
        max_epochs=100,
        gpus=1,
        early_stopping_patience=5
    )
    optimizer_config = OptimizerConfig()

    # model_config = TabNetModelConfig(
    #     task="classification",
    #     learning_rate=1e-3*bsize/1024,
    #     n_d=16,
    #     n_a=16,
    #     n_steps=5,
    #     gamma=1.3
    # )

    model_config = NodeConfig(
        task="classification",
        num_layers=2,  # Number of Dense Layers
        num_trees=1024,  # Number of Trees in each layer
        depth=3,  # Depth of each Tree
        embed_categorical=True,
        # If True, will use a learned embedding, else it will use LeaveOneOutEncoding for categorical columns
        learning_rate=1e-3 * bsize / 1024,
        additional_tree_output_dim=6
    )

    # Training the Model
    # tabular_mode.fit(train=train, validation=val)
    # # Evaluating the Model
    # # #Loss and Metrics on New Data¶
    # result = tabular_mode.evaluate(test)

    cv = StratifiedKFold(n_splits=10, shuffle=True)

    res_pred = []
    res_test = []
    for i, (train_idx, test_idx) in enumerate(cv.split(X=data, y=data.target.values)):
        train, test = data.iloc[train_idx], data.iloc[test_idx]
        train, val = train_test_split(train, random_state=42)

        tabular_mode = TabularModel(
            data_config=data_config,
            optimizer_config=optimizer_config,
            model_config=model_config,
            trainer_config=trainer_config
        )
        weighted_loss = get_class_weighted_cross_entropy(train["target"].values.ravel(), mu=0.1)

        # Training the Model
        tabular_mode.fit(train=train, validation=val, max_epochs=100, loss=weighted_loss)
        pred_df = tabular_mode.predict(test).loc[:, ["prediction"]]
        res_pred.append(pred_df)

        print(f"Fold {i} AUC score: {roc_auc_score(test.target.values, pred_df.prediction.values)}")
        print_metrics(test.target, pred_df.prediction, tag="Holdout")
        # tabular_mode.save_model(f"Analysis/basic_tabnet_rep{i}")

        ns = 20000
        nrep = int(test_data.shape[0] / ns)
        nlist = []
        for i in range(nrep):
            pp = tabular_mode.predict(test_data.iloc[np.arange(ns * i, ns * (i + 1))])
            nlist.append(pp)

        pred = pd.concat(nlist)
        res_test.append(pred)

    pred_df = pd.concat([res_testi.loc[:, ["0_probability"]] for res_testi in res_test], axis=1).apply(np.mean, axis=1)
    pred_df2 = pred_df.map(lambda x: 0 if x > 0.5 else 1)

    sample_submisson = pd.read_csv("Data/sample_submission.csv")
    sample_submisson["target"] = pred_df2.values

    # ns = 20000
    # nrep = int(test_data.shape[0] / ns)
    # nlist = []
    # for i in range(nrep):
    #     pp = tabular_mode.predict(test_data.iloc[np.arange(ns * i, ns * (i + 1))])
    #     nlist.append(pp)

    # #New Predictions as DataFrame
    pred_tot = pd.concat(res_pred).sort_index()

    print_metrics(data['target'], pred_tot["prediction"], tag="Holdout")

    # pred_df = pd.concat([res_testi.loc[:, ["0_probability"]] for res_testi in res_test], axis=1).apply(np.mean, axis=1)
    # pred_df2 = pred_df.map(lambda x: 1 if x>0.5 else 0)

    # sample_submisson = pd.read_csv("Data/sample_submission.csv")
    # sample_submisson["target"] = pred_tot.prediction.values

    sample_submisson.to_csv("Analysis/submission_6_node.csv", index=False)

    print(confusion_matrix(data['target'], pred_tot["prediction"]))


if __name__ == '__main__':
    main()
