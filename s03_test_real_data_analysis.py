from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import random
import numpy as np
import pandas as pd
import os

from pytorch_tabular.utils import get_class_weighted_cross_entropy
from pytorch_tabular import TabularModel
from pytorch_tabular.models import CategoryEmbeddingModelConfig
from pytorch_tabular.config import DataConfig, OptimizerConfig, TrainerConfig, ExperimentConfig


def data_load():
    path_data = "Data/train.csv"
    path_data2 = "Data/test.csv"
    df1 = pd.read_csv(path_data, index_col=0)
    df2 = pd.read_csv(path_data2, index_col=0)

    df1_train_cyclic = df1.copy()
    columns = ['day', 'month']
    for col in columns:
        df1_train_cyclic[col + '_sin'] = np.sin((2 * np.pi * df1_train_cyclic[col]) / max(df1_train_cyclic[col]))
        df1_train_cyclic[col + '_cos'] = np.cos((2 * np.pi * df1_train_cyclic[col]) / max(df1_train_cyclic[col]))
    df1_train_cyclic = df1_train_cyclic.drop(columns, axis=1)

    df2_test_cyclic = df2.copy()
    columns = ['day', 'month']
    for col in columns:
        df2_test_cyclic[col + '_sin'] = np.sin((2 * np.pi * df2_test_cyclic[col]) / max(df2_test_cyclic[col]))
        df2_test_cyclic[col + '_cos'] = np.cos((2 * np.pi * df2_test_cyclic[col]) / max(df2_test_cyclic[col]))
    df2_test_cyclic = df2_test_cyclic.drop(columns, axis=1)

    nom_one_hot = ["nom_{}".format(i) for i in range(5)]
    df1_train_cyclic_onehot = pd.get_dummies(df1_train_cyclic.loc[:, nom_one_hot])
    df1_train_cyclic = pd.concat([df1_train_cyclic, df1_train_cyclic_onehot], axis=1)

    df2_test_cyclic_onehot = pd.get_dummies(df2_test_cyclic.loc[:, nom_one_hot])
    df2_test_cyclic = pd.concat([df2_test_cyclic, df2_test_cyclic_onehot], axis=1)

    nom_cv_names = list(df1_train_cyclic_onehot.columns)

    pos_ave = ["nom_{}".format(i) for i in range(5, 10)]
    pos_ave_list = []
    for posi in pos_ave:
        ave_values = df1_train_cyclic.groupby([posi])['target'].mean()
        pos_ave_list.append(ave_values)
        ave_values_mat = df1_train_cyclic.groupby([posi])['target'].mean().reset_index()
        ave_values_mat.columns = [posi, "{}_v".format(posi)]

        df1_train_cyclic = pd.merge(df1_train_cyclic, ave_values_mat, on=posi)
        df2_test_cyclic = pd.merge(df2_test_cyclic, ave_values_mat, on=posi, how="left")

    df1_train_cyclic.loc[:, 'ord_5a'] = df1_train_cyclic.loc[:, 'ord_5'].map(lambda x: x[0])
    df1_train_cyclic.loc[:, 'ord_5b'] = df1_train_cyclic.loc[:, 'ord_5'].map(lambda x: x[1])
    df2_test_cyclic.loc[:, 'ord_5a'] = df2_test_cyclic.loc[:, 'ord_5'].map(lambda x: x[0])
    df2_test_cyclic.loc[:, 'ord_5b'] = df2_test_cyclic.loc[:, 'ord_5'].map(lambda x: x[1])

    for posi in ["bin_3", "bin_4"] + ["ord_{}".format(i) for i in range(3, 6)] + ['ord_5' + i for i in ["a", "b"]]:
        df1_train_cyclic.loc[:, "{}_ov".format(posi)] = df1_train_cyclic.loc[:, posi].astype("category").cat.codes
        df2_test_cyclic.loc[:, "{}_ov".format(posi)] = df2_test_cyclic.loc[:, posi].astype("category").cat.codes

    cat_ord1 = ["Novice", "Grandmaster", "Contributor", "Master", "Expert"]
    cat_ord2 = ["Freezing", "Lava Hot", "Boiling Hot", "Cold", "Hot", "Warm"]
    val_ord1 = [0, 4, 2, 3, 1]
    val_ord2 = [0, 5, 4, 1, 3, 2]
    cat_ord1_df = pd.DataFrame({'ord_1': cat_ord1, 'ord_1_ov': val_ord1})
    cat_ord2_df = pd.DataFrame({'ord_2': cat_ord2, 'ord_2_ov': val_ord2})

    for posi, dfi in zip(["ord_1", "ord_2"], [cat_ord1_df, cat_ord2_df]):
        df1_train_cyclic = pd.merge(df1_train_cyclic, dfi, on=posi)
        df2_test_cyclic = pd.merge(df2_test_cyclic, dfi, on=posi, how="left")

    dayv = ["day_sin", "day_cos", "month_sin", "month_cos"]
    use_values = ["bin_{}".format(i) for i in range(3)
                  ] + ["bin_{}_ov".format(i) for i in range(3, 5)] + ["ord_0"] + [
                     "ord_{}_ov".format(i) for i in range(1, 5)] + ['ord_5' + i for i in
                                                                    ["a_ov", "b_ov"]] + nom_cv_names + [
                     "nom_{}_v".format(i) for i in range(5, 9)] + dayv

    continuous_cols_pre = pd.Series(use_values)
    continuous_cols = list(
        continuous_cols_pre.loc[continuous_cols_pre.str.contains("ord|nom_[5-8]_v|day|month")].values)
    categorical_cols = list(set(use_values) - set(continuous_cols))

    data = df1_train_cyclic.loc[:, use_values+['target']]
    test_data = df2_test_cyclic.loc[:, use_values]

    return data, test_data, categorical_cols, continuous_cols


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
    data, cat_col_names, num_col_names = data_load()
    bsize = 1024

    # ##########Define the Configs############
    data_config = DataConfig(
        target=["target"],
        continuous_cols=num_col_names,
        categorical_cols=cat_col_names
    )
    trainer_config = TrainerConfig(
        auto_lr_find=True,
        batch_size=bsize,
        max_epochs=100,
        gpus=1
    )
    optimizer_config = OptimizerConfig()

    model_config = CategoryEmbeddingModelConfig(
        task="classification",
        layers="1024-512-512",
        activation="LeakyReLU",
        learning_rate=1e-3
    )

    tabular_mode = TabularModel(
        data_config=data_config,
        model_config=model_config,
        optimizer_config=optimizer_config,
        trainer_config=trainer_config
    )

    # Training the Model
    # tabular_mode.fit(train=train, validation=val)
    # # Evaluating the Model
    # # #Loss and Metrics on New Data¶
    # result = tabular_mode.evaluate(test)

    cv = StratifiedKFold(n_splits=10, shuffle=True)

    res_pred = []
    for train_idx, test_idx in cv.split(X=data, y=data.target.values):
        train, test = data.iloc[train_idx], data.iloc[test_idx]
        train, val = train_test_split(train, random_state=42)

        tabular_mode = TabularModel(
            data_config=data_config,
            model_config=model_config,
            optimizer_config=optimizer_config,
            trainer_config=trainer_config
        )

        weighted_loss = get_class_weighted_cross_entropy(train["target"].values.ravel(), mu=0.1)

        # Training the Model
        tabular_mode.fit(train=train, validation=val, max_epochs=100, loss=weighted_loss)
        pred_df = tabular_mode.predict(test).loc[:, ["prediction"]]
        res_pred.append(pred_df)

    # #New Predictions as DataFrame
    pred_tot = pd.concat(res_pred).sort_index()

    print_metrics(data['target'], pred_tot["prediction"], tag="Holdout")

    confusion_matrix(data['target'], pred_tot["prediction"])

    # saving model
    tabular_mode.save_model("Analysis/basic")
