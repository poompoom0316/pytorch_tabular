from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import random
import numpy as np
import pandas as pd
import os

from pytorch_tabular import TabularModel
from pytorch_tabular.models import CategoryEmbeddingModelConfig
from pytorch_tabular.config import DataConfig, OptimizerConfig, TrainerConfig, ExperimentConfig


def make_mixed_classification(n_samples, n_features, n_categories):
    X, y= make_classification(n_samples=n_samples, n_features=n_features, random_state=777, n_informative=5)
    cat_cols = random.choices(list(range(X.shape[-1])), k=n_categories)
    num_cols = [i for i in range(X.shape[-1]) if i not in cat_cols]
    for col in cat_cols:
        X[:,col] = pd.qcut(X[:,col], q=4).codes.astype(int)
    col_names = []
    num_col_names = []
    cat_col_names = []

    for i in range(X.shape[-1]):
        if i in cat_cols:
            col_names.append(f"cat_col_{i}")
            cat_col_names.append(f"cat_col_{i}")
        if i in num_cols:
            col_names.append(f"num_col_{i}")
            num_col_names.append(f"num_col_{i}")
    X = pd.DataFrame(X, columns=col_names)
    y = pd.Series(y, name="target")
    data = X.join(y)
    return data, cat_col_names, num_col_names


def print_metrics(y_true, y_pred, tag):
    if isinstance(y_true, pd.DataFrame) or isinstance(y_true, pd.Series):
        y_true = y_true.values
    if isinstance(y_pred, pd.DataFrame) or isinstance(y_pred, pd.Series):
        y_pred = y_pred.values
    if y_true.ndim>1:
        y_true = y_true.ravel()
    if y_pred.ndim>1:
        y_pred = y_pred.ravel()
    val_acc = accuracy_score(y_true, y_pred)
    val_f1 = f1_score(y_true, y_pred)
    print(f"{tag} Acc: {val_acc} | {tag} F1: {val_f1}")


def main():
    # Generate Synthetic Data
    data, cat_col_names, num_col_names = make_mixed_classification(n_samples=10000, n_features=20, n_categories=4)
    train, test = train_test_split(data, random_state=42)
    train, val = train_test_split(train, random_state=42)

    # ##########Define the Configs############
    data_config = DataConfig(
        target=["target"],
        continuous_cols=num_col_names,
        categorical_cols=cat_col_names
    )
    trainer_config = TrainerConfig(
        auto_lr_find=True,
        batch_size=1024,
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
    tabular_mode.fit(train=train, validation=val)
    # Evaluating the Model
    # #Loss and Metrics on New Data¶
    result = tabular_mode.evaluate(test)

    # #New Predictions as DataFrame
    pred_df = tabular_mode.predict(test)
    pred_df.head()

    print_metrics(test['target'], pred_df["prediction"], tag="Holdout")

    # saving model
    tabular_mode.save_model("Analysis/basic")
