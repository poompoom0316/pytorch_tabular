from sklearn.model_selection import train_test_split, StratifiedKFold
import pandas as pd
from pytorch_tabular import TabularModel
from pytorch_tabular.config import DataConfig, OptimizerConfig, TrainerConfig
from pytorch_tabular.models import TabNetModelConfig, TabNetModel, NodeConfig
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, StratifiedKFold
from pytorch_tabular.utils import get_class_weighted_cross_entropy
from Scripts.s03_test_real_data_analysis import data_load, print_metrics
import numpy as np



def main():
    # Generate Synthetic Data
    data, test_data, cat_col_names, num_col_names = data_load()
    bsize = 512*5

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
        gpus=1
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
        num_trees=512,  # Number of Trees in each layer
        depth=4,  # Depth of each Tree
        embed_categorical=True,
        # If True, will use a learned embedding, else it will use LeaveOneOutEncoding for categorical columns
        learning_rate=1e-3,
        additional_tree_output_dim=10
    )

    # Training the Model
    # tabular_mode.fit(train=train, validation=val)
    # # Evaluating the Model
    # # #Loss and Metrics on New Data¶
    # result = tabular_mode.evaluate(test)

    cv = StratifiedKFold(n_splits=10, shuffle=True)

    res_pred = []
    res_test = []

    train, val = train_test_split(data, random_state=42)

    tabular_mode = TabularModel(
        data_config=data_config,
        optimizer_config=optimizer_config,
        model_config=model_config,
        trainer_config=trainer_config
    )
    weighted_loss = get_class_weighted_cross_entropy(train["target"].values.ravel(), mu=0.1)

    # Training the Model
    tabular_mode.fit(train=train, validation=val, max_epochs=100, loss=weighted_loss)
    # tabular_mode.save_model(f"Analysis/basic_node_rep{i}")

    ns = 20000
    nrep = int(test_data.shape[0]/ns)
    nlist = []
    for i in range(nrep):
        pp = tabular_mode.predict(test_data.iloc[np.arange(ns * i, ns * (i + 1))])
        nlist.append(pp)

    # #New Predictions as DataFrame
    pred_tot = pd.concat(nlist).sort_index()

    # print_metrics(data['target'], pred_tot["prediction"], tag="Holdout")

    # pred_df = pd.concat([res_testi.loc[:, ["0_probability"]] for res_testi in res_test], axis=1).apply(np.mean, axis=1)
    # pred_df2 = pred_df.map(lambda x: 1 if x>0.5 else 0)

    sample_submisson = pd.read_csv("Data/sample_submission.csv")
    sample_submisson["target"] = pred_tot.prediction.values

    sample_submisson.to_csv("Analysis/submission_2_node.csv", index=False)

    print(confusion_matrix(data['target'], pred_tot["prediction"]))


def main_64():
    # Generate Synthetic Data
    global train
    data, test_data, cat_col_names, num_col_names = data_load()
    bsize = 2500*3*2*2

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
        gpus=1
    )
    optimizer_config = OptimizerConfig()

    model_config = TabNetModelConfig(
        task="classification",
        learning_rate=1e-3*bsize/1024,
        n_d=64,
        n_a=64,
        n_steps=5,
        gamma=1.3
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
        tabular_mode.save_model(f"Analysis/basic_tabnet_rep{i}")

        pred = tabular_mode.predict(test_data)
        res_test.append(pred)

    # #New Predictions as DataFrame
    pred_tot = pd.concat(res_pred).sort_index()

    print_metrics(data['target'], pred_tot["prediction"], tag="Holdout")

    pred_df = pd.concat([res_testi.loc[:, ["0_probability"]] for res_testi in res_test], axis=1).apply(np.mean, axis=1)
    pred_df2 = pred_df.map(lambda x: 1 if x>0.5 else 0)

    sample_submisson = pd.read_csv("Data/sample_submission.csv")
    sample_submisson["target"] = pred_df2.values

    sample_submisson.to_csv("Analysis/submission_2.csv", index=False)

    print(confusion_matrix(data['target'], pred_tot["prediction"]))



def apply_test_data():
    data, test_data, cat_col_names, num_col_names = data_load()
    bsize = 2500*3*2*2

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
        gpus=1
    )
    optimizer_config = OptimizerConfig()

    model_config = TabNetModelConfig(
        task="classification",
        learning_rate=1e-3*bsize/1024,
        n_d=24,
        n_a=24,
        n_steps=5,
        gamma=1.3
    )

    tabular_mode = TabularModel(
        data_config=data_config,
        optimizer_config=optimizer_config,
        model_config=model_config,
        trainer_config=trainer_config
    )

    for i in range(10):
        diri = f"Analysis/basic_tabnet_rep{i}"
        tabular_mode.load_from_checkpoint(dir=diri)


if __name__ == '__main__':
    main()
    # apply_test_data()

