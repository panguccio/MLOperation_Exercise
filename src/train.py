import neptune

run = neptune.init_run(
    project="emmadariol/MLops",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJiZmMwMmNjNC03NmY4LTQ3ZDEtOTMxNS1kYWU0M2U4ZWU4MTkifQ==",
)  # your credentials

params = {"learning_rate": 0.001, "optimizer": "Adam"}
run["parameters"] = params

for epoch in range(10):
    run["train/loss"].append(0.9 ** epoch)

run["eval/f1_score"] = 0.66

run.stop()

import yaml
import neptune
import joblib
import os

# Updated River Imports for Online/Incremental Learning
from river.forest import ARFClassifier
from river import compose
from river import preprocessing
from river import metrics

# Local Imports
from src.data_loader import load_data
from src.preprocess import preprocess_data, df_to_river_format
from src.utils import save_model, load_latest_model

def load_config():
    with open("configs/config.yaml", "r") as f:
        return yaml.safe_load(f)

def train_or_update_model(new_data_df=None):

    config = load_config()

    try:
        if new_data_df is None:
            raise FileNotFoundError("Force Initial Training")

        model, _, prev_version = load_latest_model()
        print(f"\n🔄 LOADED Model v{prev_version}. Adapting to new data stream...")

        X_train, X_test, y_train, y_test = preprocess_data(new_data_df, test_size=0.2)
        is_update = True

    except (FileNotFoundError, ValueError):
        print("\n Starting Initial Training (From Scratch)...")
        df, _ = load_data()
        X_train, X_test, y_train, y_test = preprocess_data(df, test_size=config["data"]["test_size"])

        # Updated River Random Forest
        model = compose.Pipeline(
            preprocessing.StandardScaler(),
            ARFClassifier(
                n_models=config["model"]["n_models"],
                seed=config["model"]["seed"]
            )
        )

        is_update = False


    stream_X_train, stream_y_train = df_to_river_format(X_train, y_train)
    stream_X_test, stream_y_test = df_to_river_format(X_test, y_test)

    # Neptune Integration
    try:
        run = neptune.init_run(
            project=config["neptune"]["project"],
            api_token=config["neptune"]["api_token"],
            mode=config["neptune"].get("mode", "async")
        )
        run["config"] = config
        run["training_mode"] = "incremental" if is_update else "initial"
    except Exception as e:
        print(f"Neptune warning (check credentials): {e}")
        run = None

    print(f"Learning from {len(stream_X_train)} samples...")
    metric = metrics.Accuracy()

    # Online Training
    for x, y in zip(stream_X_train, stream_y_train):
        y_pred = model.predict_one(x)
        metric.update(y, y_pred)
        model.learn_one(x, y)

    print(f"   -> Stream Accuracy during training: {metric.get():.4f}")

    # Evaluation
    test_metric = metrics.Accuracy()
    f1_metric = metrics.F1()

    for x, y in zip(stream_X_test, stream_y_test):
        y_pred = model.predict_one(x)
        test_metric.update(y, y_pred)
        f1_metric.update(y, y_pred)

    final_acc = test_metric.get()
    final_f1 = f1_metric.get()

    print(f"Test Results -> Accuracy: {final_acc:.4f}, F1: {final_f1:.4f}")

    if run:
        run["metrics/accuracy"] = final_acc
        run["metrics/f1_score"] = final_f1

    save_model(model, None, {"accuracy": final_acc})

    if run:
        run.stop()

if __name__ == "__main__":
    train_or_update_model()
