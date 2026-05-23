import argparse
from pathlib import Path

import joblib
import pandas as pd

from river import linear_model, metrics, optim, preprocessing


TARGETS = ["temperature", "humidity", "pressure", "discomfort"]
MODEL_PATH = "sensor_models.joblib"

LAGS = {
    "30min": 6,
    "1h": 12,
    "3h": 36,
    "1d": 288,
    "2d": 576,
    "7d": 2016,
}


def load_csvs(data_dir: str) -> pd.DataFrame:
    csv_files = sorted(Path(data_dir).glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(f"CSVが見つかりません: {data_dir}")

    dfs = []

    for csv_file in csv_files:
        print(f"loading: {csv_file}")

        df = pd.read_csv(csv_file)

        df["date"] = pd.to_datetime(
            df["date"],
            format="%Y/%m/%d/%H:%M:%S",
            errors="coerce",
        )

        for col in TARGETS:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        dfs.append(df)

    df_all = pd.concat(dfs, ignore_index=True)
    df_all = df_all.dropna(subset=["date", *TARGETS])
    df_all = df_all.sort_values("date").reset_index(drop=True)

    return df_all


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["hour"] = df["date"].dt.hour
    df["minute"] = df["date"].dt.minute
    df["dayofyear"] = df["date"].dt.dayofyear
    df["weekday"] = df["date"].dt.weekday

    for col in TARGETS:
        for name, lag in LAGS.items():
            df[f"{col}_lag_{name}"] = df[col].shift(lag)

    df = df.dropna().reset_index(drop=True)

    if df.empty:
        raise ValueError("特徴量作成後のデータが空です。最低でも7日分以上のCSVが必要です。")

    return df


def make_model():
    return preprocessing.StandardScaler() | linear_model.LinearRegression(
        optimizer=optim.SGD(0.01)
    )


def train(data_dir: str):
    df = add_features(load_csvs(data_dir))

    feature_cols = [
        col for col in df.columns
        if col not in ["date", *TARGETS]
    ]

    models = {target: make_model() for target in TARGETS}
    scores = {target: metrics.MAE() for target in TARGETS}

    for _, row in df.iterrows():
        x = row[feature_cols].to_dict()

        for target in TARGETS:
            y = row[target]
            y_pred = models[target].predict_one(x) or 0.0

            scores[target].update(y, y_pred)
            models[target].learn_one(x, y)

    joblib.dump(
        {
            "models": models,
            "feature_cols": feature_cols,
            "last_rows": df.tail(max(LAGS.values())),
        },
        MODEL_PATH,
    )

    print("学習完了")
    for target in TARGETS:
        print(f"{target}: MAE = {scores[target].get():.4f}")


def predict_next():
    data = joblib.load(MODEL_PATH)

    models = data["models"]
    feature_cols = data["feature_cols"]
    last_rows = data["last_rows"].copy()

    if len(last_rows) < max(LAGS.values()):
        raise ValueError("予測に必要な過去データが不足しています。")

    latest = last_rows.iloc[-1]

    x = {
        "hour": latest["date"].hour,
        "minute": latest["date"].minute,
        "dayofyear": latest["date"].dayofyear,
        "weekday": latest["date"].weekday(),
    }

    for col in TARGETS:
        for name, lag in LAGS.items():
            x[f"{col}_lag_{name}"] = last_rows.iloc[-lag][col]

    x = {col: x[col] for col in feature_cols}

    print("次時点の予測:")
    for target in TARGETS:
        pred = models[target].predict_one(x)
        print(f"{target}: {pred:.3f}")


def update(data_dir: str):
    data = joblib.load(MODEL_PATH)

    models = data["models"]
    feature_cols = data["feature_cols"]
    old_rows = data["last_rows"]

    new_rows = load_csvs(data_dir)

    df = pd.concat([old_rows, new_rows], ignore_index=True)
    df = df.sort_values("date").reset_index(drop=True)
    df = add_features(df)

    for _, row in df.iterrows():
        x = row[feature_cols].to_dict()

        for target in TARGETS:
            models[target].learn_one(x, row[target])

    data["models"] = models
    data["last_rows"] = df.tail(max(LAGS.values()))

    joblib.dump(data, MODEL_PATH)

    print("追加学習完了")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["train", "predict", "update"])
    parser.add_argument("--data-dir", default="data_train")

    args = parser.parse_args()

    if args.mode == "train":
        train(args.data_dir)
    elif args.mode == "predict":
        predict_next()
    elif args.mode == "update":
        update(args.data_dir)


if __name__ == "__main__":
    main()