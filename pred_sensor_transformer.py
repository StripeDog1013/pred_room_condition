import argparse
from pathlib import Path
from datetime import timedelta
import math

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler


TARGETS = ["temperature", "humidity", "pressure", "discomfort"]
TIME_FEATURES = ["hour_sin", "hour_cos", "day_sin", "day_cos"]

SEQ_LEN = 288
BATCH_SIZE = 64
EPOCHS = 30
LR = 0.001

MODEL_PATH = "sensor_transformer.pt"
SCALER_PATH = "sensor_transformer_scaler.joblib"


def load_csvs(data_dir: str) -> pd.DataFrame:
    csv_files = sorted(Path(data_dir).glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(f"CSVが見つかりません: {data_dir}")

    dfs = []

    for file in csv_files:
        print(f"loading: {file}")
        df = pd.read_csv(file)

        df["date"] = pd.to_datetime(
            df["date"],
            format="%Y/%m/%d/%H:%M:%S",
            errors="coerce",
        )

        for col in TARGETS:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)
    df = df.dropna(subset=["date", *TARGETS])
    df = df.sort_values("date").reset_index(drop=True)

    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    hour = df["date"].dt.hour + df["date"].dt.minute / 60
    day = df["date"].dt.dayofyear

    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)

    df["day_sin"] = np.sin(2 * np.pi * day / 365)
    df["day_cos"] = np.cos(2 * np.pi * day / 365)

    return df


def make_time_features(timestamp: pd.Timestamp):
    hour = timestamp.hour + timestamp.minute / 60
    day = timestamp.dayofyear

    return np.array([
        np.sin(2 * np.pi * hour / 24),
        np.cos(2 * np.pi * hour / 24),
        np.sin(2 * np.pi * day / 365),
        np.cos(2 * np.pi * day / 365),
    ])


def make_xy(df: pd.DataFrame, scaler: StandardScaler | None = None):
    df = add_time_features(df)

    sensor_values = df[TARGETS].values
    time_values = df[TIME_FEATURES].values

    if scaler is None:
        scaler = StandardScaler()
        sensor_scaled = scaler.fit_transform(sensor_values)
    else:
        sensor_scaled = scaler.transform(sensor_values)

    x_values = np.concatenate([sensor_scaled, time_values], axis=1)
    y_values = sensor_scaled

    return x_values, y_values, scaler


class SensorDataset(Dataset):
    def __init__(self, x_values, y_values):
        self.x_values = x_values
        self.y_values = y_values

    def __len__(self):
        return len(self.x_values) - SEQ_LEN

    def __getitem__(self, idx):
        x = self.x_values[idx:idx + SEQ_LEN]
        y = self.y_values[idx + SEQ_LEN]

        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
        )


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)

        position = torch.arange(
            0,
            max_len,
            dtype=torch.float32,
        ).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class SensorTransformer(nn.Module):
    def __init__(
        self,
        input_size=8,
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        dropout=0.1,
        output_size=4,
    ):
        super().__init__()

        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.fc = nn.Linear(d_model, output_size)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_encoder(x)

        out = self.transformer_encoder(x)

        last = out[:, -1, :]

        return self.fc(last)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def load_model():
    device = get_device()

    model = SensorTransformer().to(device)

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=device,
        )
    )

    model.eval()

    return model, device


def build_input_sequence(sensor_values, last_timestamp, scaler):
    sensor_scaled = scaler.transform(sensor_values)

    timestamps = [
        last_timestamp - timedelta(minutes=5 * (SEQ_LEN - 1 - i))
        for i in range(SEQ_LEN)
    ]

    time_values = np.array([
        make_time_features(pd.Timestamp(ts))
        for ts in timestamps
    ])

    return np.concatenate([sensor_scaled, time_values], axis=1)


def train(data_dir: str):
    df = load_csvs(data_dir)

    if len(df) <= SEQ_LEN:
        raise ValueError("学習データ不足")

    x_values, y_values, scaler = make_xy(df)

    dataset = SensorDataset(x_values, y_values)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    device = get_device()
    print(f"device: {device}")

    model = SensorTransformer().to(device)

    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR,
    )

    for epoch in range(1, EPOCHS + 1):
        model.train()

        total_loss = 0.0

        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            pred = model(x)

            loss = criterion(pred, y)

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        print(f"epoch {epoch:03d} | loss: {total_loss / len(loader):.6f}")

    torch.save(model.state_dict(), MODEL_PATH)

    joblib.dump(
        {
            "scaler": scaler,
            "last_sensor_values": df[TARGETS].values[-SEQ_LEN:],
            "last_timestamp": df.iloc[-1]["date"],
        },
        SCALER_PATH,
    )

    print("学習完了")


def predict(data_dir: str | None):
    model, device = load_model()

    saved = joblib.load(SCALER_PATH)
    scaler = saved["scaler"]

    if data_dir:
        df = load_csvs(data_dir)
        sensor_values = df[TARGETS].values[-SEQ_LEN:]
        last_timestamp = df.iloc[-1]["date"]
    else:
        sensor_values = saved["last_sensor_values"]
        last_timestamp = saved["last_timestamp"]

    if len(sensor_values) < SEQ_LEN:
        raise ValueError("予測に必要なデータ不足")

    x_values = build_input_sequence(
        sensor_values,
        last_timestamp,
        scaler,
    )

    x = torch.tensor(
        x_values,
        dtype=torch.float32,
    ).unsqueeze(0).to(device)

    with torch.no_grad():
        pred_scaled = model(x).cpu().tolist()

    pred = scaler.inverse_transform(pred_scaled)[0]

    print("次時点の予測:")

    for name, value in zip(TARGETS, pred):
        print(f"{name}: {value:.3f}")


def predict_future(steps=288):
    model, device = load_model()

    saved = joblib.load(SCALER_PATH)
    scaler = saved["scaler"]

    sensor_history = saved["last_sensor_values"].tolist()
    current_time = saved["last_timestamp"]

    predictions = []

    for _ in range(steps):
        sensor_values = np.array(sensor_history[-SEQ_LEN:])

        x_values = build_input_sequence(
            sensor_values,
            current_time,
            scaler,
        )

        x = torch.tensor(
            x_values,
            dtype=torch.float32,
        ).unsqueeze(0).to(device)

        with torch.no_grad():
            pred_scaled = model(x).cpu().tolist()

        pred = scaler.inverse_transform(pred_scaled)[0]

        current_time = current_time + timedelta(minutes=5)

        sensor_history.append(pred.tolist())
        predictions.append((current_time, pred))

    print(f"\n未来予測 ({steps} step)\n")

    for timestamp, pred in predictions:
        print(
            f"{timestamp.strftime('%Y-%m-%d %H:%M:%S')} "
            f"| temp={pred[0]:.2f} "
            f"| hum={pred[1]:.2f} "
            f"| press={pred[2]:.2f} "
            f"| discomfort={pred[3]:.2f}"
        )


def update(data_dir: str):
    model, device = load_model()

    saved = joblib.load(SCALER_PATH)
    scaler = saved["scaler"]

    df = load_csvs(data_dir)

    if len(df) <= SEQ_LEN:
        raise ValueError("追加学習データ不足")

    x_values, y_values, _ = make_xy(df, scaler)

    dataset = SensorDataset(x_values, y_values)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR,
    )

    update_epochs = 5

    for epoch in range(1, update_epochs + 1):
        model.train()

        total_loss = 0.0

        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            pred = model(x)

            loss = criterion(pred, y)

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        print(f"update epoch {epoch:03d} | loss: {total_loss / len(loader):.6f}")

    torch.save(model.state_dict(), MODEL_PATH)

    joblib.dump(
        {
            "scaler": scaler,
            "last_sensor_values": df[TARGETS].values[-SEQ_LEN:],
            "last_timestamp": df.iloc[-1]["date"],
        },
        SCALER_PATH,
    )

    print("追加学習完了")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "mode",
        choices=["train", "predict", "predict_future", "update"],
    )

    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--steps", type=int, default=288)

    args = parser.parse_args()

    if args.mode == "train":
        train(args.data_dir or "data_train")

    elif args.mode == "predict":
        predict(args.data_dir)

    elif args.mode == "predict_future":
        predict_future(args.steps)

    elif args.mode == "update":
        update(args.data_dir or "data_update")


if __name__ == "__main__":
    main()