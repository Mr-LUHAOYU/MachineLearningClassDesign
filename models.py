import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
import datetime
import torch
import torch.nn as nn
import torch.optim as optim


class Model(object):
    def __init__(
            self, model_name, threshold: float = 0.7, n_estimators: int = 100,
            max_depth: int = 3, learning_rate: float = 0.1, n_jobs: int = -1,
            input_size: int = 24, random_state: int = 42, epochs: int = 10,
            batch_size: int = 32, hidden_size: int = 64, output_size: int = 1,
    ):
        self.threshold = threshold
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_jobs = n_jobs
        self.input_size = input_size
        self.random_state = random_state
        self.epochs = epochs
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        if model_name == 'XGBoost':
            self.model = xgb.XGBRegressor(
                n_estimators=n_estimators, max_depth=max_depth,
                n_jobs=n_jobs, random_state=random_state, learning_rate=learning_rate
            )
        elif model_name == 'MLP':
            self.model = MLPModel(
                input_size=input_size, hidden_size=hidden_size, output_size=output_size,
                learning_rate=learning_rate, epochs=epochs, batch_size=batch_size
            )
        else:
            raise ValueError('Invalid model name')

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.model.fit(X, y)

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        y = self.model.predict(X)
        y = pd.DataFrame(y, columns=['glucose'])
        y['time'] = X['time']
        return y


class MLPRegressor(nn.Module):
    def __init__(self, input_size, hidden_size=64, output_size=1):
        super(MLPRegressor, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc4 = nn.Linear(hidden_size // 2, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class MLPModel:
    def __init__(
            self, input_size, hidden_size=64, output_size=1,
            learning_rate=0.001, epochs=10,
            batch_size=32
    ):
        self.model = MLPRegressor(input_size=input_size, hidden_size=hidden_size, output_size=output_size)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.epochs = epochs
        self.batch_size = batch_size

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        X_train_np = X_train.values
        y_train_np = y_train.values.reshape(-1, 1)
        train_data = torch.utils.data.TensorDataset(torch.tensor(X_train_np, dtype=torch.float32),
                                                    torch.tensor(y_train_np, dtype=torch.float32))
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=self.batch_size, shuffle=True)

        self.model.train()
        for epoch in range(self.epochs):
            running_loss = 0.0
            for inputs, targets in train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()

            # print(f'Epoch {epoch + 1}/{self.epochs}, Loss: {running_loss / len(train_loader)}')

    def predict(self, X: pd.DataFrame):
        X_np = X.values

        self.model.eval()
        with torch.no_grad():
            inputs = torch.tensor(X_np, dtype=torch.float32)
            outputs = self.model(inputs)
        return outputs.squeeze().numpy()


def train(
        model_name: str, train_list: list[int], random_state: int = 42,
        **kwargs,
) -> Model:
    infos = pd.read_csv('data/Demographics.csv')
    infos['Gender'] = infos['Gender'].apply(lambda x: 1 if x == 'MALE' else 0)
    infos['HbA1c'] = (infos['HbA1c'] - infos['HbA1c'].mean()) / infos['HbA1c'].std()
    train_data = pd.DataFrame()
    for i in train_list:
        df = pd.read_csv(f'data/balanced_data/data_{i:03}.csv')
        df['Gender'] = infos.loc[infos['ID'] == i, 'Gender'].values[0]
        df['HbA1c'] = infos.loc[infos['ID'] == i, 'HbA1c'].values[0]
        train_data = pd.concat([train_data, df])
    train_data.sample(frac=1, random_state=random_state).reset_index(drop=True)
    model = Model(model_name, **kwargs)
    model.fit(train_data.drop(columns=['glucose']), train_data['glucose'])
    return model


def test(model: Model, i: int) -> pd.DataFrame:
    test_data = pd.read_csv(f'data/balanced_data/data_{i:03}.csv').drop(columns=['glucose'])
    infos = pd.read_csv('data/Demographics.csv')
    infos['Gender'] = infos['Gender'].apply(lambda x: 1 if x == 'MALE' else 0)
    infos['HbA1c'] = (infos['HbA1c'] - infos['HbA1c'].mean()) / infos['HbA1c'].std()
    test_data['Gender'] = infos.loc[infos['ID'] == i, 'Gender'].values[0]
    test_data['HbA1c'] = infos.loc[infos['ID'] == i, 'HbA1c'].values[0]
    return model.predict(test_data)


def evaluate(pred: pd.DataFrame, i: int, threshold: float = 0.7) -> tuple[int, int, int, int]:
    # threshold = pred['glucose'].quantile(threshold)
    test_data = pd.read_csv(f"data/{i:03}/Dexcom_{i:03}.csv")
    test_data = test_data[['Timestamp (YYYY-MM-DDThh:mm:ss)', 'Glucose Value (mg/dL)']]
    test_data.columns = ['time', 'glucose']
    test_data.dropna(inplace=True)
    start_time = datetime.datetime.strptime('00:00:00', '%H:%M:%S')

    def transform_time(time):
        t = time.split()[1]
        t = datetime.datetime.strptime(t, '%H:%M:%S')
        t = t - start_time
        t = t.total_seconds()
        t = np.ceil(t / 60)
        return int(t)

    test_data['time'] = test_data['time'].apply(transform_time)
    TP, FP, TN, FN = 0, 0, 0, 0
    for time, glucose in zip(test_data['time'], test_data['glucose']):
        bf1 = glucose > 140
        bf2 = pred[pred['time'] == time]['glucose'].values
        if len(bf2) == 0:
            bf2 = False
        else:
            # bf2 = len(bf2) / sum(map(lambda x: 1 / x, bf2)) > threshold
            bf2 = np.max(bf2) > threshold
        if bf1 and bf2:
            TP += 1
        elif bf1 and not bf2:
            FN += 1
        elif not bf1 and bf2:
            FP += 1
        else:
            TN += 1
    return TP, FP, TN, FN


def validate_model(model_name: str) -> None:
    seed = 607
    recalls = []
    accuracys = []
    threshold = 0.5
    train_list = list(range(1, 17, 2)) + [2, 4]
    validate_list = [8, 16]
    model = train(
        model_name=model_name, train_list=train_list, random_state=seed,

        n_estimators=10, max_depth=3, learning_rate=0.1, n_jobs=-1,

        input_size=24, hidden_size=64, output_size=1,
        epochs=30, batch_size=30
    )
    print(f"{model_name} trained on {train_list} and validated on {validate_list}")
    for i in validate_list:
        pred = test(model, i)
        TP, FP, TN, FN = evaluate(pred, i, threshold=threshold)
        recalls.append(TP / (TP + FN))
        accuracys.append((TP + TN) / (TP + FP + TN + FN))
    recalls = np.array(recalls)
    accuracys = np.array(accuracys)
    print(f"Recalls: {recalls.mean()}")
    print(f"Accuracys: {accuracys.mean()}")
    print(pd.DataFrame({'Recall': recalls, 'Accuracy': accuracys, 'name': validate_list}))


def test_model(model_name: str) -> None:
    seed = 607
    recalls = []
    accuracys = []
    threshold = 0.5
    train_list = list(range(1, 17, 2)) + [2, 4]
    test_list = [6, 10, 12, 14]
    model = train(
        model_name=model_name, train_list=train_list, random_state=seed,

        n_estimators=10, max_depth=3, learning_rate=0.1, n_jobs=-1,

        input_size=24, hidden_size=64, output_size=1,
        epochs=30, batch_size=30
    )
    print(f"{model_name} trained on {train_list} and tested on {test_list}")
    for i in test_list:
        pred = test(model, i)
        TP, FP, TN, FN = evaluate(pred, i, threshold=threshold)
        recalls.append(TP / (TP + FN))
        accuracys.append((TP + TN) / (TP + FP + TN + FN))
    recalls = np.array(recalls)
    accuracys = np.array(accuracys)
    print(f"Recalls: {recalls.mean()}")
    print(f"Accuracys: {accuracys.mean()}")
    print(pd.DataFrame({'Recall': recalls, 'Accuracy': accuracys, 'name': test_list}))

