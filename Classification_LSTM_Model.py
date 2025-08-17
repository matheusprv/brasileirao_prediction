import pandas as pd
import numpy as np
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
for phy_dev in physical_devices:
  tf.config.experimental.set_memory_growth(phy_dev, True)

from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, LSTM, Dropout, BatchNormalization, Concatenate
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
from tensorflow.keras.callbacks import EarlyStopping


def prepare_dataframes_and_infos(csv_path="BRA-pre-processed-lstm.csv"):
    df = pd.read_csv(csv_path)
    df["match_id"] = range(len(df))
    df

    train_df = df[df["Season"] != 2025]
    test_df = df[df["Season"] == 2025]

    target_col = "Res"
    min_count = train_df[target_col].value_counts().min()  # 1220

    # Downsample each class to 1220
    train_df = (
        train_df.groupby(target_col, group_keys=False)
        .apply(lambda x: x.sample(n=min_count, random_state=42))
    )

    train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)

    print(train_df[target_col].value_counts())

    teams = dict()

    for index, row in df.iterrows():
        match_id = index

        team_id = int(row["Home"])
        if team_id not in teams:
            teams[team_id] = []
        # goals_for, goals_against, res, odd_vitoria_PSC, odd_empate_PSC, odd_derrota_PSC, 
        # odd_vitoria_maxC, odd_empate_maxC, odd_derrota_maxC, odd_vitoria_AvgC, 
        # odd_empate_AvgC, odd_derrota_AvgC, acumulado_vitorias, acumulado_empates, 
        # acumulado_derrotas, acumulado_goals_for, acumulado_goals_against
        teams[team_id].append(
            (
                match_id,
                np.array([
                    row["HG"], row["AG"], 1 if row["Res"] == 2 else 0.5 if row["Res"] == 1 else 0, 
                    row["PSCH"], row["PSCD"], row["PSCA"], row["MaxCH"], row["MaxCD"], row["MaxCA"], 
                    row["AvgCH"], row["AvgCD"], row["AvgCA"], row["W_H"], row["D_H"], 
                    row["L_H"], row["GF_H"], row["GA_H"]
                ])
            )
        )


        team_id = int(row["Away"])
        if team_id not in teams:
            teams[team_id] = []

        teams[team_id].append(
            (
                match_id, 
                np.array([
                    row["AG"], row["HG"], 1 if row["Res"] == 0 else 0.5 if row["Res"] == 1 else 0, 
                    row["PSCA"], row["PSCD"], row["PSCH"], row["MaxCA"], row["MaxCD"], row["MaxCH"], 
                    row["AvgCA"], row["AvgCD"], row["AvgCH"], row["W_A"], row["D_A"], 
                    row["L_A"], row["GF_A"], row["GA_A"]
                ])
            )        
        )
    
    return train_df, test_df, teams





PREVIOUS_MATCHES = 5
PREVIOUS_MATCHES_FEATURES = 17
PREVIOUS_MATCHES, PREVIOUS_MATCHES_FEATURES

one_hot_results = {
    0: np.array([0.0, 0.0, 1.0]),
    1: np.array([0.0, 1.0, 0.0]),
    2: np.array([1.0, 0.0, 0.0]) 
}

class CustomDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, teams_info, matches, batch_size=32, **kwargs):
        super().__init__(**kwargs) 
        self.teams_info = teams_info
        self.matches = matches.to_numpy()
        self.batch_size = batch_size

    def __len__(self):
        return len(self.matches) // self.batch_size
        
    def __get_previous_matches(self, team, match_id):
        """
            Percorre a lista de partidas de forma inversa pegando as 5 partidas anteriores
        """
        team_matches = self.teams_info[team]
        
        previous_matches = np.zeros((PREVIOUS_MATCHES, PREVIOUS_MATCHES_FEATURES))
        total_matches = 0

        for curr_match in reversed(team_matches):
            if curr_match[0] < match_id:
                previous_matches[-1 - total_matches] = curr_match[1]
                total_matches += 1
                if total_matches == PREVIOUS_MATCHES:
                    break

        return previous_matches

    def __get_data(self, batch):
        B = len(batch)
        X_home = np.zeros((B, PREVIOUS_MATCHES, PREVIOUS_MATCHES_FEATURES), dtype=np.float32)
        X_away = np.zeros((B, PREVIOUS_MATCHES, PREVIOUS_MATCHES_FEATURES), dtype=np.float32)
        X_curr = np.zeros((B, 27), dtype=np.float32)
        Y = np.zeros((B, 3), dtype=np.float32)

        for i, x in enumerate(batch):
            home_team = int(x[1])
            away_team = int(x[2])
            match_id  = int(x[-1])

            X_home[i] = self.__get_previous_matches(home_team, match_id)
            X_away[i] = self.__get_previous_matches(away_team, match_id)
            X_curr[i] = x[6:33].astype(np.float32)

            Y[i] = one_hot_results[int(x[5])]

        X = {
            "Home team previous matches": X_home,
            "Away team previous matches": X_away,
            "Current match infos": X_curr,
        }
        return X, Y

    def __getitem__(self, index):
        begin = self.batch_size * index
        end   = self.batch_size * (index + 1)
        batch = self.matches[begin : end]
        return self.__get_data(batch)


def build_model(
    lstm_units=16,
    curr_dense_units=128,
    hidden_1=256,
    hidden_2=64,
    hidden_3=16,
    hidden_activations="relu",
    optimizer_name="sgd",
    lr=1e-2,
):
    # Inputs
    input_home = Input(shape=(PREVIOUS_MATCHES, PREVIOUS_MATCHES_FEATURES), name="Home team previous matches")
    input_away = Input(shape=(PREVIOUS_MATCHES, PREVIOUS_MATCHES_FEATURES), name="Away team previous matches")
    curr_match = Input(shape=(27,), name="Current match infos")

    # Branches
    x_home = LSTM(lstm_units, return_sequences=False, name="LSTM_Home")(input_home)
    x_away = LSTM(lstm_units, return_sequences=False, name="LSTM_Away")(input_away)
    x_curr = Dense(curr_dense_units, activation="sigmoid", name="Dense_Curr")(curr_match)

    x = Concatenate(name="Concat")([x_home, x_away, x_curr])
    x = Dense(hidden_1, activation=hidden_activations)(x)
    x = Dense(hidden_2, activation=hidden_activations)(x)
    x = Dense(hidden_3, activation=hidden_activations)(x)
    out = Dense(3, activation="softmax", name="out")(x)

    model = Model(inputs=[input_home, input_away, curr_match], outputs=out)

    # Optimizer
    opt = {
        "rmsprop": RMSprop(learning_rate=lr),
        "adam": Adam(learning_rate=lr),
        "sgd": SGD(learning_rate=lr, momentum=0.9),
    }[optimizer_name.lower()]

    model.compile(
        optimizer=opt,
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model
