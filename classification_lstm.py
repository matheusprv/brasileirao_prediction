from datetime import datetime
print("Horário de início: ", datetime.now())


from Classification_LSTM_Model import CustomDataGenerator, build_model, prepare_dataframes_and_infos

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.model_selection import StratifiedKFold, ParameterGrid


train_df, test_df, teams = prepare_dataframes_and_infos("BRA-pre-processed-lstm.csv")

treino = CustomDataGenerator(teams, train_df, 256)
teste = CustomDataGenerator(teams, test_df, 2)



def run_cv_grid(teams_info, train_df, test_df, n_splits=10, param_grid=None, epochs=10, patience=3, verbose=0):
    """
    Returns: best_params, cv_summary (dict), best_model (retrained on full train), test_metrics
    """
    if param_grid is None:
        param_grid = {
            "lstm_units": [16, 32],
            "curr_dense_units": [32, 64, 128],
            "hidden_1": [224, 256, 272],
            "hidden_2": [112, 128, 144],
            "optimizer_name": ["adam", "sgd"],
            "lr": [0.01, 0.001],
        }

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    y = train_df["Res"].to_numpy()
    folds = list(skf.split(train_df, y))

    grid = list(ParameterGrid(param_grid))
    cv_scores = []
    cv_details = []

    for gi, params in enumerate(grid, start=1):
        fold_metrics = []
        for fi, (tr_idx, va_idx) in enumerate(folds, start=1):
            K.clear_session()

            # Split dataframes
            tr_df = train_df.iloc[tr_idx].sample(frac=1, random_state=42)
            va_df = train_df.iloc[va_idx]

            # Generators
            tr_gen = CustomDataGenerator(teams_info, tr_df, batch_size=256)
            va_gen = CustomDataGenerator(teams_info, va_df, batch_size=16)

            # Model
            model = build_model(
                lstm_units=params["lstm_units"],
                curr_dense_units=params["curr_dense_units"],
                hidden_1=params["hidden_1"],
                hidden_2=params["hidden_2"],
                optimizer_name=params["optimizer_name"],
                lr=params["lr"],
            )

            # Train
            es = EarlyStopping(
                monitor="val_loss",
                patience=patience,
                restore_best_weights=True,
                verbose=0
            )

            history = model.fit(
                tr_gen,
                validation_data=va_gen,
                epochs=epochs,
                callbacks=[es],
                verbose=verbose
            )

            # Evaluate on validation
            val_loss, val_acc = model.evaluate(va_gen, verbose=0)
            fold_metrics.append({"val_loss": float(val_loss), "val_acc": float(val_acc)})

        # Aggregate across folds
        mean_acc = float(np.mean([m["val_acc"] for m in fold_metrics]))
        mean_loss = float(np.mean([m["val_loss"] for m in fold_metrics]))
        cv_scores.append({"params": params, "mean_val_acc": mean_acc, "mean_val_loss": mean_loss})
        cv_details.append({"params": params, "folds": fold_metrics})
        print(f"[{gi}/{len(grid)}] params={params}  -> mean_val_acc={mean_acc:.4f}, mean_val_loss={mean_loss:.4f}")

    cv_scores_sorted = sorted(cv_scores, key=lambda x: (-x["mean_val_acc"], x["mean_val_loss"]))
    best_params = cv_scores_sorted[0]["params"]

    print("\nBest params:", best_params)
    print("="*50)

    K.clear_session()
    full_tr_gen = CustomDataGenerator(teams_info, train_df.sample(frac=1, random_state=42), batch_size=256)
    test_gen = CustomDataGenerator(teams_info, test_df, batch_size=2) 

    best_model = build_model(
        lstm_units=best_params["lstm_units"],
        curr_dense_units=best_params["curr_dense_units"],
        hidden_1=best_params["hidden_1"],
        hidden_2=best_params["hidden_2"],
        optimizer_name=best_params["optimizer_name"],
        lr=best_params["lr"],
    )

    es_full = EarlyStopping(monitor="loss", patience=patience, restore_best_weights=True, verbose=0)
    best_model.fit(full_tr_gen, epochs=20, callbacks=[es_full], verbose=verbose)

    test_loss, test_acc = best_model.evaluate(test_gen, verbose=0)
    print(f"Test -> loss={test_loss:.4f}, acc={test_acc:.4f}")

    summary = {
        "leaderboard": cv_scores_sorted[:10],
        "all_details": cv_details,
        "test_metrics": {"loss": float(test_loss), "acc": float(test_acc)},
    }
    return best_params, summary, best_model


best_params, cv_summary, best_model = run_cv_grid(
    teams_info=teams,
    train_df=train_df,
    test_df=test_df,
    n_splits=10,
    epochs=10,        
    patience=2,
    verbose=0
)

print("Best params:", best_params)
print("Top-3 by mean_val_acc:")
for row in cv_summary["leaderboard"][:3]:
    print(row)
print("Test metrics:", cv_summary["test_metrics"])



print("Horário de finalização: ", datetime.now())
