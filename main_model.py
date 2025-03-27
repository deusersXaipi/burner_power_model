import pandas as pd
import numpy as np
import joblib
import optuna
import mysql.connector
import logging
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import PowerTransformer
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                            mean_absolute_percentage_error, explained_variance_score,
                            max_error, median_absolute_error)
from xgboost import XGBRegressor

# Setup logging
logging.basicConfig(
    filename='logs/burner_model_trainer.log',
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class BurnerModelTrainer:
    def __init__(self, host, user, password, database, table_name,
                metrics_table="efficiency_metrics1",
                model_path="route_to_/burner_model.pkl"):
        self.host = host
        self.user = user
        self.password = password
        self.database = database
        self.table_name = table_name
        self.metrics_table = metrics_table
        self.model_path = model_path

    def connect(self):
        return mysql.connector.connect(
            host=self.host,
            user=self.user,
            password=self.password,
            database=self.database,
            port=3306
        )

    def load_data(self):
        conn = self.connect()
        query = f"SELECT * FROM {self.table_name};"
        data = pd.read_sql(query, conn)
        conn.close()
        return data

    def preprocess_data(self, df):
        df = df.copy()
        df = df.replace([np.inf, -np.inf], np.nan).dropna()
        df['FUEL_PER_PROD'] = df['FUEL'] / df['PRODUCTION']
        df['TEMP_PER_MIN'] = df['TEMPERATURE'] / df['MINUTE_TIME']
        df['FUEL_PER_MIN'] = df['FUEL'] / df['MINUTE_TIME']
        df['PROD_PER_MIN'] = df['PRODUCTION'] / df['MINUTE_TIME']
        # Filtrar valores atípicos
        q_high = df['BURNER_POWER'].quantile(0.99)

        # Filtrar valores donde BURNER_POWER < 50
        df = df[(df['BURNER_POWER'] >= 50) & (df['BURNER_POWER'] <= q_high)]
        
        return df

    def optimize_model(self, X, y_trans):
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 600),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 5.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 5.0)
            }
            model = XGBRegressor(**params, random_state=42)
            score = cross_val_score(model, X, y_trans, scoring="neg_mean_squared_error", cv=3).mean()
            return -score

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=50)
        return study.best_params

    def evaluate_model(self, y_true, y_pred):
        return {
            "mae": mean_absolute_error(y_true, y_pred),
            "rmse": mean_squared_error(y_true, y_pred, squared=False),
            "r2_score": r2_score(y_true, y_pred),
            "mape": mean_absolute_percentage_error(y_true, y_pred),
            "explained_variance": explained_variance_score(y_true, y_pred),
            "max_error": max_error(y_true, y_pred),
            "median_absolute_error": median_absolute_error(y_true, y_pred)
        }

    def save_metrics(self, metrics_dict):
        conn = self.connect()
        cursor = conn.cursor()

        cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {self.metrics_table} (
            id INT AUTO_INCREMENT PRIMARY KEY,
            model_name VARCHAR(100),
            mae FLOAT,
            rmse FLOAT,
            r2_score FLOAT,
            mape FLOAT,
            explained_variance FLOAT,
            max_error FLOAT,
            median_absolute_error FLOAT,
            date_trained DATETIME DEFAULT CURRENT_TIMESTAMP
        );
        """)

        insert_query = f"""
        INSERT INTO {self.metrics_table} 
        (model_name, mae, rmse, r2_score, mape, explained_variance, max_error, median_absolute_error)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """
        cursor.execute(insert_query, (
            "burner_optuna_xgb",
            metrics_dict["mae"],
            metrics_dict["rmse"],
            metrics_dict["r2_score"],
            metrics_dict["mape"],
            metrics_dict["explained_variance"],
            metrics_dict["max_error"],
            metrics_dict["median_absolute_error"]
        ))
        conn.commit()
        cursor.close()
        conn.close()

    def save_model(self, model, transformer):
        joblib.dump({"model": model, "transformer": transformer}, self.model_path)

    def run(self):
        logger.info("Loading data from database...")
        df = self.load_data()
        df = self.preprocess_data(df)

        features = [
            'FUEL', 'PRODUCTION', 'MINUTE_TIME', 'TEMPERATURE',
            'FUEL_PER_PROD', 'TEMP_PER_MIN', 'FUEL_PER_MIN', 'PROD_PER_MIN',
            'ADJUSTMENT_FACTOR', 'EFFICIENCY', 'MANUAL_EXECUTION'
        ]
        df = df[features + ['BURNER_POWER']].dropna()
        X = df[features]
        y = df['BURNER_POWER']

        pt = PowerTransformer()
        y_trans = pt.fit_transform(y.values.reshape(-1, 1)).ravel()

        logger.info("Optimizing hyperparameters with Optuna...")
        best_params = self.optimize_model(X, y_trans)
        logger.info(f"Best parameters: {best_params}")

        logger.info("Training model...")
        X_train, X_test, y_train, y_test = train_test_split(X, y_trans, test_size=0.2, random_state=42)
        model = XGBRegressor(**best_params, random_state=42)
        model.fit(X_train, y_train)

        y_pred_trans = model.predict(X_test)
        y_pred = pt.inverse_transform(y_pred_trans.reshape(-1, 1)).ravel()
        y_test_orig = pt.inverse_transform(y_test.reshape(-1, 1)).ravel()

        metrics = self.evaluate_model(y_test_orig, y_pred)
        for k, v in metrics.items():
            logger.info(f"{k.upper()}: {v:.4f}")

        self.save_model(model, pt)
        self.save_metrics(metrics)
        logger.info("Model and metrics saved successfully.")
        logger.info("----------------------------------------------------------------------------------")


trainer = BurnerModelTrainer(
    host="your_mysql_host",
    user="your_user",
    password="your_password",
    database="your_database",
    table_name="your_table"
)
trainer.run()
