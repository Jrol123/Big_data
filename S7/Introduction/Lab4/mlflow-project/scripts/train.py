import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import joblib
from mlflow.models import infer_signature

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("logs/training.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class ModelTrainer:
    def __init__(self):
        self.models = {
            "random_forest": RandomForestRegressor(n_estimators=100, random_state=42),
            "gradient_boosting": GradientBoostingRegressor(
                n_estimators=100, random_state=42
            ),
            "linear_regression": LinearRegression(),
        }

    def load_data(self, data_path):
        """Загрузка данных"""
        logger.info(f"Загрузка данных из {data_path}")
        df = pd.read_csv(data_path)

        # Удаление ненужных колонок
        if "collection_date" in df.columns:
            df = df.drop("collection_date", axis=1)

        # Разделение на признаки и целевую переменную
        X = df.drop("MedHouseVal", axis=1)
        y = df["MedHouseVal"]

        # Разделение на тренировочную и тестовую выборки
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Масштабирование признаков
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        return X_train_scaled, X_test_scaled, y_train, y_test, scaler

    def train_model(self, model_name, model, X_train, X_test, y_train, y_test):
        """Обучение одной модели с логированием в MLflow"""
        logger.info(f"Обучение модели: {model_name}")
        
        # Обучение модели
        model.fit(X_train, y_train)
        
        # Предсказания на тестовых данных
        y_pred = model.predict(X_test)
        
        # Расчет метрик
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Кросс-валидация
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
        
        # Создание сигнатуры модели
        # Берем небольшой пример входных данных (первые 5 строк)
        input_example = X_train[:5]
        
        # Получаем предсказания для примера
        output_example = model.predict(input_example)
        
        # Выводим сигнатуру на основе примера
        signature = infer_signature(input_example, output_example)
        
        logger.info(f"Создана сигнатура для модели {model_name}")
        logger.info(f"Входные данные: {signature.inputs}")
        logger.info(f"Выходные данные: {signature.outputs}")
        
        with mlflow.start_run(run_name=f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            # Логирование параметров
            if hasattr(model, 'get_params'):
                mlflow.log_params(model.get_params())
            
            # Логирование метрик
            mlflow.log_metric("mse", mse)
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("r2_score", r2)
            mlflow.log_metric("cv_r2_mean", cv_scores.mean())
            mlflow.log_metric("cv_r2_std", cv_scores.std())
            
            # Логирование модели С сигнатурой и примером
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path=f"model_{model_name}",
                signature=signature,
                input_example=input_example
            )
            
            # Также сохраняем модель локально
            os.makedirs('models', exist_ok=True)
            model_path = f"models/{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.joblib"
            joblib.dump(model, model_path)
            
            # Сохраняем информацию о модели
            model_info = {
                'model_name': model_name,
                'model_path': model_path,
                'signature': {
                    'inputs': str(signature.inputs),
                    'outputs': str(signature.outputs)
                },
                'input_example_shape': input_example.shape,
                'metrics': {
                    'mse': float(mse),
                    'mae': float(mae),
                    'r2': float(r2),
                    'cv_r2_mean': float(cv_scores.mean()),
                    'cv_r2_std': float(cv_scores.std())
                },
                'timestamp': datetime.now().isoformat()
            }
            
            # Сохраняем информацию о модели в JSON
            info_path = f"models/{model_name}_info.json"
            with open(info_path, 'w') as f:
                json.dump(model_info, f, indent=4)
            
            logger.info(f"Модель {model_name} обучена. R2: {r2:.4f}")
            logger.info(f"Сигнатура сохранена: {signature}")
            
            return model_info

    def create_plots(self, y_test, y_pred, model_name):
        """Создание визуализаций"""
        os.makedirs("logs/plots", exist_ok=True)

        # График предсказаний vs реальных значений
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.plot(
            [y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2
        )
        plt.xlabel("Реальные значения")
        plt.ylabel("Предсказанные значения")
        plt.title(f"Предсказания vs Реальные значения - {model_name}")
        plot_path = f'logs/plots/predictions_{model_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        plt.savefig(plot_path)
        plt.close()

        # Логирование графика в MLflow
        mlflow.log_artifact(plot_path, "plots")

        # График остатков
        plt.figure(figsize=(10, 6))
        residuals = y_test - y_pred
        plt.scatter(y_pred, residuals, alpha=0.5)
        plt.axhline(y=0, color="r", linestyle="--")
        plt.xlabel("Предсказанные значения")
        plt.ylabel("Остатки")
        plt.title(f"График остатков - {model_name}")
        residuals_path = f'logs/plots/residuals_{model_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        plt.savefig(residuals_path)
        plt.close()

        mlflow.log_artifact(residuals_path, "plots")

        # Гистограмма ошибок
        plt.figure(figsize=(10, 6))
        plt.hist(residuals, bins=30, edgecolor="black", alpha=0.7)
        plt.xlabel("Ошибка предсказания")
        plt.ylabel("Частота")
        plt.title(f"Распределение ошибок - {model_name}")
        errors_path = f'logs/plots/errors_{model_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        plt.savefig(errors_path)
        plt.close()

        mlflow.log_artifact(errors_path, "plots")

    def compare_models(self, results):
        """Сравнение моделей и выбор лучшей"""
        logger.info("Сравнение моделей...")

        best_model = None
        best_score = -np.inf

        for result in results:
            if result["metrics"]["r2"] > best_score:
                best_score = result["metrics"]["r2"]
                best_model = result

        # Создание отчета сравнения
        comparison_report = {
            "best_model": best_model["model_name"],
            "best_r2": best_model["metrics"]["r2"],
            "comparison": [
                {
                    "model": r["model_name"],
                    "r2": r["metrics"]["r2"],
                    "mse": r["metrics"]["mse"],
                    "mae": r["metrics"]["mae"],
                }
                for r in results
            ],
            "timestamp": datetime.now().isoformat(),
        }

        # Сохранение отчета
        report_path = (
            f"logs/model_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(report_path, "w") as f:
            json.dump(comparison_report, f, indent=4)

        # Логирование отчета
        mlflow.log_artifact(report_path, "reports")

        logger.info(
            f"Лучшая модель: {best_model['model_name']} с R2: {best_model['metrics']['r2']:.4f}"
        )

        return best_model


def main():
    # Настройка MLflow
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment("California Housing Prediction")

    # Получение пути к очищенным данным
    try:
        with open("data/latest_cleaned.txt", "r") as f:
            data_path = f.read().strip()
    except:
        # Поиск последнего очищенного файла
        data_files = [
            f for f in os.listdir("data") if "cleaned" in f and f.endswith(".csv")
        ]
        if not data_files:
            raise FileNotFoundError("Нет доступных очищенных датасетов")
        data_path = os.path.join("data", sorted(data_files)[-1])

    # Инициализация тренера
    trainer = ModelTrainer()

    # Загрузка данных
    X_train, X_test, y_train, y_test, scaler = trainer.load_data(data_path)

    # Обучение моделей
    results = []
    for model_name, model in trainer.models.items():
        result = trainer.train_model(
            model_name, model, X_train, X_test, y_train, y_test
        )
        results.append(result)

    # Сравнение моделей и выбор лучшей
    best_model = trainer.compare_models(results)

    # Сохранение информации о лучшей модели
    with open("models/best_model_info.json", "w") as f:
        json.dump(
            {
                "model_name": best_model["model_name"],
                "model_path": best_model["model_path"],
                "metrics": best_model["metrics"],
                "timestamp": datetime.now().isoformat(),
            },
            f,
            indent=4,
        )

    # Сохранение scaler
    scaler_path = f"models/scaler_{datetime.now().strftime('%Y%m%d_%H%M%S')}.joblib"
    joblib.dump(scaler, scaler_path)

    logger.info("Обучение завершено успешно!")
    return best_model


if __name__ == "__main__":
    main()
