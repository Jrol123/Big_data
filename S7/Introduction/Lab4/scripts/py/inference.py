"""
Предсказание на новых данных
"""

import mlflow
import pandas as pd
import numpy as np
import joblib
import logging
from datetime import datetime
import json

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("logs/inference.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class InferencePipeline:
    def __init__(self, log_to_mlflow=True):
        self.model = None
        self.scaler = None
        self._log_to_mlflow = log_to_mlflow

    def load_latest_model(self):
        """Загрузка последней обученной модели"""
        logger.info("Загрузка последней модели...")

        try:
            # Загрузка информации о лучшей модели
            with open("models/best_model_info.json", "r") as f:
                model_info = json.load(f)

            model_path = model_info["model_path"]
            self.model = joblib.load(model_path)

            # Загрузка последнего scaler
            scaler_files = [f for f in os.listdir("models") if f.startswith("scaler")]
            if scaler_files:
                latest_scaler = sorted(scaler_files)[-1]
                self.scaler = joblib.load(f"models/{latest_scaler}")

            logger.info(f"Модель загружена: {model_info['model_name']}")
            logger.info(f"Метрики модели: R2={model_info['metrics']['r2']:.4f}")

            return model_info

        except Exception as e:
            logger.error(f"Ошибка загрузки модели: {e}")
            # Попытка загрузки через MLflow
            self.load_from_mlflow()

    def load_from_mlflow(self):
        """Загрузка модели из MLflow"""
        logger.info("Загрузка модели из MLflow...")

        mlflow.set_tracking_uri(
            os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
        )

        # Поиск последней модели
        experiment = mlflow.get_experiment_by_name("California Housing Prediction")
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["metrics.r2_score DESC"],
            max_results=1,
        )

        if len(runs) > 0:
            run_id = runs.iloc[0].run_id
            model_uri = f"runs:/{run_id}/model"
            self.model = mlflow.sklearn.load_model(model_uri)
            logger.info(f"Модель загружена из MLflow, Run ID: {run_id}")

    def _log_inference_to_mlflow(self, input_data, predictions, run_name=None):
        """Логирование результатов инференса в MLflow"""
        if run_name is None:
            run_name = f"inference_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        with mlflow.start_run(run_name=run_name) as run:
            # Устанавливаем тег, чтобы отличать инференс от обучения
            mlflow.set_tag("stage", "inference")
            mlflow.set_tag("model_used", str(type(self.model).__name__))

            # Логируем параметры
            if isinstance(input_data, dict):
                mlflow.log_params(input_data)
            elif isinstance(input_data, pd.DataFrame):
                mlflow.log_param("num_samples", len(input_data))
                mlflow.log_param("num_features", len(input_data.columns))

            # Логируем метрики (если есть)
            if predictions and len(predictions) > 0:
                mlflow.log_metric("num_predictions", len(predictions))
                if isinstance(predictions, (list, np.ndarray)):
                    mlflow.log_metric("prediction_mean", float(np.mean(predictions)))
                    mlflow.log_metric("prediction_std", float(np.std(predictions)))

            # Логируем артефакты (результаты предсказания)
            if isinstance(input_data, pd.DataFrame):
                # Сохраняем результаты в файл
                result_df = input_data.copy()
                result_df["prediction"] = predictions
                artifact_path = (
                    f"inference_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                )
                result_df.to_csv(artifact_path, index=False)
                mlflow.log_artifact(artifact_path, "inference_results")

            logger.info(
                f"Inference залогирован в run: {run.info.run_id} с именем: {run_name}"
            )
            return run.info.run_id

    def predict(self, input_data, run_name=None):
        """Предсказание на новых данных"""
        logger.info("Выполнение предсказаний...")

        try:
            # Преобразование входных данных
            if isinstance(input_data, dict):
                df = pd.DataFrame([input_data])
            elif isinstance(input_data, list):
                df = pd.DataFrame(input_data)
            else:
                df = input_data.copy()

            # Масштабирование признаков
            if self.scaler:
                df_scaled = self.scaler.transform(df)
            else:
                df_scaled = df.values

            # Предсказание
            predictions = self.model.predict(df_scaled)

            if self._log_to_mlflow:
                self._log_inference_to_mlflow(df, predictions, run_name)

            # Создание результата
            result = {
                "predictions": predictions.tolist(),
                "timestamp": datetime.now().isoformat(),
                "model_used": str(type(self.model).__name__),
            }

            logger.info(f"Выполнено предсказаний: {len(predictions)}")

            return result

        except Exception as e:
            logger.error(f"Ошибка при предсказании: {e}")
            raise

    def batch_predict(self, file_path, run_name=None):
        """Пакетное предсказание из файла"""
        logger.info(f"Пакетное предсказание из файла: {file_path}")

        # Загрузка данных
        df = pd.read_csv(file_path)

        # Предсказание
        predictions = self.predict(df, run_name=run_name)

        # Сохранение результатов
        output_file = f"data/predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df["predicted_value"] = predictions["predictions"]
        df.to_csv(output_file, index=False)

        logger.info(f"Результаты сохранены в: {output_file}")

        return output_file, predictions


def main():
    mlflow.autolog(disable=True)
    mlflow.set_experiment("California Housing Prediction")
    # Инициализация пайплайна
    pipeline = InferencePipeline()

    # Загрузка модели
    model_info = pipeline.load_latest_model()

    # Пример данных для предсказания
    sample_data = {
        "MedInc": 3.0,
        "HouseAge": 25.0,
        "AveRooms": 5.0,
        "AveBedrms": 1.0,
        "Population": 1000.0,
        "AveOccup": 3.0,
        "Latitude": 34.0,
        "Longitude": -118.0,
    }

    # Единичное предсказание
    result = pipeline.predict(sample_data)
    print(f"\nРезультат предсказания: {result}")

    # Пакетное предсказание (если есть файл)
    test_files = [f for f in os.listdir("data") if "test" in f and f.endswith(".csv")]
    if test_files:
        test_file = os.path.join("data", test_files[0])
        output_file, batch_result = pipeline.batch_predict(test_file)
        print(f"\nПакетное предсказание завершено. Результаты в: {output_file}")


if __name__ == "__main__":
    import os

    main()
