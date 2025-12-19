"""
Проверка данных

Проверка на пропущенные значения, выбросы и т. д.
"""
import pandas as pd
import numpy as np
import logging
from datetime import datetime
import json
import os

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("logs/data_validation.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class DataValidator:
    def __init__(self):
        self.validation_report = {}

    def validate_data(self, data_path):
        """Валидация и предобработка данных"""
        logger.info(f"Начало валидации данных: {data_path}")

        try:
            # Загрузка данных
            df = pd.read_csv(data_path)

            # 1. Проверка структуры данных
            self.validation_report["shape"] = df.shape
            self.validation_report["columns"] = df.columns.tolist()
            self.validation_report["dtypes"] = df.dtypes.astype(str).to_dict()

            # 2. Проверка на пропущенные значения
            missing_values = df.isnull().sum()
            missing_percentage = (missing_values / len(df)) * 100

            self.validation_report["missing_values"] = missing_values[
                missing_values > 0
            ].to_dict()
            self.validation_report["missing_percentage"] = missing_percentage[
                missing_percentage > 0
            ].to_dict()

            if missing_values.sum() > 0:
                logger.warning(
                    f"Обнаружены пропущенные значения: {missing_values.sum()}"
                )
                # Заполнение пропущенных значений медианой
                for col in df.columns:
                    if df[col].isnull().sum() > 0:
                        df[col].fillna(df[col].median(), inplace=True)

            # 3. Проверка на выбросы
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            outliers_report = {}

            for col in numeric_cols:
                if col not in ["collection_date"]:
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR

                    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
                    outliers_count = len(outliers)

                    if outliers_count > 0:
                        outliers_report[col] = {
                            "count": outliers_count,
                            "percentage": (outliers_count / len(df)) * 100,
                            "lower_bound": lower_bound,
                            "upper_bound": upper_bound,
                        }

            self.validation_report["outliers"] = outliers_report

            # 4. Проверка распределения
            distribution_report = {}
            for col in numeric_cols:
                if col not in ["collection_date"]:
                    distribution_report[col] = {
                        "mean": float(df[col].mean()),
                        "std": float(df[col].std()),
                        "min": float(df[col].min()),
                        "max": float(df[col].max()),
                        "skew": float(df[col].skew()),
                    }

            self.validation_report["distribution"] = distribution_report

            # 5. Сохранение очищенных данных
            cleaned_path = data_path.replace(".csv", "_cleaned.csv")
            df.to_csv(cleaned_path, index=False)

            # 6. Сохранение отчета
            report_path = f"logs/validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(report_path, "w") as f:
                json.dump(self.validation_report, f, indent=4)

            logger.info(f"Валидация завершена. Отчет сохранен в {report_path}")
            logger.info(f"Очищенные данные сохранены в {cleaned_path}")

            return cleaned_path, self.validation_report

        except Exception as e:
            logger.error(f"Ошибка при валидации данных: {e}")
            raise


def main():
    # Получение пути к последнему датасету
    try:
        with open("data/latest_dataset.txt", "r") as f:
            data_path = f.read().strip()
    except:
        # Поиск последнего файла
        data_files = [f for f in os.listdir("data") if f.endswith(".csv")]
        if not data_files:
            raise FileNotFoundError("Нет доступных датасетов")
        data_path = os.path.join("data", sorted(data_files)[-1])

    validator = DataValidator()
    cleaned_path, report = validator.validate_data(data_path)

    # Сохранение пути к очищенным данным
    with open("data/latest_cleaned.txt", "w") as f:
        f.write(cleaned_path)

    return cleaned_path


if __name__ == "__main__":
    main()
