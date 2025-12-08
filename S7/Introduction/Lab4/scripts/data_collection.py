"""
Загрузка данных
"""
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
import logging
from datetime import datetime
import os

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("logs/data_collection.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def collect_data():
    """Автоматический сбор данных"""
    logger.info("Начало сбора данных...")

    try:
        # Загрузка датасета California Housing
        housing = fetch_california_housing()
        df = pd.DataFrame(housing.data, columns=housing.feature_names)
        df["MedHouseVal"] = housing.target

        # Добавление временной метки
        df["collection_date"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Сохранение данных
        os.makedirs("data", exist_ok=True)
        data_path = f"data/housing_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(data_path, index=False)

        logger.info(f"Данные успешно собраны и сохранены в {data_path}")
        logger.info(f"Размер датасета: {df.shape}")
        logger.info(f"Колонки: {df.columns.tolist()}")

        # Сохранение пути к последнему датасету
        with open("data/latest_dataset.txt", "w") as f:
            f.write(data_path)

        return data_path

    except Exception as e:
        logger.error(f"Ошибка при сборе данных: {e}")
        raise


if __name__ == "__main__":
    collect_data()
