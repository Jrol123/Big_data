@REM 1. Клонирование MLflow (опционально, если нужны исходники)
@REM git clone https://github.com/mlflow/mlflow.git

@REM 2. Создание структуры проекта
mkdir mlflow-project
cd mlflow-project
mkdir data
mkdir models
mkdir scripts
mkdir notebooks
mkdir logs
cd ..
@REM # 3. Копирование всех файлов в соответствующие директории
scp scripts/* mlflow-project/scripts/
scp notebooks/* mlflow-project/notebooks/

@REM 4. Запуск проекта
cd mlflow-project
docker compose down -v
docker compose up -d

@REM 5. Проверка сервисов
docker compose ps

@REM 6. Доступ к интерфейсам:
echo MLflow UI: http://localhost:5000
echo MinIO UI: http://localhost:9001 (логин: minio, пароль: minio123)
echo Jupyter: http://localhost:8888

@REM 7. Просмотр логов
docker compose logs -f training-service

@REM @REM 8. Запуск обучения вручную
@REM docker compose exec training-service python scripts/train.py

@REM @REM 9. Запуск инференса
@REM docker compose exec training-service python scripts/inference.py