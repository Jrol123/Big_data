@REM Создание структуры проекта
mkdir mlflow-project
cd mlflow-project
mkdir data
mkdir models
mkdir scripts
mkdir notebooks
mkdir logs
cd ..
@REM Копирование всех файлов в соответствующие директории
scp scripts/* mlflow-project/scripts/
scp notebooks/*.ipynb mlflow-project/notebooks/

@REM Запуск проекта
cd mlflow-project
docker compose down
docker compose up -d

@REM Проверка сервисов
docker compose ps

@REM Доступ к интерфейсам:
echo MLflow UI: http://localhost:5000
echo MinIO UI: http://localhost:9001
echo Jupyter: http://localhost:8888

@REM Просмотр логов
docker compose logs -f training-service

@REM @REM 8. Запуск обучения вручную
@REM docker compose exec training-service python scripts/train.py

@REM @REM 9. Запуск инференса
@REM docker compose exec training-service python scripts/inference.py