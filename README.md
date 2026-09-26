# Развёртывание модели классификации изображений на NVIDIA Triton Inference Server

Практическая работа №11: лучшая модель классификации изображений из практической работы №10
(«Глубокая CNN», F1 = 0.981, классы **cats / dogs / panda**, вход 32×32×3) развёрнута через
**NVIDIA Triton Inference Server** с API-шлюзом на FastAPI, мониторингом Prometheus + Grafana
и нагрузочным тестированием Dynamic Batching.

## Архитектура решения

```
Клиент (Streamlit / Swagger / скрипты)
        │  POST /predict (multipart-файл или base64 в JSON)
        ▼
FastAPI Gateway (:8080)  ←→  Triton Inference Server (gRPC :8001, HTTP :8000)
                                     │  бэкенд onnxruntime, ONNX-модель
                                     ▼
                              метрики :8002/metrics
                                     │
                              Prometheus (:9090)
                                     │
                              Grafana (:3000) — дашборд Triton Inference Monitoring
```

- Препроцессинг изображения (resize 32×32) выполняется в шлюзе, нормализация `/255`
  зашита внутрь ONNX-графа.
- Ответ `/predict`: предсказанный класс, вероятности всех классов, время инференса Triton.

## Запуск

```bash
git clone <ссылка на ваш репозиторий>
cd triton-classification
docker-compose up -d --build
```

После старта (Triton становится готовым за ~10–30 с):

| Сервис | Адрес |
|---|---|
| FastAPI Swagger UI | http://localhost:8080/docs |
| Triton HTTP API | http://localhost:8000 |
| Prometheus | http://localhost:9090 |
| Grafana (admin/admin) | http://localhost:3000 |

## API endpoints

| Метод | Путь | Описание |
|---|---|---|
| GET | `/health` | Состояние шлюза и Triton (`triton_live`, `model_ready`) |
| GET | `/classes` | Список классов модели |
| POST | `/predict` | Классификация изображения: JSON `{"image": "<base64>"}` |
| POST | `/predict/file` | Классификация multipart-файла (удобно из Swagger UI) |

## Проверка

```bash
# статус модели в Triton
curl http://localhost:8000/v2/models/image_classifier

# предсказание
curl -X POST http://localhost:8080/predict \
     -H "Content-Type: application/json" \
     -d "{\"image\": \"<base64 изображения PNG 32x32>\"}"

# функциональные и стресс-тесты
python test_client.py

# нагрузочный тест (1000 запросов, 50 параллельных)
pip install aiohttp pillow numpy
python load_test.py
```

## Нагрузочное тестирование и Dynamic Batching

Конфигурации батчинга задаются в `model_repository/image_classifier/config.pbtxt`
(блок `dynamic_batching`); после изменения конфига — `docker-compose restart triton`.

| Конфигурация | preferred_batch_size | max_queue_delay_microseconds |
|---|---|---|
| Без батчинга | — (блок удалён) | — |
| Малый батч | [2, 4] | 50000 |
| Средний батч | [4, 8, 16] | 100000 |
| Большой батч | [8, 16, 32] | 200000 |

Результаты замеров и графики — в ноутбуке практической работы (Раздел 5).

## Мониторинг

Grafana автоматически (provisioning) подхватывает источник данных Prometheus
и дашборд `Triton Inference Monitoring` (`grafana/provisioning/dashboards/triton.json`) с панелями:
RPS (успешные/ошибки), латентность p50/p95/p99 (`histogram_quantile` по
`nv_inference_request_duration_us_bucket`), очередь запросов, счётчики запросов,
среднее время в очереди.
