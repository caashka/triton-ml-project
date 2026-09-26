"""
FastAPI Gateway для Triton Inference Server (gRPC версия).
Этот модуль создаёт REST API, который принимает запросы от клиентов
и перенаправляет их в Triton Inference Server для получения предсказаний.

Модель: image_classifier — классификация изображений 32x32x3
на 3 класса (cats, dogs, panda).
Имена слоёв модели определяются автоматически из метаданных Triton.
"""

# ═══════════════════════════════════════════════════════════════════
# ИМПОРТЫ
# ═══════════════════════════════════════════════════════════════════

from fastapi import FastAPI, HTTPException, UploadFile, File  # FastAPI — веб-фреймворк, UploadFile/File — приём файлов
from pydantic import BaseModel, Field       # Pydantic — валидация входных данных из JSON
import tritonclient.grpc as grpcclient      # gRPC клиент для общения с Triton Server (порт 8001)
import numpy as np                          # NumPy — работа с числовыми массивами
import asyncio                              # asyncio.sleep — асинхронная пауза без блокировки
import os                                   # os — работа с переменными окружения и файловой системой
import base64                               # base64 — декодирование изображения из JSON-запроса
import cv2                                  # OpenCV — декодирование и ресайз изображений
import time                                 # time — замер времени инференса

# ═══════════════════════════════════════════════════════════════════
# КОНФИГУРАЦИЯ
# ═══════════════════════════════════════════════════════════════════

# Адрес Triton сервера. os.getenv читает переменную окружения,
# если её нет — берёт значение по умолчанию "triton:8001"
# Порт 8001 — это gRPC порт Triton (8000 — HTTP, 8002 — метрики)
TRITON_URL = os.getenv("TRITON_URL", "triton:8001")

# Имя модели в Triton. Должно совпадать с названием папки в model_repository/
MODEL_NAME = os.getenv("MODEL_NAME", "image_classifier")

# Классы модели в порядке выходных нейронов (как в обучающей выборке)
CLASS_NAMES = os.getenv("CLASS_NAMES", "cats,dogs,panda").split(",")

# Размер входного изображения модели
IMG_SIZE = 32

# ═══════════════════════════════════════════════════════════════════
# СОЗДАНИЕ FASTAPI ПРИЛОЖЕНИЯ
# ═══════════════════════════════════════════════════════════════════

app = FastAPI(
    title="Image Classification API (Triton)",  # Заголовок в Swagger UI (/docs)
    description="API для классификации изображений (cats/dogs/panda) через NVIDIA Triton Inference Server",  # Описание в Swagger UI
    version="1.0.0"  # Версия API
)

# ═══════════════════════════════════════════════════════════════════
# ГЛОБАЛЬНЫЕ ПЕРЕМЕННЫЕ
# ═══════════════════════════════════════════════════════════════════

triton_client = None  # gRPC клиент Triton — инициализируется в startup()
input_name = None     # Имя входного слоя модели — получаем из метаданных Triton
output_name = None    # Имя выходного слоя модели — получаем из метаданных Triton

# ═══════════════════════════════════════════════════════════════════
# МОДЕЛИ ДАННЫХ (Pydantic)
# ═══════════════════════════════════════════════════════════════════

class PredictRequest(BaseModel):
    """
    Схема запроса на предсказание в формате JSON:
    {"image": "<base64-строка изображения (PNG/JPEG)>"}
    """
    image: str = Field(..., description="Изображение в base64")


def decode_image_bytes(raw: bytes) -> np.ndarray:
    """
    Декодирует изображение и готовит его к отправке в модель.

    Препроцессинг ПОЛНОСТЬЮ повторяет обучающую выборку практической работы №10:
    - cv2.resize(image, (32, 32)) — интерполяция INTER_LINEAR (по умолчанию)
    - приведение к float32
    Нормализация /255 выполняется ВНУТРИ ONNX-графа, поэтому на вход
    подаются значения 0-255. Каналы остаются в порядке cv2 (BGR), как при обучении.
    """
    buffer = np.frombuffer(raw, dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Не удалось декодировать изображение (поддерживаются PNG/JPEG)")
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    return image.astype(np.float32)  # (32, 32, 3), значения 0-255, BGR


def run_inference(data: np.ndarray) -> np.ndarray:
    """
    Отправляет изображение в Triton и получает вероятности классов.

    :param data: np.ndarray формы (32, 32, 3), float32
    :return: np.ndarray формы (3,) — вероятности классов
    """
    # Добавляем batch-размер: (32, 32, 3) -> (1, 32, 32, 3)
    batch = np.expand_dims(data, axis=0)

    # Формируем входной тензор для Triton
    infer_input = grpcclient.InferInput(input_name, batch.shape, "FP32")
    infer_input.set_data_from_numpy(batch)

    # Указываем, какой выход хотим получить
    requested_output = grpcclient.InferRequestedOutput(output_name)

    # Замеряем время инференса (чистый вызов Triton, без сетевых задержек клиента)
    start = time.perf_counter()
    response = triton_client.infer(MODEL_NAME, inputs=[infer_input], outputs=[requested_output])
    inference_time_ms = (time.perf_counter() - start) * 1000

    probs = response.as_numpy(output_name)[0]
    return probs, inference_time_ms

# ═══════════════════════════════════════════════════════════════════
# STARTUP — выполняется один раз при запуске приложения
# ═══════════════════════════════════════════════════════════════════

@app.on_event("startup")  # Декоратор — эта функция вызовется при старте FastAPI
async def startup():
    """
    Инициализация при запуске:
    1. Подключение к Triton Server
    2. Ожидание готовности модели
    3. Получение имён слоёв из метаданных модели
    """
    global triton_client, input_name, output_name

    print(f"🚀 Запуск API | Triton: {TRITON_URL}")  # Лог в консоль для отладки

    # Создаём gRPC клиент для связи с Triton Server
    triton_client = grpcclient.InferenceServerClient(url=TRITON_URL, verbose=False)

    # Цикл ожидания готовности Triton (макс 30 попыток × 2 сек = 60 сек)
    for attempt in range(30):
        try:
            if triton_client.is_server_live() and triton_client.is_model_ready(MODEL_NAME):
                print(f"✅ Модель '{MODEL_NAME}' готова")

                metadata = triton_client.get_model_metadata(MODEL_NAME)
                input_name = metadata.inputs[0].name
                output_name = metadata.outputs[0].name
                print(f"📊 Слои модели — Input: '{input_name}', Output: '{output_name}'")
                return
        except Exception as e:
            print(f"⏳ Попытка {attempt + 1}/30: {e}")

        await asyncio.sleep(2)

    raise RuntimeError("❌ Не удалось подключиться к Triton!")

# ═══════════════════════════════════════════════════════════════════
# SHUTDOWN — выполняется при остановке приложения
# ═══════════════════════════════════════════════════════════════════

@app.on_event("shutdown")
async def shutdown():
    """Корректное закрытие соединения с Triton при остановке."""
    if triton_client:
        triton_client.close()
        print("🔌 Соединение закрыто")

# ═══════════════════════════════════════════════════════════════════
# ЭНДПОИНТЫ
# ═══════════════════════════════════════════════════════════════════

@app.get("/")
async def root():
    """Информация о сервисе."""
    return {
        "service": "Image Classification API",
        "model": MODEL_NAME,
        "classes": CLASS_NAMES,
        "input_size": f"{IMG_SIZE}x{IMG_SIZE}x3",
        "endpoints": ["GET /health", "GET /classes", "POST /predict"]
    }


@app.get("/health")
async def health():
    """Проверка состояния сервисов: FastAPI-шлюз и Triton."""
    try:
        live = triton_client.is_server_live()
        ready = triton_client.is_model_ready(MODEL_NAME)
        status = "ok" if (live and ready) else "degraded"
        return {
            "api": "ok",
            "triton_live": live,
            "model_ready": ready,
            "status": status
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Triton недоступен: {e}")


@app.get("/classes")
async def classes():
    """Список классов модели."""
    return {"model": MODEL_NAME, "classes": CLASS_NAMES}


# Обработчики намеренно синхронные (def): FastAPI выполняет их в ThreadPoolExecutor,
# поэтому блокирующие gRPC-вызовы Triton не сериализуют обработку запросов,
# и при высокой нагрузке запросы приходят в Triton параллельно —
# это позволяет динамическому батчингу собирать батчи.


def _predict_raw(raw: bytes):
    """Общая логика предсказания для JSON (base64) и multipart (файл)."""
    if triton_client is None or input_name is None:
        raise HTTPException(status_code=503, detail="API ещё не инициализирован")

    image = decode_image_bytes(raw)
    probs, inference_time_ms = run_inference(image)

    predicted_idx = int(np.argmax(probs))
    return {
        "predicted_class": CLASS_NAMES[predicted_idx],
        "class_index": predicted_idx,
        "probabilities": {
            CLASS_NAMES[i]: float(np.round(p, 6)) for i, p in enumerate(probs)
        },
        "inference_time_ms": round(inference_time_ms, 3),
        "model": MODEL_NAME
    }


@app.post("/predict")
def predict(request: PredictRequest):
    """
    Классификация изображения, переданного в JSON в формате base64:
    {"image": "<base64>"}

    Возвращает предсказанный класс и вероятности по всем классам.
    """
    try:
        raw = base64.b64decode(request.image, validate=True)
    except Exception:
        raise HTTPException(status_code=400, detail="Поле 'image' должно быть корректной base64-строкой")
    return _predict_raw(raw)


@app.post("/predict/file")
def predict_file(file: bytes = File(...)):
    """
    Классификация изображения, загруженного как multipart-файл.
    Удобно для проверки через Swagger UI (/docs).
    """
    raw = file
    if not raw:
        raise HTTPException(status_code=400, detail="Пустой файл")
    return _predict_raw(raw)
