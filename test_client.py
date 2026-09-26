"""Стресс-тест API"""
import base64
import io
import time
import numpy as np
import requests
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed

API_URL = "http://localhost:8080"


def make_image_b64(size=32):
    """Случайное изображение в base64 (формат входа модели image_classifier)"""
    img = Image.fromarray(np.random.randint(0, 256, (size, size, 3), dtype=np.uint8))
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode()


# ═══════════════════════════════════════════════════════════════
# Вспомогательные функции
# ═══════════════════════════════════════════════════════════════

def request(method, url, **kwargs):
    """Выполняет запрос и возвращает (статус, время_мс)."""
    start = time.perf_counter()
    try:
        resp = requests.request(method, url, timeout=60, **kwargs)
        return resp.status_code, (time.perf_counter() - start) * 1000
    except Exception as e:
        return 0, str(e)


def print_result(name, status, time_ms, expected=200):
    """Выводит результат теста."""
    ok = "✅" if status == expected else "❌"
    time_str = f"{time_ms:.1f} мс" if isinstance(time_ms, float) else time_ms
    print(f"{ok} {name:<45} | Статус: {status:<3} | Время: {time_str}")


def concurrent_requests(url, payload, count):
    """Выполняет count параллельных запросов."""
    with ThreadPoolExecutor(max_workers=count) as executor:
        futures = [executor.submit(request, "POST", url, json=payload) for _ in range(count)]
        return [f.result() for f in as_completed(futures)]


# ═══════════════════════════════════════════════════════════════
# Тесты
# ═══════════════════════════════════════════════════════════════

def run_tests():
    print("=" * 80)
    print("🧪 СТРЕСС-ТЕСТИРОВАНИЕ API")
    print("=" * 80)

    # ─────────── Базовые тесты ───────────
    print("\n📋 БАЗОВЫЕ ТЕСТЫ")
    print("-" * 80)

    status, ms = request("GET", f"{API_URL}/health")
    print_result("Health check", status, ms)

    status, ms = request("GET", f"{API_URL}/classes")
    print_result("Список классов", status, ms)

    status, ms = request("POST", f"{API_URL}/predict", json={"image": make_image_b64()})
    print_result("Predict (base64, 1 изображение)", status, ms)

    # ─────────── Негативные тесты ───────────
    print("\n📋 НЕГАТИВНЫЕ ТЕСТЫ")
    print("-" * 80)

    status, ms = request("POST", f"{API_URL}/predict", json={})
    print_result("Пустой запрос", status, ms, expected=422)

    status, ms = request("POST", f"{API_URL}/predict", json={"image": "не-base64!!!"})
    print_result("Некорректный base64", status, ms, expected=400)

    status, ms = request("POST", f"{API_URL}/predict", json={"image": base64.b64encode(b"not an image").decode()})
    print_result("Не изображение", status, ms, expected=400)

    status, ms = request("GET", f"{API_URL}/nonexistent")
    print_result("Несуществующий эндпоинт", status, ms, expected=404)

    # ─────────── Стресс-тесты (параллельные запросы) ───────────
    print("\n📋 СТРЕСС-ТЕСТЫ (ПАРАЛЛЕЛЬНЫЕ ЗАПРОСЫ)")
    print("-" * 80)

    for concurrent in [10, 50, 100, 200]:
        start = time.perf_counter()
        results = concurrent_requests(f"{API_URL}/predict", {"image": make_image_b64()}, concurrent)
        total_ms = (time.perf_counter() - start) * 1000

        success = sum(1 for s, _ in results if s == 200)
        times = [ms for _, ms in results if isinstance(ms, float)]
        avg_ms = sum(times) / len(times) if times else 0

        ok = "✅" if success == concurrent else "⚠️"
        print(f"{ok} {concurrent} параллельных запросов              | Успешно: {success}/{concurrent:<3} | Общее: {total_ms:>6.0f} мс | Среднее: {avg_ms:.1f} мс")

    # ─────────── Итог ───────────
    print("\n" + "=" * 80)
    print("✅ ТЕСТИРОВАНИЕ ЗАВЕРШЕНО")
    print("=" * 80)


if __name__ == "__main__":
    run_tests()
