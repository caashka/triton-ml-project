"""
Скрипт нагрузочного тестирования API
"""

import asyncio
import aiohttp
import time
import numpy as np
from PIL import Image
import io
import base64


# === НАСТРОЙКИ ===
API_URL = "http://localhost:8080/predict"
TOTAL_REQUESTS = 1000
CONCURRENT = 50


def generate_test_image() -> str:
    """Генерирует случайное изображение 32x32 в base64 (формат входа модели image_classifier)"""
    img = Image.fromarray(np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8))
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode()


async def send_request(session, semaphore, payload):
    """Отправляет один запрос"""
    async with semaphore:
        start = time.perf_counter()
        try:
            async with session.post(API_URL, json=payload, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                await resp.json()
                return resp.status == 200, time.perf_counter() - start
        except Exception:
            return False, time.perf_counter() - start


async def main():
    print(f"🚀 Запуск теста: {TOTAL_REQUESTS} запросов, {CONCURRENT} параллельных\n")

    payload = {"image": generate_test_image()}
    semaphore = asyncio.Semaphore(CONCURRENT)

    async with aiohttp.ClientSession() as session:
        # Прогрев
        await asyncio.gather(*[send_request(session, semaphore, payload) for _ in range(5)])

        # Основной тест
        start_time = time.perf_counter()
        results = await asyncio.gather(*[send_request(session, semaphore, payload) for _ in range(TOTAL_REQUESTS)])
        total_time = time.perf_counter() - start_time

    # Подсчёт результатов
    successful = [r[1] for r in results if r[0]]
    failed = len(results) - len(successful)

    # Вывод
    print(f"📊 РЕЗУЛЬТАТЫ")
    print(f"{'='*40}")
    print(f"Успешных:    {len(successful)}/{TOTAL_REQUESTS} ({len(successful)/TOTAL_REQUESTS*100:.1f}%)")
    print(f"Неуспешных:  {failed}")
    print(f"Время:       {total_time:.2f} сек")
    print(f"\n⚡ Throughput: {TOTAL_REQUESTS/total_time:.2f} RPS")
    print(f"\n⏱️  Латентность (мс):")
    print(f"   Avg:  {np.mean(successful)*1000:.2f}")
    print(f"   p50:  {np.percentile(successful, 50)*1000:.2f}")
    print(f"   p95:  {np.percentile(successful, 95)*1000:.2f}")
    print(f"   p99:  {np.percentile(successful, 99)*1000:.2f}")


if __name__ == "__main__":
    asyncio.run(main())
