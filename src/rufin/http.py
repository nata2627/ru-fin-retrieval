"""Вежливый HTTP-клиент для государственных сайтов.

Сайты ЦБ и Минфина не рассчитаны на массовый автоматический сбор, поэтому
запросы идут строго последовательно, с паузой и внятным User-Agent. Никакого
параллелизма: выигрыш в минутах не стоит нагрузки на чужой сервер.
"""
from __future__ import annotations

import time
import requests

# заголовки HTTP передаются в latin-1, поэтому строка только из ASCII
UA = "ru-fin-retrieval/0.1 (research corpus collection; sequential, rate-limited)"


class Client:
    def __init__(self, pause: float = 1.5, timeout: int = 120, retries: int = 3):
        self.pause = pause
        self.timeout = timeout
        self.retries = retries
        self._s = requests.Session()
        self._s.headers["User-Agent"] = UA
        self._last = 0.0

    def get(self, url: str, **params) -> requests.Response:
        """GET с соблюдением паузы между запросами и повтором при сетевом сбое."""
        last_err = None
        for attempt in range(self.retries):
            # пауза считается от конца предыдущего запроса, а не от начала:
            # иначе медленные ответы «съедают» задержку и темп растёт
            wait = self.pause - (time.monotonic() - self._last)
            if wait > 0:
                time.sleep(wait)
            try:
                r = self._s.get(url, params=params or None, timeout=self.timeout)
                self._last = time.monotonic()
                r.raise_for_status()
                return r
            except Exception as e:  # noqa: BLE001 - сеть, причина не важна, важен повтор
                last_err = e
                self._last = time.monotonic()
                time.sleep(2 ** attempt)
        raise RuntimeError(f"не удалось получить {url}: {last_err}")
