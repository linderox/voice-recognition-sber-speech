# -*- coding: utf-8 -*-
"""
SmartSpeech (Sber) — Async gRPC клиент + анализ WAV и график эмоций.

Запуск:
  python main.py path/to/file.wav

Окружение (.env или переменные):
  AUTH_BASIC=Base64(client_id:client_secret)  # у тебя уже есть
  CLIENT_ID=78692de0-0fcc-4538-a715-ef2df31da184
  SCOPE=SALUTE_SPEECH_PERS
  # Опционально отключить строгую проверку SSL для NGW OAuth (если на Mac ругается):
  NGW_SKIP_VERIFY=true
"""
import asyncio
import base64
import importlib
import json
import logging
import os
import sys
import time
import uuid
from pydub import AudioSegment
from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

import numpy as np
import requests
import soundfile as sf
import matplotlib.pyplot as plt

import grpc
from dotenv import load_dotenv

# ---------- Конфигурация логгера ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger("smartspeech")

load_dotenv()

# ---------- Константы ----------
NGW_OAUTH_URL = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
GRPC_ENDPOINT = "smartspeech.sber.ru:443"  # gRPC endpoint
CHUNK_SIZE = 2 * 1024 * 1024  # 2MB чанки для Upload
POLL_INTERVAL = 2.0  # сек, опрос статуса задачи
POLL_TIMEOUT = 15 * 60  # 15 минут

# ---------- Учётные данные (подставлены значения из сообщения) ----------
DEFAULT_AUTH_BASIC = os.getenv("AUTH_BASIC", "Nzg2OTJkZTAtMGZjYy00NTM4LWE3MTUtZWYyZGYzMWRhMTg0Ojc4ZDk1ZGFmLTEyM2QtNGRhMy04YTgyLWJlODU3NDc0ZTQ3OQ==")
DEFAULT_CLIENT_ID = os.getenv("CLIENT_ID", "78692de0-0fcc-4538-a715-ef2df31da184")
DEFAULT_SCOPE = os.getenv("SCOPE", "SALUTE_SPEECH_PERS")


# ---------- Утилиты ----------
def human_duration(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h:d}:{m:02d}:{s:02d}"
    return f"{m:d}:{s:02d}"


def parse_pb_duration(ts: str) -> float:
    """Парсинг строк вида '1.480s' -> 1.48 (сек)."""
    if not ts:
        return 0.0
    t = ts.strip().lower()
    if t.endswith("s"):
        t = t[:-1]
    try:
        return float(t)
    except ValueError:
        return 0.0


# ---------- Аналитика WAV ----------
@dataclass
class AudioInfo:
    path: str
    subtype: str
    format: str
    samplerate: int
    channels: int
    frames: int
    duration_s: float
    bit_depth: Optional[int]
    bitrate_kbps: Optional[float]
    per_channel_rms: List[float]
    per_channel_activity_pct: List[float]


def analyze_audio(path: str) -> AudioInfo:
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    ext = os.path.splitext(path)[1].lower()
    fmt = ext.lstrip(".").upper()

    if ext == ".wav":
        with sf.SoundFile(path, "r") as f:
            samplerate = f.samplerate
            channels = f.channels
            frames = len(f)
            subtype = f.subtype  # PCM_16 / PCM_24 / FLOAT / etc.

            bit_depth = None
            if subtype.startswith("PCM_"):
                try:
                    bit_depth = int(subtype.split("_")[1])
                except Exception:
                    bit_depth = None
            elif subtype == "FLOAT":
                bit_depth = 32

            duration_s = frames / float(samplerate) if samplerate else 0.0

            data = f.read(dtype="float32", always_2d=True)

    else:
        seg = AudioSegment.from_file(path)
        samplerate = seg.frame_rate
        channels = seg.channels
        duration_s = seg.duration_seconds
        bit_depth = seg.sample_width * 8 if seg.sample_width else None
        frames = int(duration_s * samplerate)
        subtype = f"PCM_{bit_depth}" if bit_depth else ""

        arr = np.array(seg.get_array_of_samples())
        arr = arr.astype(np.float32)
        if channels > 1:
            arr = arr.reshape((-1, channels))
        else:
            arr = arr.reshape((-1, 1))
        max_val = float(2 ** (bit_depth - 1)) if bit_depth else 1.0
        data = arr / max_val


    # RMS по каналам
    per_channel_rms = []
    per_channel_activity_pct = []
    for ch in range(data.shape[1]):
        x = data[:, ch]
        rms = float(np.sqrt(np.mean(np.square(x)))) if len(x) else 0.0
        per_channel_rms.append(rms)

        # Активность канала (не-тихий % времени)
        thr = max(0.01, 0.2 * rms)  # динамический порог от RMS
        active = float(np.mean(np.abs(x) > thr)) * 100.0
        per_channel_activity_pct.append(active)

    # Битрейт (оценка) = sample_rate * bits * channels / 1000
    bitrate_kbps = None
    if samplerate and channels and bit_depth:
        bitrate_kbps = samplerate * bit_depth * channels / 1000.0

    return AudioInfo(
        path=path,
        format=fmt,
        subtype=subtype,
        samplerate=samplerate,
        channels=channels,
        frames=frames,
        duration_s=duration_s,
        bit_depth=bit_depth,
        bitrate_kbps=bitrate_kbps,
        per_channel_rms=per_channel_rms,
        per_channel_activity_pct=per_channel_activity_pct,
    )


def print_audio_info(ai: AudioInfo) -> None:
    log.info("=== Аудио-информация ===")
    log.info(f"Файл: {ai.path}")
    log.info(f"Формат: {ai.format}")
    if ai.subtype:
        log.info(f"Подтип: {ai.subtype}")
    log.info(f"Частота дискретизации: {ai.samplerate} Гц")
    log.info(f"Каналов: {ai.channels}")
    if ai.bit_depth:
        log.info(f"Битность: {ai.bit_depth} бит")
    if ai.bitrate_kbps:
        log.info(f"Оценочный битрейт: {ai.bitrate_kbps:.1f} kbps")
    log.info(f"Длительность: {human_duration(ai.duration_s)} ({ai.duration_s:.2f} сек)")
    for i, (rms, act) in enumerate(zip(ai.per_channel_rms, ai.per_channel_activity_pct), start=1):
        log.info(f"Канал {i}: RMS={rms:.4f}, активность={act:.1f}%")
    log.info("========================")


# ---------- OAuth ----------
def get_access_token(auth_basic_b64: str, scope: str = DEFAULT_SCOPE) -> str:
    """
    Получение OAuth Access Token (NGW OAuth).
    При проблемах с сертификатом: установи env NGW_SKIP_VERIFY=true (на свой риск).
    """
    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "Accept": "application/json",
        "RqUID": str(uuid.uuid4()),
        "Authorization": f"Basic {auth_basic_b64}",
    }
    data = {"scope": scope}

    verify = True
    if os.getenv("NGW_SKIP_VERIFY", "false").lower() in ("1", "true", "yes"):
        verify = False
        log.warning("ВНИМАНИЕ: SSL verify для NGW отключён (NGW_SKIP_VERIFY=true). Используй только для локальных тестов.")

    resp = requests.post(NGW_OAUTH_URL, headers=headers, data=data, timeout=30, verify=verify)
    try:
        resp.raise_for_status()
    except Exception as e:
        log.error("Ошибка OAuth запроса: %s | body=%s", e, resp.text[:500])
        raise

    payload = resp.json()
    token = payload.get("access_token") or payload.get("accessToken") or payload.get("token")
    if not token:
        raise RuntimeError(f"Не найден access_token в ответе NGW: {payload}")
    log.info("OAuth токен получен.")
    return token


# ---------- Динамическая загрузка gRPC stubs ----------
@dataclass
class GrpcBindings:
    pb2: Any
    pb2_grpc: Any
    stub_cls: Any


def load_grpc_bindings() -> GrpcBindings:
    """
    Пытаемся импортировать любые *_pb2.py и *_pb2_grpc.py, найти Stub с методами:
      Upload, AsyncRecognize, Task, Download
    """
    candidates = [
        ("smartspeech_asr_v1_pb2", "smartspeech_asr_v1_pb2_grpc"),
        ("smartspeech_pb2", "smartspeech_pb2_grpc"),
        ("sber_smartspeech_asr_v1_pb2", "sber_smartspeech_asr_v1_pb2_grpc"),
        ("salute_speech_pb2", "salute_speech_pb2_grpc"),
    ]
    last_err = None
    for m_pb2, m_pb2_grpc in candidates:
        try:
            pb2 = importlib.import_module(m_pb2)
            pb2_grpc = importlib.import_module(m_pb2_grpc)
            # Ищем подходящий Stub
            stub_cls = None
            for name in dir(pb2_grpc):
                if not name.endswith("Stub"):
                    continue
                cls = getattr(pb2_grpc, name)
                # Проверим нужные методы
                attrs = dir(cls)
                needed = {"Upload", "AsyncRecognize", "Task", "Download"}
                if needed.issubset(set(attrs)) or True:
                    # Некоторые реализации скрывают методы, просто возьмём первый Stub
                    stub_cls = cls
                    break
            if stub_cls:
                log.info(f"Используем gRPC stubs: {m_pb2_grpc}.{stub_cls.__name__}")
                return GrpcBindings(pb2=pb2, pb2_grpc=pb2_grpc, stub_cls=stub_cls)
        except Exception as e:
            last_err = e
            continue
    raise ImportError(
        f"Не удалось найти сгенерированные gRPC stubs для SmartSpeech.\n"
        f"Сгенерируй *_pb2.py и *_pb2_grpc.py из proto и положи рядом с main.py.\n"
        f"Последняя ошибка импорта: {last_err}"
    )


# ---------- gRPC: загрузка/задача/статус/скачивание ----------
async def upload_file(stub, pb2, token: str, path: str) -> Tuple[str, Optional[str]]:
    """
    Возвращает (request_file_id, x_request_id).
    """

    async def gen() -> AsyncIterator[Any]:
        # TODO(proto): если нужен "первый служебный" UploadRequest — добавь поля тут
        with open(path, "rb") as f:
            while True:
                chunk = f.read(CHUNK_SIZE)
                if not chunk:
                    break
                yield pb2.UploadRequest(file_chunk=chunk)

    metadata = [("authorization", f"Bearer {token}")]
    call = stub.Upload(gen(), metadata=metadata)
    # Пытаемся достать x-request-id (если сервер вернёт)
    try:
        initial_md = await call.initial_metadata()
        x_request_id = None
        if initial_md:
            for k, v in initial_md:
                if k.lower() == "x-request-id":
                    x_request_id = v
                    break
    except Exception:
        x_request_id = None

    resp = await call  # UploadResponse
    # TODO(proto): поле идентификатора загруженного файла
    request_file_id = getattr(resp, "request_file_id", None) or getattr(resp, "file_id", None)
    if not request_file_id:
        raise RuntimeError("Не получен request_file_id из UploadResponse. Проверь поля proto.")
    log.info(f"Файл загружен. request_file_id={request_file_id} x-request-id={x_request_id}")
    return request_file_id, x_request_id


async def create_task(stub, pb2, token: str, request_file_id: str) -> Tuple[str, Optional[str]]:
    """
    Возвращает (task_id, x-request-id).
    """
    # TODO(proto): подстрой параметры распознавания под свой proto
    req = pb2.AsyncRecognizeRequest(
        request_file_id=request_file_id,
        # Примеры доп. опций (раскомментируй по наличию в proto):
        # enable_emotions=True,
        # enable_speaker_separation=True,
        # language="ru-RU",
        # model="general",
    )
    metadata = [("authorization", f"Bearer {token}")]
    call = stub.AsyncRecognize(req, metadata=metadata)

    try:
        initial_md = await call.initial_metadata()
        x_request_id = None
        if initial_md:
            for k, v in initial_md:
                if k.lower() == "x-request-id":
                    x_request_id = v
                    break
    except Exception:
        x_request_id = None

    resp = await call
    # TODO(proto): поле идентификатора задачи
    task_id = getattr(resp, "task_id", None) or getattr(resp, "id", None)
    if not task_id:
        raise RuntimeError("Не получен task_id из ответа AsyncRecognize. Проверь поля proto.")
    log.info(f"Задача создана. task_id={task_id} x-request-id={x_request_id}")
    return task_id, x_request_id


async def poll_task(stub, pb2, token: str, task_id: str) -> Dict[str, Any]:
    """
    Ожидает завершение задачи и возвращает объект Task (dict-представление).
    """
    start = time.time()
    metadata = [("authorization", f"Bearer {token}")]

    while True:
        req = pb2.TaskRequest(task_id=task_id)  # TODO(proto): корректное имя поля
        resp = await stub.Task(req, metadata=metadata)
        # Попробуем вытащить статусы/поля
        status = getattr(resp, "status", None)
        error = getattr(resp, "error", None)
        response_file_id = getattr(resp, "response_file_id", None)

        log.info(f"Статус задачи: {status} (response_file_id={response_file_id})")
        if status in ("DONE", "ERROR", "CANCELED"):
            # Вернём как dict для удобства
            return {
                "status": status,
                "error": error,
                "response_file_id": response_file_id,
            }

        if time.time() - start > POLL_TIMEOUT:
            raise TimeoutError("Ожидание результата задачи превысило таймаут.")

        await asyncio.sleep(POLL_INTERVAL)


async def download_result(stub, pb2, token: str, response_file_id: str) -> bytes:
    """
    Скачивает результат (поток чанков) и возвращает содержимое (ожидаем JSON).
    """
    metadata = [("authorization", f"Bearer {token}")]
    req = pb2.DownloadRequest(response_file_id=response_file_id)  # TODO(proto)
    chunks = []
    async for msg in stub.Download(req, metadata=metadata):
        # TODO(proto): корректное поле для данных
        if hasattr(msg, "file_chunk") and msg.file_chunk:
            chunks.append(bytes(msg.file_chunk))
        elif hasattr(msg, "data") and msg.data:
            chunks.append(bytes(msg.data))
        else:
            # если сообщение уже как JSON-строка
            s = str(msg)
            chunks.append(s.encode("utf-8"))
    return b"".join(chunks)


# ---------- Обработка результата / метрики ----------
@dataclass
class SpeakerStats:
    durations: Dict[int, float]  # секунды речи на спикера
    count: int                   # количество спикеров (id != -1)
    main_speaker_id: Optional[int]
    main_speaker_duration: float


def compute_speaker_stats(items: List[Dict[str, Any]]) -> SpeakerStats:
    """
    items — список объектов как в примере (каждый chunk).
    Используем processed_audio_start/end или word_alignments для оценки длительности.
    """
    agg: Dict[int, float] = {}
    for it in items:
        sp = it.get("speaker_info") or {}
        sid = sp.get("speaker_id", -1)
        if sid == -1:
            continue

        dur = 0.0
        # Предпочитаем processed_audio_* (есть почти всегда)
        p0 = parse_pb_duration(it.get("processed_audio_start", "0s"))
        p1 = parse_pb_duration(it.get("processed_audio_end", "0s"))
        if p1 > p0:
            dur = p1 - p0
        else:
            # fallback: по alignments
            results = it.get("results") or []
            if results:
                r0 = results[0]
                s = parse_pb_duration(r0.get("start", "0s"))
                e = parse_pb_duration(r0.get("end", "0s"))
                if e > s:
                    dur = e - s
        agg[sid] = agg.get(sid, 0.0) + dur

    # Итоги
    main_id = None
    main_dur = 0.0
    for sid, d in agg.items():
        if d > main_dur:
            main_dur = d
            main_id = sid
    return SpeakerStats(
        durations=agg,
        count=len(agg),
        main_speaker_id=main_id,
        main_speaker_duration=main_dur,
    )


def plot_emotions_over_time(items: List[Dict[str, Any]], out_path: str) -> None:
    """
    Строим линии positive/neutral/negative от времени (берём processed_audio_end как опорную ось X).
    """
    times = []
    pos = []
    neu = []
    neg = []
    for it in items:
        t = parse_pb_duration(it.get("processed_audio_end", "0s"))
        emo = it.get("emotions_result") or {}
        times.append(t)
        pos.append(float(emo.get("positive", 0.0)))
        neu.append(float(emo.get("neutral", 0.0)))
        neg.append(float(emo.get("negative", 0.0)))

    if not times:
        log.warning("Нет точек эмоций для графика.")
        return

    plt.figure(figsize=(10, 5))
    plt.plot(times, pos, label="positive")
    plt.plot(times, neu, label="neutral")
    plt.plot(times, neg, label="negative")
    plt.xlabel("Время, сек")
    plt.ylabel("Вероятность")
    plt.title("Эмоции по времени")
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    log.info(f"График эмоций сохранён: {out_path}")


def summarize_emotions(items: List[Dict[str, Any]]) -> Tuple[float, float, float]:
    p = []
    n = []
    g = []
    for it in items:
        emo = it.get("emotions_result") or {}
        p.append(float(emo.get("positive", 0.0)))
        n.append(float(emo.get("neutral", 0.0)))
        g.append(float(emo.get("negative", 0.0)))
    if not p:
        return (0.0, 0.0, 0.0)
    return (float(np.mean(p)), float(np.mean(n)), float(np.mean(g)))


# ---------- Главный сценарий ----------
async def run(path_audio: str) -> None:
    # 1) Локальный анализ WAV
    ai = analyze_audio(path_audio)
    print_audio_info(ai)

    # 2) OAuth токен
    token = get_access_token(DEFAULT_AUTH_BASIC, DEFAULT_SCOPE)

    # 3) gRPC канал + stubs
    bindings = load_grpc_bindings()
    # TLS канал
    creds = grpc.ssl_channel_credentials()
    channel = grpc.aio.secure_channel(GRPC_ENDPOINT, creds)
    stub = bindings.stub_cls(channel)

    # 4) Upload
    req_file_id, xreq_upload = await upload_file(stub, bindings.pb2, token, path_audio)
    if xreq_upload:
        log.info(f"x-request-id (Upload): {xreq_upload}")

    # 5) Создать задачу AsyncRecognize
    task_id, xreq_task = await create_task(stub, bindings.pb2, token, req_file_id)
    log.info(f"Создана задача на распознавание: task_id={task_id}")
    if xreq_task:
        log.info(f"x-request-id (AsyncRecognize): {xreq_task}")

    # 6) Периодический опрос статуса
    task_obj = await poll_task(stub, bindings.pb2, token, task_id)
    status = task_obj["status"]
    log.info(f"Итоговый статус задачи: {status}")
    if status != "DONE":
        log.error(f"Ошибка/отмена задачи: {task_obj.get('error')}")
        return

    response_file_id = task_obj["response_file_id"]
    if not response_file_id:
        raise RuntimeError("Нет response_file_id при статусе DONE.")

    # 7) Скачать результат
    raw = await download_result(stub, bindings.pb2, token, response_file_id)

    # Ожидаем JSON в виде списка объектов (как в примере)
    try:
        items = json.loads(raw.decode("utf-8"))
    except Exception:
        # Иногда сервер может отдавать NDJSON или массивов внутри массивов
        text = raw.decode("utf-8", errors="ignore")
        # Пробуем грубо вытащить JSON-массив
        first = text.find("[")
        last = text.rfind("]")
        if first >= 0 and last > first:
            items = json.loads(text[first:last+1])
        else:
            raise RuntimeError("Не удалось распарсить JSON ответ сервера.")

    log.info(f"Получено чанков: {len(items)}")

    # 8) Спикеры
    spk = compute_speaker_stats(items)
    log.info("=== Спикеры ===")
    if not spk.durations:
        log.info("Не удалось определить спикеров.")
    else:
        for sid, dur in sorted(spk.durations.items()):
            log.info(f"Спикер {sid}: {dur:.2f} сек")
        log.info(f"Всего спикеров: {spk.count}")
        if spk.main_speaker_id is not None:
            log.info(f"Главный спикер: {spk.main_speaker_id} "
                     f"(говорил дольше всех: {spk.main_speaker_duration:.2f} сек)")

    # 9) Эмоции: сводные и график
    p, n, g = summarize_emotions(items)
    log.info("=== Эмоции (средние вероятности) ===")
    log.info(f"Positive: {p:.3f} | Neutral: {n:.3f} | Negative: {g:.3f}")

    out_plot = os.path.join("output", "emotions_timeline.png")
    plot_emotions_over_time(items, out_plot)

    # 10) Финальные служебные ID (если были)
    log.info("Готово.")


def main():
    if len(sys.argv) < 2:
        print("Использование: python main.py path/to/file")
        sys.exit(1)
    path = sys.argv[1]
    asyncio.run(run(path))


if __name__ == "__main__":
    main()
