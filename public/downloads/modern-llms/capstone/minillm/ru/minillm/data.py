"""Данные для капстоуна: токенизатор-заглушка, упаковка и детерминированные батчи.

Здесь намеренно нет зависимостей. Байтовый токенизатор — не рабочий выбор для
курса (модуль 2 объясняет, почему byte-level без обученных merge'ей неприемлемо
длинен), а способ запустить обучение и тесты в любом окружении. Ноутбук
претрейна подменяет его на BBPE, обученный в модуле 2, ничего не меняя дальше
по конвейеру: контракт один — `encode(str) -> list[int]`.

Контракт форм: ``seq_len`` означает число обучаемых переходов. Поток отдаёт
окно длины ``seq_len + 1``; MiniLLM сдвигает его внутри loss и тем самым обучает
все переходы, включая границу между соседними окнами.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


class ByteTokenizer:
    """UTF-8 байты как идентификаторы. Словарь фиксирован: 256 + служебные."""

    BOS, EOS, PAD = 256, 257, 258
    vocab_size = 259

    def encode(self, text: str, add_special: bool = False) -> list[int]:
        ids = list(text.encode("utf-8"))
        return [self.BOS, *ids, self.EOS] if add_special else ids

    def decode(self, ids: list[int]) -> str:
        payload = bytes(i for i in ids if i < 256)
        return payload.decode("utf-8", errors="replace")


def toy_corpus(repeats: int = 400) -> str:
    """Маленький корпус с выучиваемой структурой.

    Смысл — не язык, а проверяемость: у текста низкая энтропия, поэтому уже
    несколько сотен шагов на CPU обязаны увести лосс заметно ниже равномерного
    ln(V). Если этого не происходит, сломан цикл обучения, а не «мало данных».
    """
    templates = [
        "коты спят на окне, а собаки спят на полу.\n",
        "утром светло, вечером темно, ночью совсем темно.\n",
        "один плюс один равно два, два плюс два равно четыре.\n",
        "the cat sleeps on the mat and the dog sleeps on the floor.\n",
        "one plus one is two, two plus two is four.\n",
    ]
    return "".join(templates[i % len(templates)] for i in range(repeats * len(templates)))


@dataclass
class TokenStream:
    """Плоский поток токенов + детерминированная нарезка на окна."""

    tokens: torch.Tensor          # [T_total], dtype=long
    seq_len: int

    @classmethod
    def from_text(cls, text: str, tokenizer: ByteTokenizer, seq_len: int) -> "TokenStream":
        ids = tokenizer.encode(text)
        if len(ids) < seq_len + 1:
            raise ValueError(f"корпус короче окна: {len(ids)} < {seq_len + 1}")
        return cls(torch.tensor(ids, dtype=torch.long), seq_len)

    def __len__(self) -> int:
        return (self.tokens.numel() - 1) // self.seq_len

    def checksum(self) -> str:
        """SHA-256 потока: контракт данных из NOTEBOOK_POLICY §3."""
        return hashlib.sha256(self.tokens.numpy().tobytes()).hexdigest()[:16]

    def batch(self, indices: torch.Tensor) -> torch.Tensor:
        """indices [B] -> windows [B, seq_len+1] containing seq_len targets."""
        starts = indices * self.seq_len
        offsets = torch.arange(self.seq_len + 1)
        return self.tokens[starts[:, None] + offsets[None, :]]

    def loader(self, batch_size: int, seed: int = 0, shuffle: bool = True,
               start_batch: int = 0):
        """Бесконечный детерминированный генератор батчей.

        `start_batch` позволяет продолжить с произвольной позиции, не читая
        предыдущие батчи: перестановка каждой эпохи выводится из (seed, epoch),
        поэтому позиция полностью определяется глобальным номером батча. Это и
        есть «курсор» из NOTEBOOK_POLICY §3 — забег, прерванный на середине,
        обязан продолжить с того же места в данных.
        """
        n = len(self)
        per_epoch = n // batch_size
        if per_epoch == 0:
            raise ValueError(f"batch_size={batch_size} больше числа окон {n}")
        global_batch = start_batch
        while True:
            epoch, within = divmod(global_batch, per_epoch)
            if shuffle:
                gen = torch.Generator().manual_seed(seed * 1_000_003 + epoch)
                order = torch.randperm(n, generator=gen)
            else:
                order = torch.arange(n)
            for i in range(within, per_epoch):
                yield self.batch(order[i * batch_size:(i + 1) * batch_size])
                global_batch += 1


# =============================================================================
# Шардированный поток для профиля L
# =============================================================================
def write_shards(tokens, out_dir: str | Path, shard_tokens: int = 1_000_000,
                 dtype: str = "uint16") -> dict:
    """Пишет плоский поток токенов в .bin-шарды и index.json с контрольными суммами.

    Зачем шарды. Корпус профиля L (миллиарды токенов) в память не поместится, а
    держать его одним файлом неудобно для докачки и проверки целостности.
    uint16 хватает при словаре до 65 536 — это ровно половина памяти против
    uint32, и на 20B токенов разница составляет 40 GB на диске.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    target_dtype = np.dtype(dtype)
    if not np.issubdtype(target_dtype, np.integer):
        raise ValueError(f"dtype шардов должен быть целочисленным, получен {dtype}")
    source = np.asarray(tokens)
    if source.ndim != 1 or source.size == 0:
        raise ValueError("tokens должен быть непустым одномерным массивом")
    if not np.issubdtype(source.dtype, np.integer):
        raise ValueError(f"идентификаторы должны быть целыми, получен {source.dtype}")
    limits = np.iinfo(target_dtype)
    smallest, largest = int(source.min()), int(source.max())
    # Диапазон проверяется ДО приведения dtype. Иначе NumPy-массив с ID 70000
    # тихо превращал его в 4464 при записи uint16.
    if smallest < limits.min or largest > limits.max:
        raise ValueError(f"идентификаторы не помещаются в {dtype}")
    array = source.astype(target_dtype, copy=False)

    shards = []
    for i, start in enumerate(range(0, array.size, shard_tokens)):
        chunk = array[start:start + shard_tokens]
        name = f"shard_{i:05d}.bin"
        (out_dir / name).write_bytes(chunk.tobytes())
        shards.append({
            "name": name,
            "tokens": int(chunk.size),
            "sha256": hashlib.sha256(chunk.tobytes()).hexdigest()[:32],
        })
    index = {"dtype": dtype, "total_tokens": int(array.size), "shards": shards}
    (out_dir / "index.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return index


class ShardStream:
    """Ленивое чтение шардов через memmap с тем же интерфейсом, что TokenStream.

    Данные не грузятся в память: numpy отображает файл, а страницы подтягивает
    операционная система. Проверка целостности по SHA-256 — опциональная и
    выключена по умолчанию, потому что на десятках гигабайт она стоит полного
    прохода по диску.
    """

    def __init__(self, directory: str | Path, seq_len: int, verify: bool = False) -> None:
        self.directory = Path(directory)
        self.seq_len = seq_len
        self.index = json.loads((self.directory / "index.json").read_text(encoding="utf-8"))
        self.dtype = np.dtype(self.index["dtype"])
        self.maps, offsets, total = [], [], 0
        for shard in self.index["shards"]:
            path = self.directory / shard["name"]
            data = np.memmap(path, dtype=self.dtype, mode="r")
            if verify:
                digest = hashlib.sha256(data.tobytes()).hexdigest()[:32]
                if digest != shard["sha256"]:
                    raise RuntimeError(f"контрольная сумма шарда {shard['name']} не сошлась")
            self.maps.append(data)
            offsets.append(total)
            total += data.size
        self.offsets = np.array(offsets)
        self.total_tokens = total

    def __len__(self) -> int:
        return (self.total_tokens - 1) // self.seq_len

    def close(self) -> None:
        """Освободить файловые отображения, в том числе на Windows.

        На Unix временный каталог обычно можно удалить и при живом memmap, а
        Windows держит файл заблокированным до закрытия mapping. Явный lifecycle
        делает поведение одинаковым и не оставляет очистку на сборщик мусора.
        """
        for data in self.maps:
            mapping = getattr(data, "_mmap", None)
            if mapping is not None:
                mapping.close()
        self.maps.clear()

    def __enter__(self) -> "ShardStream":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            # Деструктор не должен маскировать исходную ошибку интерпретатора.
            pass

    def checksum(self) -> str:
        return hashlib.sha256("".join(s["sha256"] for s in self.index["shards"])
                              .encode()).hexdigest()[:16]

    def _read(self, start: int, length: int) -> np.ndarray:
        """Чтение окна, возможно пересекающего границу шардов."""
        out = np.empty(length, dtype=self.dtype)
        written = 0
        shard = int(np.searchsorted(self.offsets, start, side="right") - 1)
        position = start - self.offsets[shard]
        while written < length:
            data = self.maps[shard]
            take = min(length - written, data.size - position)
            out[written:written + take] = data[position:position + take]
            written += take
            shard += 1
            position = 0
            if shard >= len(self.maps) and written < length:
                raise IndexError("окно выходит за конец потока")
        return out

    def batch(self, indices: torch.Tensor) -> torch.Tensor:
        rows = [self._read(int(i) * self.seq_len, self.seq_len + 1) for i in indices]
        return torch.from_numpy(np.stack(rows).astype(np.int64))

    loader = TokenStream.loader
