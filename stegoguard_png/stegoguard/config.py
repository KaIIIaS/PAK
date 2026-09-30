"""Конфигурация комплекса: пороги принятия решений и параметры оценки стойкости."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class Thresholds:
    # Оценка доли заполнения контейнера (бит на отсчёт), выше которой изображение подозрительно
    payload_suspicious: float = 0.05
    # ... и выше которой фиксируется обнаружение вложения
    payload_stego: float = 0.10
    # p-значение атаки хи-квадрат, выше которого фиксируется последовательное LSB-вложение
    chi_square_p: float = 0.95
    # Интегральный риск (0..100)
    risk_suspicious: float = 30.0
    risk_stego: float = 60.0
    # Структурные эвристики
    text_chunk_max_len: int = 2048
    ancillary_max_bytes: int = 65536
    high_entropy: float = 7.2


@dataclass
class RobustnessSettings:
    rates: List[float] = field(default_factory=lambda: [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0])
    methods: List[str] = field(default_factory=lambda: ["lsb_replacement", "lsb_matching"])
    trials: int = 2
    max_side: int = 512
    detection_threshold: float = 0.5


@dataclass
class Config:
    thresholds: Thresholds = field(default_factory=Thresholds)
    robustness: RobustnessSettings = field(default_factory=RobustnessSettings)
    analysis_max_pixels: int = 4_000_000
    seed: int = 20240901
    include_alpha: bool = False
    db_path: str = "stegoguard_history.db"
    log_level: str = "INFO"
    log_file: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Config":
        cfg = cls()
        for key, value in (data or {}).items():
            if key == "thresholds":
                cfg.thresholds = Thresholds(**{**asdict(cfg.thresholds), **value})
            elif key == "robustness":
                cfg.robustness = RobustnessSettings(**{**asdict(cfg.robustness), **value})
            elif hasattr(cfg, key):
                setattr(cfg, key, value)
            else:
                raise ValueError(f"Неизвестный параметр конфигурации: {key}")
        cfg.validate()
        return cfg

    def validate(self) -> None:
        t = self.thresholds
        if not 0 <= t.payload_suspicious <= t.payload_stego <= 1:
            raise ValueError("Требуется 0 <= payload_suspicious <= payload_stego <= 1")
        if not 0 <= t.risk_suspicious <= t.risk_stego <= 100:
            raise ValueError("Требуется 0 <= risk_suspicious <= risk_stego <= 100")
        r = self.robustness
        if not r.rates or any(not 0 < x <= 1 for x in r.rates):
            raise ValueError("Доли заполнения robustness.rates должны лежать в (0, 1]")
        unknown = set(r.methods) - {"lsb_replacement", "lsb_matching"}
        if unknown:
            raise ValueError(f"Неизвестные методы встраивания: {sorted(unknown)}")
        if r.trials < 1 or r.max_side < 32:
            raise ValueError("robustness.trials >= 1 и robustness.max_side >= 32")

    @classmethod
    def load(cls, path: Optional[str] = None) -> "Config":
        if not path:
            return cls()
        with open(path, "r", encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, ensure_ascii=False, indent=2)
