"""Низкоуровневый разбор структуры PNG (ISO/IEC 15948) и поиск структурных аномалий,
характерных для сокрытия данных вне пиксельной области."""
from __future__ import annotations

import re
import struct
import zlib
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"

KNOWN_CHUNKS = {
    b"IHDR", b"PLTE", b"IDAT", b"IEND", b"tRNS", b"cHRM", b"gAMA", b"iCCP", b"sBIT",
    b"sRGB", b"cICP", b"mDCV", b"mDCv", b"cLLI", b"cLLi", b"tEXt", b"zTXt", b"iTXt",
    b"bKGD", b"hIST", b"pHYs", b"sPLT", b"eXIf", b"tIME", b"acTL", b"fcTL", b"fdAT",
    b"oFFs", b"pCAL", b"sCAL", b"gIFg", b"gIFx", b"gIFt", b"sTER", b"dSIG", b"vpAg",
    b"caBX", b"iDOT", b"CgBI",
}
TEXT_CHUNKS = {b"tEXt", b"zTXt", b"iTXt"}
CRITICAL_UNIQUE = {b"IHDR", b"PLTE", b"IEND"}
CHANNELS_BY_COLOR_TYPE = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}
ALLOWED_BIT_DEPTHS = {0: {1, 2, 4, 8, 16}, 2: {8, 16}, 3: {1, 2, 4, 8}, 4: {8, 16}, 6: {8, 16}}

FILE_SIGNATURES: List[Tuple[bytes, str]] = [
    (b"PK\x03\x04", "ZIP/Office-архив"),
    (b"%PDF-", "PDF-документ"),
    (b"\x7fELF", "исполняемый файл ELF"),
    (b"Rar!\x1a\x07", "RAR-архив"),
    (b"7z\xbc\xaf\x27\x1c", "7-Zip-архив"),
    (b"\xff\xd8\xff", "JPEG-изображение"),
    (b"\x89PNG\r\n\x1a\n", "вложенное PNG-изображение"),
    (b"GIF89a", "GIF-изображение"),
    (b"GIF87a", "GIF-изображение"),
    (b"\x1f\x8b\x08", "GZIP-поток"),
    (b"BZh91AY&SY", "BZIP2-поток"),
    (b"#!/", "сценарий командной оболочки"),
    (b"<?php", "PHP-сценарий"),
    (b"<script", "JavaScript/HTML-сценарий"),
    (b"powershell", "команда PowerShell"),
    (b"This program cannot be run in DOS mode", "исполняемый файл Windows PE"),
    (b"-----BEGIN", "PEM-ключ/сертификат"),
    (b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1", "документ MS Office (OLE)"),
]

SEVERITY_WEIGHTS = {"info": 0, "low": 10, "medium": 35, "high": 65, "critical": 90}
SEVERITY_LABELS = {"info": "Информация", "low": "Низкая", "medium": "Средняя",
                   "high": "Высокая", "critical": "Критическая"}


@dataclass
class Chunk:
    type: str
    offset: int
    length: int
    crc: int
    crc_ok: bool
    data: bytes = field(repr=False, default=b"")

    def to_dict(self) -> Dict:
        return {"type": self.type, "offset": self.offset, "length": self.length,
                "crc": f"{self.crc:08X}", "crc_ok": self.crc_ok,
                "critical": self.type[:1].isupper(), "known": self.type.encode("latin1") in KNOWN_CHUNKS}


@dataclass
class Finding:
    code: str
    severity: str
    title: str
    details: str = ""
    offset: Optional[int] = None

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["severity_label"] = SEVERITY_LABELS[self.severity]
        return d


@dataclass
class PNGStructure:
    valid_signature: bool = False
    file_size: int = 0
    chunks: List[Chunk] = field(default_factory=list)
    findings: List[Finding] = field(default_factory=list)
    ihdr: Dict = field(default_factory=dict)
    trailing_bytes: int = 0
    idat_count: int = 0
    idat_total: int = 0
    decompressed_size: Optional[int] = None
    expected_size: Optional[int] = None
    text_entries: List[Dict] = field(default_factory=list)

    def add(self, code: str, severity: str, title: str, details: str = "", offset: Optional[int] = None):
        self.findings.append(Finding(code, severity, title, details, offset))

    @property
    def max_severity(self) -> str:
        if not self.findings:
            return "info"
        return max(self.findings, key=lambda f: SEVERITY_WEIGHTS[f.severity]).severity

    @property
    def structural_risk(self) -> float:
        return float(max((SEVERITY_WEIGHTS[f.severity] for f in self.findings), default=0))

    def to_dict(self) -> Dict:
        return {
            "valid_signature": self.valid_signature,
            "file_size": self.file_size,
            "ihdr": self.ihdr,
            "chunks": [c.to_dict() for c in self.chunks],
            "chunk_summary": _chunk_summary(self.chunks),
            "findings": [f.to_dict() for f in self.findings],
            "trailing_bytes": self.trailing_bytes,
            "idat_count": self.idat_count,
            "idat_total": self.idat_total,
            "decompressed_size": self.decompressed_size,
            "expected_size": self.expected_size,
            "text_entries": self.text_entries,
            "structural_risk": self.structural_risk,
            "max_severity": self.max_severity,
        }


def _chunk_summary(chunks: List[Chunk]) -> Dict[str, Dict[str, int]]:
    summary: Dict[str, Dict[str, int]] = {}
    for c in chunks:
        s = summary.setdefault(c.type, {"count": 0, "bytes": 0})
        s["count"] += 1
        s["bytes"] += c.length
    return summary


def shannon_entropy(data: bytes) -> float:
    if not data:
        return 0.0
    counts = np.bincount(np.frombuffer(data, dtype=np.uint8), minlength=256).astype(np.float64)
    p = counts[counts > 0] / len(data)
    return float(-(p * np.log2(p)).sum())


def scan_signatures(data: bytes, limit: int = 20) -> List[Tuple[int, str]]:
    """Поиск сигнатур известных форматов файлов/сценариев внутри блока данных."""
    found: List[Tuple[int, str]] = []
    lower = data.lower()
    for sig, name in FILE_SIGNATURES:
        textual = all(32 <= b < 127 for b in sig) and any(chr(b).isalpha() for b in sig)
        hay, needle = (lower, sig.lower()) if textual else (data, sig)
        start = 0
        while len(found) < limit:
            idx = hay.find(needle, start)
            if idx < 0:
                break
            found.append((idx, name))
            start = idx + 1
    if data[:2] == b"MZ":
        found.append((0, "исполняемый файл Windows (MZ)"))
    return sorted(set(found))[:limit]


_B64_RE = re.compile(rb"^[A-Za-z0-9+/=\r\n]+$")
_HEX_RE = re.compile(rb"^[0-9A-Fa-f\r\n ]+$")


def classify_blob(data: bytes) -> List[str]:
    """Признаки закодированных/зашифрованных данных в текстовом блоке."""
    notes = []
    if len(data) >= 64 and _B64_RE.match(data):
        notes.append("похоже на Base64")
    if len(data) >= 64 and _HEX_RE.match(data):
        notes.append("похоже на шестнадцатеричный дамп")
    if len(data) >= 256 and shannon_entropy(data) > 7.2:
        notes.append("высокая энтропия (сжатые/зашифрованные данные)")
    return notes


def expected_raw_size(width: int, height: int, bit_depth: int, color_type: int, interlace: int) -> int:
    ch = CHANNELS_BY_COLOR_TYPE.get(color_type, 1)
    bits = ch * bit_depth

    def rowbytes(w: int) -> int:
        return (w * bits + 7) // 8

    if interlace == 0:
        return height * (1 + rowbytes(width)) if width and height else 0
    total = 0
    for x0, y0, dx, dy in ((0, 0, 8, 8), (4, 0, 8, 8), (0, 4, 4, 8), (2, 0, 4, 4),
                           (0, 2, 2, 4), (1, 0, 2, 2), (0, 1, 1, 2)):
        pw = (width - x0 + dx - 1) // dx if width > x0 else 0
        ph = (height - y0 + dy - 1) // dy if height > y0 else 0
        if pw and ph:
            total += ph * (1 + rowbytes(pw))
    return total


def _decode_text_chunk(ctype: bytes, data: bytes) -> Tuple[str, bytes, Optional[str]]:
    """Возвращает (ключ, сырые байты текста, ошибка)."""
    try:
        key, _, rest = data.partition(b"\x00")
        keyword = key.decode("latin1", "replace")
        if ctype == b"tEXt":
            return keyword, rest, None
        if ctype == b"zTXt":
            if not rest:
                return keyword, b"", "пустой zTXt"
            return keyword, zlib.decompress(rest[1:]), None
        # iTXt: flag, method, lang\0, translated\0, text
        flag, method = rest[0], rest[1]
        rest = rest[2:]
        _lang, _, rest = rest.partition(b"\x00")
        _tr, _, text = rest.partition(b"\x00")
        if flag == 1:
            text = zlib.decompress(text)
        return keyword, text, None
    except Exception as exc:  # noqa: BLE001
        return "?", b"", f"ошибка декодирования: {exc}"


def parse_png(src, thresholds=None) -> PNGStructure:
    """Разбор PNG-файла (путь или bytes) с выявлением структурных аномалий."""
    from .config import Thresholds

    thr = thresholds or Thresholds()
    if isinstance(src, (bytes, bytearray)):
        data = bytes(src)
    else:
        with open(src, "rb") as fh:
            data = fh.read()

    st = PNGStructure(file_size=len(data))
    if not data.startswith(PNG_SIGNATURE):
        st.add("NOT_PNG", "critical", "Файл не является PNG",
               "Сигнатура PNG отсутствует. Возможна подмена типа файла.", 0)
        sigs = scan_signatures(data[:4096], limit=3)
        if sigs:
            st.findings[-1].details += " Обнаружено: " + ", ".join(n for _, n in sigs)
        return st
    st.valid_signature = True

    pos = 8
    iend = False
    while pos + 8 <= len(data):
        length = struct.unpack(">I", data[pos:pos + 4])[0]
        ctype = data[pos + 4:pos + 8]
        if not all(65 <= b <= 90 or 97 <= b <= 122 for b in ctype):
            st.add("CORRUPT_CHUNK", "high", "Повреждённый заголовок чанка",
                   f"Недопустимый тип чанка {ctype!r}", pos)
            break
        if length > 0x7FFFFFFF or pos + 12 + length > len(data):
            st.add("TRUNCATED", "high", "Усечённый чанк",
                   f"Чанк {ctype.decode('latin1')} объявляет {length} байт, выходящих за конец файла", pos)
            break
        cdata = data[pos + 8:pos + 8 + length]
        crc = struct.unpack(">I", data[pos + 8 + length:pos + 12 + length])[0]
        ok = (zlib.crc32(ctype + cdata) & 0xFFFFFFFF) == crc
        st.chunks.append(Chunk(ctype.decode("latin1"), pos, length, crc, ok, cdata))
        pos += 12 + length
        if ctype == b"IEND":
            iend = True
            break

    trailing = data[pos:] if iend else b""
    st.trailing_bytes = len(trailing)
    _check_order(st, iend)
    _check_crc(st)
    _check_ihdr_plte(st)
    _check_unknown_and_sizes(st, thr)
    _check_text(st, thr)
    _check_idat(st)
    _check_trailing(st, trailing, pos, thr)
    return st


def _check_order(st: PNGStructure, iend: bool) -> None:
    types = [c.type for c in st.chunks]
    if not types or types[0] != "IHDR":
        st.add("NO_IHDR_FIRST", "high", "Первый чанк не IHDR", "Нарушение спецификации PNG")
    if not iend:
        st.add("NO_IEND", "high", "Отсутствует чанк IEND", "Файл повреждён или усечён")
    for t in CRITICAL_UNIQUE:
        n = types.count(t.decode())
        if n > 1:
            st.add("DUP_CRITICAL", "high", f"Дублирование критического чанка {t.decode()}",
                   f"Найдено {n} экземпляров")
    idat_idx = [i for i, t in enumerate(types) if t == "IDAT"]
    st.idat_count = len(idat_idx)
    if not idat_idx:
        st.add("NO_IDAT", "high", "Отсутствуют данные изображения (IDAT)")
    elif idat_idx[-1] - idat_idx[0] + 1 != len(idat_idx):
        st.add("IDAT_GAP", "high", "Чанки IDAT разделены другими чанками",
               "Нарушение спецификации: между IDAT размещены посторонние чанки")


def _check_crc(st: PNGStructure) -> None:
    for c in st.chunks:
        if not c.crc_ok:
            st.add("CRC_MISMATCH", "high", f"Ошибка контрольной суммы чанка {c.type}",
                   "CRC32 не совпадает: содержимое изменено после записи", c.offset)


def _check_ihdr_plte(st: PNGStructure) -> None:
    ihdr = next((c for c in st.chunks if c.type == "IHDR"), None)
    if ihdr is None or ihdr.length != 13:
        if ihdr is not None:
            st.add("BAD_IHDR", "high", "Некорректная длина IHDR", f"{ihdr.length} байт вместо 13")
        return
    w, h, bd, ct, comp, flt, il = struct.unpack(">IIBBBBB", ihdr.data)
    st.ihdr = {"width": w, "height": h, "bit_depth": bd, "color_type": ct,
               "compression": comp, "filter": flt, "interlace": il}
    if ct not in ALLOWED_BIT_DEPTHS or bd not in ALLOWED_BIT_DEPTHS[ct]:
        st.add("BAD_IHDR", "high", "Недопустимое сочетание типа цвета и глубины",
               f"color_type={ct}, bit_depth={bd}")
    if comp != 0 or flt != 0 or il not in (0, 1):
        st.add("BAD_IHDR", "medium", "Нестандартные параметры IHDR",
               f"compression={comp}, filter={flt}, interlace={il}")
    if w == 0 or h == 0 or w > 100000 or h > 100000:
        st.add("BAD_DIMENSIONS", "high", "Аномальные размеры изображения", f"{w}x{h}")

    plte = next((c for c in st.chunks if c.type == "PLTE"), None)
    if plte is not None:
        if plte.length % 3 or plte.length == 0 or plte.length > 768:
            st.add("BAD_PLTE", "high", "Некорректная длина палитры", f"{plte.length} байт")
        else:
            entries = [plte.data[i:i + 3] for i in range(0, plte.length, 3)]
            if ct == 3 and len(entries) > (1 << bd):
                st.add("PLTE_OVERSIZE", "medium", "Палитра больше допустимой для глубины цвета",
                       f"{len(entries)} записей при глубине {bd} бит")
            dup = len(entries) - len(set(entries))
            if dup > 0 and ct == 3:
                st.add("PLTE_DUPLICATES", "medium", "Дублирующиеся цвета в палитре",
                       f"{dup} повторов. Признак палитровой стеганографии (EzStego-подобные методы)")
    elif ct == 3:
        st.add("NO_PLTE", "high", "Отсутствует обязательная палитра")


def _check_unknown_and_sizes(st: PNGStructure, thr) -> None:
    for c in st.chunks:
        raw = c.type.encode("latin1")
        if raw not in KNOWN_CHUNKS:
            critical = c.type[0].isupper()
            private = c.type[1].islower()
            sev = "high" if (critical or c.length > 1024) else "medium"
            st.add("UNKNOWN_CHUNK", sev, f"Неизвестный чанк {c.type}",
                   f"{c.length} байт; {'критический' if critical else 'вспомогательный'}, "
                   f"{'частный' if private else 'публичный'}; энтропия {shannon_entropy(c.data):.2f} бит/байт",
                   c.offset)
            sigs = scan_signatures(c.data, limit=3)
            if sigs:
                st.add("EMBEDDED_FILE", "critical", f"Сигнатура файла внутри чанка {c.type}",
                       ", ".join(f"{n} (+{o})" for o, n in sigs), c.offset)
        elif raw not in (b"IDAT", b"iCCP", b"eXIf") and c.length > thr.ancillary_max_bytes:
            st.add("LARGE_ANCILLARY", "medium", f"Аномально большой чанк {c.type}",
                   f"{c.length} байт", c.offset)
        elif raw == b"eXIf":
            st.add("EXIF_PRESENT", "info", "Присутствуют метаданные EXIF", f"{c.length} байт", c.offset)
        elif raw in (b"acTL", b"fcTL", b"fdAT") and c.type == "acTL":
            st.add("APNG", "low", "Анимированный PNG (APNG)",
                   "Кадры кроме первого не отображаются обычными просмотрщиками", c.offset)


def _check_text(st: PNGStructure, thr) -> None:
    for c in st.chunks:
        raw = c.type.encode("latin1")
        if raw not in TEXT_CHUNKS:
            continue
        key, text, err = _decode_text_chunk(raw, c.data)
        preview = text[:120].decode("utf-8", "replace")
        entry = {"chunk": c.type, "keyword": key, "length": len(text),
                 "entropy": round(shannon_entropy(text), 3), "preview": preview}
        st.text_entries.append(entry)
        if err:
            st.add("TEXT_DECODE_ERROR", "medium", f"Ошибка декодирования {c.type}", err, c.offset)
            continue
        notes = classify_blob(text)
        sigs = scan_signatures(text, limit=3)
        if sigs:
            st.add("EMBEDDED_FILE", "critical", f"Сигнатура файла в текстовом чанке «{key}»",
                   ", ".join(f"{n} (+{o})" for o, n in sigs), c.offset)
        if notes:
            st.add("ENCODED_TEXT", "high" if len(text) > thr.text_chunk_max_len else "medium",
                   f"Закодированные данные в текстовом чанке «{key}»",
                   f"{len(text)} байт: " + "; ".join(notes), c.offset)
        elif len(text) > thr.text_chunk_max_len:
            st.add("LONG_TEXT", "medium", f"Длинный текстовый чанк «{key}»", f"{len(text)} байт", c.offset)
        else:
            st.add("TEXT_METADATA", "info", f"Текстовые метаданные «{key}»", preview[:60], c.offset)


def _check_idat(st: PNGStructure) -> None:
    idats = [c for c in st.chunks if c.type == "IDAT"]
    if not idats:
        return
    stream = b"".join(c.data for c in idats)
    st.idat_total = len(stream)
    try:
        d = zlib.decompressobj()
        raw = d.decompress(stream)
        raw += d.flush()
        extra = d.unused_data
        if not d.eof:
            st.add("ZLIB_TRUNCATED", "high", "Незавершённый zlib-поток IDAT")
    except zlib.error as exc:
        st.add("ZLIB_ERROR", "high", "Ошибка распаковки IDAT", str(exc), idats[0].offset)
        return
    st.decompressed_size = len(raw)
    if extra:
        st.add("ZLIB_TRAILING", "critical", "Скрытые данные после конца zlib-потока в IDAT",
               f"{len(extra)} байт, энтропия {shannon_entropy(extra):.2f} бит/байт", idats[-1].offset)
    if st.ihdr:
        exp = expected_raw_size(st.ihdr["width"], st.ihdr["height"], st.ihdr["bit_depth"],
                                st.ihdr["color_type"], st.ihdr["interlace"])
        st.expected_size = exp
        if len(raw) > exp:
            st.add("IDAT_OVERSIZE", "critical", "Избыточные данные в распакованном потоке IDAT",
                   f"{len(raw) - exp} байт сверх необходимого ({exp})")
        elif len(raw) < exp:
            st.add("IDAT_UNDERSIZE", "high", "Недостаточно данных изображения",
                   f"Получено {len(raw)} из {exp} байт")
        elif st.ihdr["interlace"] == 0 and exp:
            ch = CHANNELS_BY_COLOR_TYPE.get(st.ihdr["color_type"], 1)
            rb = (st.ihdr["width"] * ch * st.ihdr["bit_depth"] + 7) // 8
            filters = np.frombuffer(raw, dtype=np.uint8)[::rb + 1][:st.ihdr["height"]]
            bad = int(np.count_nonzero(filters > 4))
            if bad:
                st.add("BAD_FILTER", "high", "Недопустимые типы фильтров строк",
                       f"{bad} строк с типом фильтра > 4")


def _check_trailing(st: PNGStructure, trailing: bytes, offset: int, thr) -> None:
    if not trailing:
        return
    if len(trailing) <= 16 and not trailing.strip(b"\x00"):
        st.add("TRAILING_PADDING", "low", "Нулевое выравнивание после IEND", f"{len(trailing)} байт", offset)
        return
    ent = shannon_entropy(trailing)
    sigs = scan_signatures(trailing, limit=5)
    details = f"{len(trailing)} байт, энтропия {ent:.2f} бит/байт"
    if sigs:
        details += "; сигнатуры: " + ", ".join(f"{n} (+{o})" for o, n in sigs)
    st.add("TRAILING_DATA", "critical", "Данные после чанка IEND", details, offset)
