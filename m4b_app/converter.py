import os
import re
import json
import subprocess
import tempfile
import shutil
import wave
from pathlib import Path
from typing import Any, List, Tuple, Callable, Optional

# Supported model types
MODEL_STREAMING = 'vibevoice_streaming'   # VibeVoice-Realtime-0.5B  — supported
MODEL_FULL      = 'vibevoice'             # VibeVoice-1.5B / Large   — in progress

PARALLEL_CHUNKS = 2   # concurrent chunk workers; set to 1 to disable

SAMPLE_RATE     = 24000
MAX_CHUNK_CHARS = 3000  # streaming model needs enough text for its windowed attention
MIN_CHUNK_CHARS = 300   # below this the windowed attention breaks — skip with warning


def detect_model_type(model_path: str) -> str:
    """Read model_type from config.json. Raises if not a VibeVoice model."""
    cfg_path = os.path.join(model_path, 'config.json')
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f'config.json not found in {model_path}')
    with open(cfg_path, 'r', encoding='utf-8') as f:
        cfg = json.load(f)
    model_type = cfg.get('model_type', '')
    if model_type not in (MODEL_STREAMING, MODEL_FULL):
        raise ValueError(f'Unrecognised model_type "{model_type}" in config.json')
    return model_type


try:
    import ebooklib
    from ebooklib import epub
    from bs4 import BeautifulSoup
    EPUB_AVAILABLE = True
except ImportError:
    EPUB_AVAILABLE = False


# ---------------------------------------------------------------------------
# Text parsing
# ---------------------------------------------------------------------------

def parse_epub(path: str) -> List[Tuple[str, str]]:
    if not EPUB_AVAILABLE:
        raise ImportError(
            "ebooklib and beautifulsoup4 required for EPUB support.\n"
            "Run: pip install ebooklib beautifulsoup4 lxml"
        )
    book = epub.read_epub(path)
    chapters = []

    for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        soup = BeautifulSoup(item.get_content(), features='xml')

        title = None
        for tag in ['h1', 'h2', 'h3']:
            heading = soup.find(tag)
            if heading:
                title = heading.get_text(strip=True)
                break

        text = soup.get_text(separator='\n', strip=True)
        text = re.sub(r'\n{3,}', '\n\n', text)

        if len(text.strip()) < 100:
            continue

        if not title:
            title = f"Chapter {len(chapters) + 1}"

        chapters.append((title, text.strip()))

    if not chapters:
        parts = []
        for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
            soup = BeautifulSoup(item.get_content(), features='xml')
            parts.append(soup.get_text(separator='\n', strip=True))
        chapters = [(Path(path).stem, '\n\n'.join(parts))]

    return chapters


def parse_txt(path: str) -> List[Tuple[str, str]]:
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        text = f.read()

    pattern = re.compile(
        r'^(chapter\s+\w+[:\.\s\-]*.{0,60}|part\s+\w+[:\.\s\-]*.{0,60})',
        re.IGNORECASE | re.MULTILINE
    )
    matches = list(pattern.finditer(text))

    if len(matches) >= 2:
        chapters = []
        for i, m in enumerate(matches):
            chap_title = m.group().strip()
            start = m.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            body = text[start:end].strip()
            if body:
                chapters.append((chap_title, body))
        if chapters:
            return chapters

    return [(Path(path).stem, text.strip())]


# ---------------------------------------------------------------------------
# Text chunking
# ---------------------------------------------------------------------------

def chunk_text(text: str, max_chars: int = MAX_CHUNK_CHARS) -> List[str]:
    """Split text at paragraph/sentence boundaries within max_chars."""
    if len(text) <= max_chars:
        return [text]

    paragraphs = [p.strip() for p in re.split(r'\n{2,}', text) if p.strip()]
    chunks: List[str] = []
    current: List[str] = []
    current_len = 0

    for para in paragraphs:
        if len(para) > max_chars:
            if current:
                chunks.append('\n\n'.join(current))
                current, current_len = [], 0
            sentences = re.split(r'(?<=[.!?])\s+', para)
            sub: List[str] = []
            sub_len = 0
            for sent in sentences:
                if sub_len + len(sent) + 1 > max_chars and sub:
                    chunks.append(' '.join(sub))
                    sub, sub_len = [sent], len(sent)
                else:
                    sub.append(sent)
                    sub_len += len(sent) + 1
            if sub:
                chunks.append(' '.join(sub))
        elif current_len + len(para) + 2 > max_chars and current:
            chunks.append('\n\n'.join(current))
            current, current_len = [para], len(para)
        else:
            current.append(para)
            current_len += len(para) + 2

    if current:
        chunks.append('\n\n'.join(current))

    return [c for c in chunks if c.strip()]


# ---------------------------------------------------------------------------
# Audio utilities
# ---------------------------------------------------------------------------

def get_wav_duration(path: str) -> float:
    with wave.open(path, 'r') as w:
        return w.getnframes() / w.getframerate()


def clean_text(text: str) -> str:
    return (text
        .replace('\u2018', "'").replace('\u2019', "'")
        .replace('\u201c', '"').replace('\u201d', '"')
        .replace('\u2014', ' - ').replace('\u2013', ' - ')
        .replace('\u2026', '...')
    )


# ---------------------------------------------------------------------------
# FFmpeg M4B builder
# ---------------------------------------------------------------------------

def check_ffmpeg():
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        raise RuntimeError(
            "ffmpeg not found on PATH.\n"
            "Install from https://ffmpeg.org/download.html and add to PATH."
        )


def build_m4b(wav_paths: List[str], output_path: str,
              title: str, author: str,
              chapter_info: List[Tuple[str, float]],
              log_cb: Callable):
    """Concatenate WAVs and encode to M4B with chapter markers."""
    tmpdir = tempfile.mkdtemp()
    try:
        concat_file = os.path.join(tmpdir, 'concat.txt')
        with open(concat_file, 'w', encoding='utf-8') as f:
            for p in wav_paths:
                f.write(f"file '{p}'\n")

        total_dur = sum(get_wav_duration(p) for p in wav_paths)
        meta_lines = [
            ';FFMETADATA1',
            f'title={title}',
            f'artist={author}',
            '',
        ]
        for i, (chap_title, start_sec) in enumerate(chapter_info):
            end_sec = chapter_info[i + 1][1] if i + 1 < len(chapter_info) else total_dur
            meta_lines += [
                '[CHAPTER]',
                'TIMEBASE=1/1000',
                f'START={int(start_sec * 1000)}',
                f'END={int(end_sec * 1000)}',
                f'title={chap_title}',
                '',
            ]

        meta_file = os.path.join(tmpdir, 'metadata.txt')
        with open(meta_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(meta_lines))

        concat_wav = os.path.join(tmpdir, 'full.wav')
        log_cb('Concatenating audio segments...')
        result = subprocess.run(
            ['ffmpeg', '-y', '-f', 'concat', '-safe', '0',
             '-i', concat_file, '-c', 'copy', concat_wav],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg concat failed:\n{result.stderr[-500:]}")

        log_cb('Encoding to M4B (AAC 64k)...')
        result = subprocess.run(
            ['ffmpeg', '-y',
             '-i', concat_wav,
             '-i', meta_file,
             '-map_metadata', '1',
             '-vn', '-acodec', 'aac', '-b:a', '64k',
             output_path],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg encode failed:\n{result.stderr[-500:]}")

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Converter factory
# ---------------------------------------------------------------------------

def get_converter(model_path: str,
                  model_type: Optional[str] = None,
                  **kwargs) -> Any:
    """Detect model type and return the appropriate converter instance."""
    if not model_type:
        model_type = detect_model_type(model_path)

    if model_type == MODEL_STREAMING:
        from converter_streaming import VibeVoiceStreamingConverter
        return VibeVoiceStreamingConverter(model_path=model_path, **kwargs)
    elif model_type == MODEL_FULL:
        from converter_full import VibeVoiceFullConverter
        return VibeVoiceFullConverter(model_path=model_path, **kwargs)
    else:
        raise ValueError(f'Unknown model_type: {model_type!r}')
