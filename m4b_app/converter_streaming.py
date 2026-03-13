import os
import re
import sys
import json
import copy
import time
import contextlib
import subprocess
import tempfile
import shutil
import wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import queue as _queue
import threading

import torch

from converter import (
    SAMPLE_RATE, MAX_CHUNK_CHARS, MIN_CHUNK_CHARS, PARALLEL_CHUNKS,
    check_ffmpeg, build_m4b, parse_epub, parse_txt,
    chunk_text, clean_text, get_wav_duration,
)


class VibeVoiceStreamingConverter:
    """Converter for VibeVoice-Realtime-0.5B (streaming inference model)."""

    def __init__(self,
                 model_path: str,
                 voice_path: str,
                 device: Optional[str] = None,
                 ddpm_steps: int = 10,
                 cfg_scale: float = 1.5,
                 log_cb: Optional[Callable] = None,
                 progress_cb: Optional[Callable] = None,
                 chunk_cb: Optional[Callable] = None,
                 debug: bool = False,
                 parallel_chunks: int = PARALLEL_CHUNKS,
                 model_instances: int = 1):
        self.model_path = model_path
        self.voice_path = voice_path
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.ddpm_steps = ddpm_steps
        self.cfg_scale = cfg_scale
        self.log_cb = log_cb or print
        self.progress_cb = progress_cb
        self.chunk_cb = chunk_cb
        self.debug = debug
        self.parallel_chunks = parallel_chunks
        self.model_instances = model_instances
        self.processor = None
        self._model_pool: _queue.Queue = _queue.Queue()
        self._stop = False

    def stop(self):
        self._stop = True

    def _acquire_model(self):
        """Acquire a model from the pool, polling so cancellation is respected."""
        while True:
            if self._stop:
                raise RuntimeError('Cancelled.')
            try:
                return self._model_pool.get(timeout=0.5)
            except _queue.Empty:
                continue

    def _dbg(self, msg: str):
        if self.debug:
            self.log_cb(f'[DEBUG] {msg}')

    def _dbg_voice_structure(self):
        obj = self._voice_prompt_orig

        def _describe(v):
            if torch.is_tensor(v):
                return f'tensor {tuple(v.shape)} {v.dtype} on {v.device}'
            if isinstance(v, (list, tuple)):
                inner = _describe(v[0]) if v else 'empty'
                return f'{type(v).__name__}[{len(v)}] -> {inner}'
            try:
                return f'{type(v).__name__}: {v!r}'
            except Exception as e:
                return f'{type(v).__name__}: <repr error: {e}>'

        self._dbg(f'voice prompt type: {type(obj).__name__}')
        if isinstance(obj, dict):
            for k, v in obj.items():
                self._dbg(f'  [{k}] {_describe(v)}')
        if hasattr(obj, '__dict__'):
            for k, v in vars(obj).items():
                self._dbg(f'  .{k} {_describe(v)}')

    def load_model(self):
        from vibevoice.modular.modeling_vibevoice_streaming_inference import (
            VibeVoiceStreamingForConditionalGenerationInference)
        from vibevoice.processor.vibevoice_streaming_processor import (
            VibeVoiceStreamingProcessor)

        self.log_cb(f'Loading processor from {self.model_path}...')
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='.*torch_dtype.*deprecated.*')
            self.processor = VibeVoiceStreamingProcessor.from_pretrained(self.model_path)

        dtype = torch.bfloat16 if self.device == 'cuda' else torch.float32
        attn = 'flash_attention_2' if self.device == 'cuda' else 'sdpa'

        for i in range(self.model_instances):
            tag = f' [{i+1}/{self.model_instances}]' if self.model_instances > 1 else ''
            self.log_cb(f'Loading model{tag} on {self.device} ({dtype})...')
            try:
                if self.device == 'mps':
                    m = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
                        self.model_path, dtype=dtype, attn_implementation=attn)
                    m.to('mps')
                else:
                    m = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
                        self.model_path, dtype=dtype,
                        device_map=self.device, attn_implementation=attn)
            except Exception as e:
                self.log_cb(f'Attention fallback: {e}')
                m = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
                    self.model_path, dtype=dtype,
                    device_map=(self.device if self.device != 'mps' else None),
                    attn_implementation='sdpa')
                if self.device == 'mps':
                    m.to('mps')

            m.eval()
            m.set_ddpm_inference_steps(num_steps=self.ddpm_steps)

            if self.device == 'cuda':
                if i == 0:
                    self.log_cb('Compiling model(s) (first run will be slower)...')
                m = torch.compile(m)

            self._model_pool.put(m)

        self.log_cb(f'Loading voice: {os.path.basename(self.voice_path)}')

        self._voice_prompt_orig = torch.load(
            self.voice_path, map_location=self.device, weights_only=False)
        self.log_cb('Model ready.')
        if self.debug:
            self._dbg_voice_structure()

    def unload(self):
        """Drain the model pool, delete all model instances, and free CUDA memory."""
        while not self._model_pool.empty():
            try:
                m = self._model_pool.get_nowait()
                del m
            except _queue.Empty:
                break
        self._voice_prompt_orig = None
        self.processor = None
        if self.device == 'cuda':
            torch.cuda.empty_cache()

    def _clone_prompt(self, obj):
        """Recursively clone obj: tensors use .clone() (stays on GPU), containers
        are rebuilt preserving their type, other objects are shallow-copied."""
        if torch.is_tensor(obj):
            return obj.clone()
        if isinstance(obj, dict):
            new = copy.copy(obj)
            dict.clear(new)
            dict.update(new, {k: self._clone_prompt(v) for k, v in obj.items()})
            return new
        if isinstance(obj, (tuple, list)):
            cloned = [self._clone_prompt(v) for v in obj]
            return type(obj)(cloned)
        if hasattr(obj, '__dict__') and not isinstance(obj, type):
            new = copy.copy(obj)
            for k, v in list(vars(obj).items()):
                setattr(new, k, self._clone_prompt(v))
            return new
        return obj

    def _fresh_prompt(self):
        """Clone the voice prompt using GPU tensor.clone() instead of CPU deepcopy."""
        if self.debug:
            t = time.time()
            result = self._clone_prompt(self._voice_prompt_orig)
            self._dbg(f'clone time: {time.time() - t:.3f}s')
            return result
        return self._clone_prompt(self._voice_prompt_orig)

    def _write_silence(self, out_wav: str, duration: float = 0.5):
        import struct
        n_samples = int(SAMPLE_RATE * duration)
        with wave.open(out_wav, 'w') as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(SAMPLE_RATE)
            w.writeframes(struct.pack('<' + 'h' * n_samples, *([0] * n_samples)))

    def _generate_chunk(self, text: str, out_wav: str):
        """Generate audio for one text chunk. Splits in half and retries on tensor errors."""
        try:
            if self.debug:
                self._dbg(f'chunk preview: {text[:80]!r}')
                if self.device == 'cuda':
                    self._dbg(f'GPU mem: {torch.cuda.memory_allocated() / 1e6:.0f}MB alloc'
                              f' / {torch.cuda.memory_reserved() / 1e6:.0f}MB reserved')

            t_prep = time.time()
            voice_prompt = self._fresh_prompt()

            inputs = self.processor.process_input_with_cached_prompt(
                text=text,
                cached_prompt=voice_prompt,
                padding=True,
                return_tensors='pt',
                return_attention_mask=True,
            )
            for k, v in inputs.items():
                if torch.is_tensor(v):
                    inputs[k] = v.to(self.device)

            all_prefilled = self._clone_prompt(voice_prompt)

            if self.debug:
                self._dbg(f'prep time: {time.time() - t_prep:.3f}s')
                t_gen = time.time()

            model = self._acquire_model()
            try:
                with open(os.devnull, 'w') as _devnull, \
                     contextlib.redirect_stderr(_devnull if not self.debug else sys.stderr):
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=None,
                        cfg_scale=self.cfg_scale,
                        tokenizer=self.processor.tokenizer,
                        generation_config={'do_sample': False},
                        verbose=False,
                        all_prefilled_outputs=all_prefilled,
                    )
            finally:
                self._model_pool.put(model)

            if self.debug:
                self._dbg(f'inference time: {time.time() - t_gen:.3f}s')

            self.processor.save_audio(outputs.speech_outputs[0], output_path=out_wav)

        except RuntimeError as e:
            if 'size of tensor' not in str(e) and 'must match' not in str(e):
                raise

            if len(text) < MIN_CHUNK_CHARS:
                self.log_cb(f'    WARNING: chunk too short ({len(text)} chars) for windowed attention — skipped.')
                self._write_silence(out_wav, duration=0.5)
                return

            sentences = re.split(r'(?<=[.!?])\s+', text.strip())
            mid = max(1, len(sentences) // 2)
            half_a = ' '.join(sentences[:mid])
            half_b = ' '.join(sentences[mid:])
            if not half_b:
                self.log_cb(f'    WARNING: cannot split further ({len(text)} chars) — skipped.')
                self._write_silence(out_wav, duration=0.5)
                return

            self.log_cb(f'    Tensor mismatch — retrying as 2 sub-chunks ({len(half_a)}/{len(half_b)} chars)')
            base, ext = os.path.splitext(out_wav)
            wav_a, wav_b = f'{base}_a{ext}', f'{base}_b{ext}'
            self._generate_chunk(half_a, wav_a)
            self._generate_chunk(half_b, wav_b)

            lst = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
            lst.write(f"file '{wav_a}'\nfile '{wav_b}'\n")
            lst.close()
            try:
                result = subprocess.run(
                    ['ffmpeg', '-y', '-f', 'concat', '-safe', '0',
                     '-i', lst.name, '-c', 'copy', out_wav],
                    capture_output=True, text=True
                )
            finally:
                os.unlink(lst.name)
                for f in (wav_a, wav_b):
                    if os.path.exists(f):
                        os.unlink(f)
            if result.returncode != 0:
                raise RuntimeError(f"ffmpeg sub-chunk concat failed:\n{result.stderr[-300:]}")

    def _ts(self) -> str:
        elapsed = int(time.time() - self._conv_start)
        m, s = divmod(elapsed, 60)
        return f'[+{m:02d}:{s:02d}]'

    def convert(self, input_path: str, output_path: str,
                title: str = '', author: str = '',
                meta: Optional[Dict] = None) -> str:
        self._stop = False
        self._conv_start = time.time()
        check_ffmpeg()

        ext = Path(input_path).suffix.lower()
        self.log_cb(f'Parsing {ext.upper()} file...')
        if ext == '.epub':
            chapters = parse_epub(input_path)
        else:
            chapters = parse_txt(input_path)

        title = title or Path(input_path).stem
        self.log_cb(f'Found {len(chapters)} chapter(s).')

        if self._model_pool.empty():
            self.load_model()

        # Persistent work dir for resume support (next to output file)
        work_dir = output_path + '.work'
        manifest_path = os.path.join(work_dir, 'manifest.json')
        manifest_lock = threading.Lock()
        os.makedirs(work_dir, exist_ok=True)

        manifest: Dict[str, str] = {}
        if os.path.exists(manifest_path):
            try:
                with open(manifest_path) as f:
                    manifest = json.load(f)
                n_done = sum(1 for k in manifest if not k.startswith('_'))
                self.log_cb(f'Resuming: {n_done} chunk(s) already completed.')
            except Exception:
                self.log_cb('Warning: manifest unreadable — starting fresh.')
                manifest = {}

        if '_meta' not in manifest and meta:
            manifest['_meta'] = meta
            with open(manifest_path, 'w') as mf:
                json.dump(manifest, mf)

        wav_paths: List[str] = []
        chapter_info: List[Tuple[str, float]] = []
        current_sec = 0.0
        tmpdir = tempfile.mkdtemp()

        try:
            for ci, (chap_title, chap_text) in enumerate(chapters):
                if self._stop:
                    raise RuntimeError('Cancelled.')

                self.log_cb(f'Chapter {ci + 1}/{len(chapters)}: {chap_title}')
                chapter_info.append((chap_title, current_sec))

                chunks = chunk_text(clean_text(chap_text))
                valid = [(ki, chunk) for ki, chunk in enumerate(chunks) if chunk.strip()]
                total_k = len(valid)
                total_c = len(chapters)

                # Capture loop variables for closure
                _ci, _total_c, _total_k = ci, total_c, total_k

                def do_chunk(item, _ci=_ci, _total_c=_total_c, _total_k=_total_k):
                    ki, chunk = item
                    chunk_key = f'c{_ci:04d}_k{ki:04d}'
                    out_wav = os.path.join(tmpdir, f'{chunk_key}.wav')
                    backup_wav = os.path.join(work_dir, f'{chunk_key}.wav')

                    with manifest_lock:
                        already_done = chunk_key in manifest and os.path.exists(backup_wav)

                    if already_done:
                        self.log_cb(f'  Chunk {ki + 1}/{_total_k} {self._ts()} (resumed)')
                        shutil.copy2(backup_wav, out_wav)
                        if self.chunk_cb:
                            self.chunk_cb(backup_wav)
                        return out_wav

                    if self._stop:
                        raise RuntimeError('Cancelled.')
                    if self.progress_cb:
                        self.progress_cb(_ci, _total_c, ki, _total_k)
                    self.log_cb(f'  Chunk {ki + 1}/{_total_k} {self._ts()} ({len(chunk)} chars)')
                    self._generate_chunk(chunk, out_wav)

                    # Back up to work dir and update manifest
                    shutil.copy2(out_wav, backup_wav)
                    with manifest_lock:
                        manifest[chunk_key] = backup_wav
                        with open(manifest_path, 'w') as mf:
                            json.dump(manifest, mf)

                    if self.chunk_cb:
                        self.chunk_cb(backup_wav)
                    return out_wav

                with ThreadPoolExecutor(max_workers=self.parallel_chunks) as pool:
                    for out_wav in pool.map(do_chunk, valid):
                        if self._stop:
                            raise RuntimeError('Cancelled.')
                        wav_paths.append(out_wav)
                        current_sec += get_wav_duration(out_wav)

            if not wav_paths:
                raise RuntimeError('No audio was generated.')

            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            build_m4b(wav_paths, output_path, title, author, chapter_info, self.log_cb)
            self.log_cb(f'Done! {self._ts()} -> {output_path}')
            shutil.rmtree(work_dir, ignore_errors=True)
            return output_path

        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)
