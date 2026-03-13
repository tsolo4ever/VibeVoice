from typing import Callable, Optional


class VibeVoiceFullConverter:
    """Converter for VibeVoice 1.5B (full model) — implementation pending."""

    def __init__(self,
                 model_path: str,
                 voice_path: str,
                 device: Optional[str] = None,
                 ddpm_steps: int = 10,
                 cfg_scale: float = 1.5,
                 log_cb: Optional[Callable] = None,
                 progress_cb: Optional[Callable] = None,
                 chunk_cb: Optional[Callable] = None,
                 debug: bool = False):
        self.model_path = model_path
        self.voice_path = voice_path
        self.log_cb = log_cb or print
        self.progress_cb = progress_cb
        self.debug = debug
        self._stop = False

    def stop(self):
        self._stop = True

    def convert(self, input_path: str, output_path: str,
                title: str = '', author: str = '', meta=None) -> str:
        raise NotImplementedError(
            'VibeVoice 1.5B (full model) is not yet implemented.\n'
            'Please use VibeVoice-Realtime-0.5B (model_type: vibevoice_streaming).'
        )
