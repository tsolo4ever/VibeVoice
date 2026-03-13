import os
import json
import time
import queue
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path

DEFAULT_MODEL_PATH = '/mnt/d/GitHub/VibeVoice-Realtime-0.5B'
DEFAULT_VOICES_DIR = '/mnt/d/GitHub/VibeVoice/demo/voices/streaming_model/en'
PREFERRED_VOICES   = ['en-Carter_man', 'en-Emma_woman', 'en-Grace_woman', 'en-Mike_man']


def get_voices(voices_dir: str) -> dict:
    """Return {display_name: full_path} for all .pt files in voices_dir."""
    if not os.path.isdir(voices_dir):
        return {}
    return {
        os.path.splitext(f)[0]: os.path.join(voices_dir, f)
        for f in sorted(os.listdir(voices_dir))
        if f.endswith('.pt')
    }



class App(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title('VibeVoice -> M4B Audiobook Converter')
        self.resizable(False, False)
        self._queue = queue.Queue()
        self._conv = None
        self._voices: dict = {}
        self._start_time = None
        self._timer_running = False
        self._build_ui()
        self._reload_voices()
        self._detect_model_type()
        self._poll_queue()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        p = {'padx': 8, 'pady': 3}

        # --- Files ---
        ff = ttk.LabelFrame(self, text='Files')
        ff.grid(row=0, column=0, sticky='ew', padx=10, pady=(10, 4))
        ff.columnconfigure(1, weight=1)

        ttk.Label(ff, text='Input (EPUB / TXT):').grid(row=0, column=0, sticky='w', **p)
        self._input_var = tk.StringVar()
        ttk.Entry(ff, textvariable=self._input_var, width=52).grid(row=0, column=1, sticky='ew', **p)
        ttk.Button(ff, text='Browse', command=self._browse_input).grid(row=0, column=2, **p)

        ttk.Label(ff, text='Output M4B:').grid(row=1, column=0, sticky='w', **p)
        self._output_var = tk.StringVar()
        ttk.Entry(ff, textvariable=self._output_var, width=52).grid(row=1, column=1, sticky='ew', **p)
        ttk.Button(ff, text='Browse', command=self._browse_output).grid(row=1, column=2, **p)

        ttk.Label(ff, text='Title:').grid(row=2, column=0, sticky='w', **p)
        self._title_var = tk.StringVar()
        ttk.Entry(ff, textvariable=self._title_var, width=52).grid(row=2, column=1, columnspan=2, sticky='ew', **p)

        ttk.Label(ff, text='Author:').grid(row=3, column=0, sticky='w', **p)
        self._author_var = tk.StringVar()
        ttk.Entry(ff, textvariable=self._author_var, width=52).grid(row=3, column=1, columnspan=2, sticky='ew', **p)

        # --- Model / Voice ---
        mf = ttk.LabelFrame(self, text='Model / Voice')
        mf.grid(row=1, column=0, sticky='ew', padx=10, pady=4)
        mf.columnconfigure(1, weight=1)

        ttk.Label(mf, text='Model Path:').grid(row=0, column=0, sticky='w', **p)
        self._model_var = tk.StringVar(value=DEFAULT_MODEL_PATH)
        self._model_var.trace_add('write', lambda *_: self._detect_model_type())
        ttk.Entry(mf, textvariable=self._model_var, width=52).grid(row=0, column=1, sticky='ew', **p)
        ttk.Button(mf, text='Browse', command=self._browse_model).grid(row=0, column=2, **p)

        ttk.Label(mf, text='Model Type:').grid(row=1, column=0, sticky='w', **p)
        self._model_type_var = tk.StringVar(value='Auto-detect')
        self._model_type_combo = ttk.Combobox(
            mf, textvariable=self._model_type_var, width=49, state='readonly',
            values=['Auto-detect', 'Streaming 0.5B', 'Full 1.5B'])
        self._model_type_combo.grid(row=1, column=1, columnspan=2, sticky='ew', **p)

        ttk.Label(mf, text='Voices Dir:').grid(row=2, column=0, sticky='w', **p)
        self._voices_dir_var = tk.StringVar(value=DEFAULT_VOICES_DIR)
        self._voices_dir_var.trace_add('write', lambda *_: self._reload_voices())
        ttk.Entry(mf, textvariable=self._voices_dir_var, width=52).grid(row=2, column=1, sticky='ew', **p)
        ttk.Button(mf, text='Browse', command=self._browse_voices_dir).grid(row=2, column=2, **p)

        ttk.Label(mf, text='Voice:').grid(row=3, column=0, sticky='w', **p)
        self._voice_var = tk.StringVar()
        self._voice_combo = ttk.Combobox(mf, textvariable=self._voice_var, width=49, state='readonly')
        self._voice_combo.grid(row=3, column=1, columnspan=2, sticky='ew', **p)

        # --- Settings ---
        sf = ttk.LabelFrame(self, text='Settings')
        sf.grid(row=2, column=0, sticky='ew', padx=10, pady=4)

        ttk.Label(sf, text='Quality:').grid(row=0, column=0, sticky='w', **p)
        self._quality_var = tk.StringVar(value='Balanced')
        for col, (label, val) in enumerate([
            ('Fast  (5 steps)',  'Fast'),
            ('Balanced  (10)',   'Balanced'),
            ('Best  (20 steps)', 'Best'),
        ]):
            ttk.Radiobutton(sf, text=label, variable=self._quality_var, value=val).grid(
                row=0, column=col + 1, padx=6, pady=3)

        ttk.Label(sf, text='CFG Scale:').grid(row=1, column=0, sticky='w', **p)
        self._cfg_var = tk.DoubleVar(value=1.5)
        self._cfg_label = ttk.Label(sf, text='1.5', width=4)
        self._cfg_var.trace_add('write', lambda *_: self._cfg_label.config(
            text=f'{self._cfg_var.get():.1f}'))
        ttk.Scale(sf, from_=1.0, to=3.0, variable=self._cfg_var,
                  orient='horizontal', length=220).grid(row=1, column=1, columnspan=2, sticky='w', **p)
        self._cfg_label.grid(row=1, column=3, sticky='w')

        self._debug_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(sf, text='Debug logging', variable=self._debug_var).grid(
            row=2, column=1, sticky='w', padx=6, pady=3)

        # --- Actions ---
        af = ttk.Frame(self)
        af.grid(row=3, column=0, sticky='ew', padx=10, pady=6)

        self._convert_btn = ttk.Button(af, text='Convert', command=self._start_convert, width=18)
        self._convert_btn.pack(side='left', padx=(0, 6))

        self._cancel_btn = ttk.Button(af, text='Cancel', command=self._cancel,
                                      width=10, state='disabled')
        self._cancel_btn.pack(side='left')

        # Progress
        self._progress_var = tk.DoubleVar(value=0)
        ttk.Progressbar(self, variable=self._progress_var, maximum=100, length=580).grid(
            row=4, column=0, sticky='ew', padx=10, pady=(0, 2))

        sf = ttk.Frame(self)
        sf.grid(row=5, column=0, sticky='ew', padx=10)
        sf.columnconfigure(0, weight=1)
        self._status_var = tk.StringVar(value='Ready')
        ttk.Label(sf, textvariable=self._status_var, anchor='w').grid(row=0, column=0, sticky='ew')
        self._timer_var = tk.StringVar(value='')
        ttk.Label(sf, textvariable=self._timer_var, anchor='e').grid(row=0, column=1, sticky='e', padx=(8, 0))

        # Log
        lf = ttk.LabelFrame(self, text='Log')
        lf.grid(row=6, column=0, sticky='nsew', padx=10, pady=(4, 10))
        lf.columnconfigure(0, weight=1)
        lf.rowconfigure(0, weight=1)
        self.rowconfigure(6, weight=1)
        self.columnconfigure(0, weight=1)

        self._log_text = tk.Text(lf, height=11, width=74, state='disabled',
                                 wrap='word', font=('Consolas', 9))
        scroll = ttk.Scrollbar(lf, command=self._log_text.yview)
        self._log_text.configure(yscrollcommand=scroll.set)
        self._log_text.grid(row=0, column=0, sticky='nsew')
        scroll.grid(row=0, column=1, sticky='ns')

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _detect_model_type(self):
        """Read config.json from the model dir and update the Model Type dropdown."""
        cfg = os.path.join(self._model_var.get().strip(), 'config.json')
        if not os.path.isfile(cfg):
            return
        try:
            with open(cfg, 'r', encoding='utf-8') as f:
                mt = json.load(f).get('model_type', '')
            display = {'vibevoice_streaming': 'Streaming 0.5B', 'vibevoice': 'Full 1.5B'}.get(mt)
            if display:
                self._model_type_var.set(display)
        except Exception:
            pass

    def _reload_voices(self):
        self._voices = get_voices(self._voices_dir_var.get())
        names = list(self._voices.keys())
        self._voice_combo['values'] = names
        if names:
            for pref in PREFERRED_VOICES:
                if pref in names:
                    self._voice_var.set(pref)
                    return
            self._voice_var.set(names[0])

    def _ddpm_steps(self) -> int:
        return {'Fast': 5, 'Balanced': 10, 'Best': 20}[self._quality_var.get()]

    def _log(self, msg: str):
        print(msg, flush=True)
        self._log_text.config(state='normal')
        self._log_text.insert('end', f'> {msg}\n')
        self._log_text.see('end')
        self._log_text.config(state='disabled')

    def _log_clear(self):
        self._log_text.config(state='normal')
        self._log_text.delete('1.0', 'end')
        self._log_text.config(state='disabled')

    def _set_converting(self, converting: bool):
        self._convert_btn.config(state='disabled' if converting else 'normal')
        self._cancel_btn.config(state='normal' if converting else 'disabled')
        if not converting:
            self._timer_running = False
            if self._start_time is not None:
                elapsed = int(time.time() - self._start_time)
                m, s = divmod(elapsed, 60)
                self._timer_var.set(f'{m:02d}:{s:02d}')

    def _update_timer(self):
        if not self._timer_running or self._start_time is None:
            return
        elapsed = int(time.time() - self._start_time)
        m, s = divmod(elapsed, 60)
        self._timer_var.set(f'{m:02d}:{s:02d}')
        self.after(1000, self._update_timer)

    # ------------------------------------------------------------------
    # Browse dialogs
    # ------------------------------------------------------------------

    def _browse_input(self):
        path = filedialog.askopenfilename(
            title='Select input file',
            filetypes=[
                ('Supported', '*.epub *.txt'),
                ('EPUB', '*.epub'),
                ('Text', '*.txt'),
                ('All files', '*.*'),
            ]
        )
        if path:
            self._input_var.set(path)
            self._title_var.set(Path(path).stem)
            self._output_var.set(str(Path(path).with_suffix('.m4b')))

    def _browse_output(self):
        path = filedialog.asksaveasfilename(
            title='Save M4B as',
            defaultextension='.m4b',
            filetypes=[('M4B Audiobook', '*.m4b'), ('All files', '*.*')]
        )
        if path:
            self._output_var.set(path)

    def _browse_model(self):
        path = filedialog.askdirectory(title='Select model directory')
        if path:
            self._model_var.set(path)

    def _browse_voices_dir(self):
        path = filedialog.askdirectory(title='Select voices directory')
        if path:
            self._voices_dir_var.set(path)

    # ------------------------------------------------------------------
    # Conversion
    # ------------------------------------------------------------------

    def _start_convert(self):
        input_path  = self._input_var.get().strip()
        output_path = self._output_var.get().strip()

        if not input_path or not os.path.exists(input_path):
            messagebox.showerror('Error', 'Input file not found.')
            return
        if not output_path:
            messagebox.showerror('Error', 'Please set an output path.')
            return
        out_dir = os.path.dirname(os.path.abspath(output_path))
        if not os.path.isdir(out_dir):
            messagebox.showerror('Error', f'Output directory does not exist:\n{out_dir}')
            return
        if not self._voice_var.get() or self._voice_var.get() not in self._voices:
            messagebox.showerror('Error', 'No voice selected. Check the Voices Dir.')
            return
        if not os.path.isdir(self._model_var.get()):
            messagebox.showerror('Error', f'Model path not found:\n{self._model_var.get()}')
            return

        self._set_converting(True)
        self._log_clear()
        self._progress_var.set(0)
        self._status_var.set('Starting...')
        self._timer_var.set('00:00')
        self._start_time = time.time()
        self._timer_running = True
        self._update_timer()

        thread = threading.Thread(target=self._run_conversion, daemon=True)
        thread.start()

    def _run_conversion(self):
        try:
            import traceback
            from converter import get_converter

            def log_cb(msg):
                self._queue.put(("log", msg))

            def progress_cb(ci, total_c, ki, total_k):
                if total_c > 0 and total_k > 0:
                    pct = (ci / total_c + (ki / total_k) / total_c) * 100
                    status = f"Ch {ci+1}/{total_c}  |  Chunk {ki+1}/{total_k}"
                    self._queue.put(("progress", pct, status))

            mt_map = {'Streaming 0.5B': 'vibevoice_streaming', 'Full 1.5B': 'vibevoice'}
            mt_arg = mt_map.get(self._model_type_var.get(), '') or None

            conv = get_converter(
                model_path=self._model_var.get(),
                model_type=mt_arg,
                voice_path=self._voices[self._voice_var.get()],
                ddpm_steps=self._ddpm_steps(),
                cfg_scale=self._cfg_var.get(),
                log_cb=log_cb,
                progress_cb=progress_cb,
                debug=self._debug_var.get(),
            )
            self._conv = conv

            conv.convert(
                input_path=self._input_var.get(),
                output_path=self._output_var.get(),
                title=self._title_var.get(),
                author=self._author_var.get(),
            )

            self._conv = None
            self._queue.put(("done", None))

        except Exception as e:
            self._conv = None
            if 'Cancelled.' in str(e):
                return
            import traceback
            for line in traceback.format_exc().splitlines():
                self._queue.put(("log", line))
            self._queue.put(("error", str(e)))

    def _cancel(self):
        conv = self._conv
        if conv is not None:
            conv.stop()
        self._conv = None
        self._log('Cancelled.')
        self._start_time = None
        self._set_converting(False)

    # ------------------------------------------------------------------
    # Queue polling (thread -> UI bridge)
    # ------------------------------------------------------------------

    def _poll_queue(self):
        try:
            while True:
                item = self._queue.get_nowait()
                kind = item[0]
                if kind == 'log':
                    self._log(item[1])
                elif kind == 'progress':
                    self._progress_var.set(item[1])
                    self._status_var.set(item[2])
                elif kind == 'done':
                    self._progress_var.set(100)
                    self._status_var.set('Complete!')
                    self._set_converting(False)
                    messagebox.showinfo('Done', f'Saved to:\n{self._output_var.get()}')
                elif kind == 'error':
                    self._status_var.set('Error')
                    self._set_converting(False)
                    self._log(f'ERROR: {item[1]}')
                    messagebox.showerror('Conversion Error', item[1])
        except queue.Empty:
            pass
        self.after(100, self._poll_queue)


if __name__ == '__main__':
    app = App()
    app.mainloop()