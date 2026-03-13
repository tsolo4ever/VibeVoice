import os
import json
import time
import queue
import subprocess
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path

DEFAULT_MODEL_PATH  = '/mnt/d/GitHub/VibeVoice-Realtime-0.5B'
DEFAULT_VOICES_DIR  = '/mnt/d/GitHub/VibeVoice/demo/voices/streaming_model/en'
DEFAULT_OUTPUT_DIR  = '/mnt/h/audiobooks'
PREFERRED_VOICES    = ['en-Carter_man', 'en-Emma_woman', 'en-Grace_woman', 'en-Mike_man']
CONFIG_PATH         = Path.home() / '.vibevoice_config.json'


def get_voices(voices_dir: str) -> dict:
    """Return {display_name: full_path} for all .pt files in voices_dir."""
    if not os.path.isdir(voices_dir):
        return {}
    return {
        os.path.splitext(f)[0]: os.path.join(voices_dir, f)
        for f in sorted(os.listdir(voices_dir))
        if f.endswith('.pt')
    }


class SettingsDialog(tk.Toplevel):

    def __init__(self, parent):
        super().__init__(parent)
        self._app = parent
        self.title('Settings')
        self.resizable(False, False)
        self.transient(parent)
        self.grab_set()

        # Snapshot current values so Cancel can revert
        self._saved = {
            'output_dir':     parent._output_dir_var.get(),
            'voices_dir':     parent._voices_dir_var.get(),
            'model_path':     parent._model_var.get(),
            'model_type':     parent._model_type_var.get(),
            'quality':        parent._quality_var.get(),
            'cfg_scale':      parent._cfg_var.get(),
            'parallel_chunks': parent._parallel_var.get(),
            'model_instances': parent._instances_var.get(),
            'debug':          parent._debug_var.get(),
        }

        self._build()
        self.protocol('WM_DELETE_WINDOW', self._cancel)
        self.wait_window()

    def _build(self):
        p = {'padx': 8, 'pady': 3}
        app = self._app

        # --- Paths ---
        pf = ttk.LabelFrame(self, text='Paths')
        pf.grid(row=0, column=0, sticky='ew', padx=10, pady=(10, 4))
        pf.columnconfigure(1, weight=1)

        ttk.Label(pf, text='Output Folder:').grid(row=0, column=0, sticky='w', **p)
        ttk.Entry(pf, textvariable=app._output_dir_var, width=50).grid(row=0, column=1, sticky='ew', **p)
        ttk.Button(pf, text='Browse', command=self._browse_output_dir).grid(row=0, column=2, **p)

        ttk.Label(pf, text='Voices Dir:').grid(row=1, column=0, sticky='w', **p)
        ttk.Entry(pf, textvariable=app._voices_dir_var, width=50).grid(row=1, column=1, sticky='ew', **p)
        ttk.Button(pf, text='Browse', command=self._browse_voices_dir).grid(row=1, column=2, **p)

        ttk.Label(pf, text='Model Path:').grid(row=2, column=0, sticky='w', **p)
        ttk.Entry(pf, textvariable=app._model_var, width=50).grid(row=2, column=1, sticky='ew', **p)
        ttk.Button(pf, text='Browse', command=self._browse_model).grid(row=2, column=2, **p)

        ttk.Label(pf, text='Model Type:').grid(row=3, column=0, sticky='w', **p)
        ttk.Combobox(pf, textvariable=app._model_type_var, width=47, state='readonly',
                     values=['Auto-detect', 'Streaming 0.5B', 'Full 1.5B']).grid(
            row=3, column=1, columnspan=2, sticky='ew', **p)

        # --- Inference ---
        inf = ttk.LabelFrame(self, text='Inference')
        inf.grid(row=1, column=0, sticky='ew', padx=10, pady=4)

        ttk.Label(inf, text='Quality:').grid(row=0, column=0, sticky='w', **p)
        for col, (label, val) in enumerate([
            ('Fast  (5 steps)',  'Fast'),
            ('Balanced  (10)',   'Balanced'),
            ('Best  (20 steps)', 'Best'),
        ]):
            ttk.Radiobutton(inf, text=label, variable=app._quality_var, value=val).grid(
                row=0, column=col + 1, padx=6, pady=3)

        ttk.Label(inf, text='CFG Scale:').grid(row=1, column=0, sticky='w', **p)
        self._cfg_label = ttk.Label(inf, text=f'{app._cfg_var.get():.1f}', width=4)
        ttk.Scale(inf, from_=1.0, to=3.0, variable=app._cfg_var, orient='horizontal',
                  length=220, command=lambda v: self._cfg_label.config(text=f'{float(v):.1f}')
                  ).grid(row=1, column=1, columnspan=2, sticky='w', **p)
        self._cfg_label.grid(row=1, column=3, sticky='w')

        ttk.Label(inf, text='Parallel chunks:').grid(row=2, column=0, sticky='w', **p)
        ttk.Spinbox(inf, from_=1, to=8, textvariable=app._parallel_var, width=4).grid(
            row=2, column=1, sticky='w', padx=6, pady=3)
        ttk.Checkbutton(inf, text='Debug logging', variable=app._debug_var).grid(
            row=2, column=2, sticky='w', padx=6, pady=3)

        ttk.Label(inf, text='Model instances:').grid(row=3, column=0, sticky='w', **p)
        ttk.Spinbox(inf, from_=1, to=4, textvariable=app._instances_var, width=4).grid(
            row=3, column=1, sticky='w', padx=6, pady=3)
        ttk.Label(inf, text='(2 = true parallel inference, ~6GB VRAM)',
                  foreground='gray').grid(row=3, column=2, columnspan=2, sticky='w', padx=6, pady=3)

        # --- Buttons ---
        bf = ttk.Frame(self)
        bf.grid(row=2, column=0, pady=(4, 10))
        ttk.Button(bf, text='OK', command=self._ok, width=10).pack(side='left', padx=6)
        ttk.Button(bf, text='Cancel', command=self._cancel, width=10).pack(side='left', padx=6)

    def _browse_output_dir(self):
        path = filedialog.askdirectory(title='Select default output folder')
        if path:
            self._app._output_dir_var.set(path)

    def _browse_voices_dir(self):
        path = filedialog.askdirectory(title='Select voices directory')
        if path:
            self._app._voices_dir_var.set(path)

    def _browse_model(self):
        path = filedialog.askdirectory(title='Select model directory')
        if path:
            self._app._model_var.set(path)

    def _ok(self):
        self._app._reload_voices()
        self._app._detect_model_type()
        self._app._save_config()
        self.destroy()

    def _cancel(self):
        app = self._app
        app._output_dir_var.set(self._saved['output_dir'])
        app._voices_dir_var.set(self._saved['voices_dir'])
        app._model_var.set(self._saved['model_path'])
        app._model_type_var.set(self._saved['model_type'])
        app._quality_var.set(self._saved['quality'])
        app._cfg_var.set(self._saved['cfg_scale'])
        app._parallel_var.set(self._saved['parallel_chunks'])
        app._instances_var.set(self._saved['model_instances'])
        app._debug_var.set(self._saved['debug'])
        self.destroy()


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
        self._last_chunk_wav: str = ''
        self._preview_proc = None
        self._build_ui()
        accepted = self._load_config()
        if not accepted:
            if not self._show_disclaimer():
                self.destroy()
                return
        self._reload_voices()
        self._detect_model_type()
        self._poll_queue()
        self.protocol('WM_DELETE_WINDOW', self._on_close)

    # ------------------------------------------------------------------
    # Config persistence
    # ------------------------------------------------------------------

    def _load_config(self):
        try:
            with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
                cfg = json.load(f)
            if cfg.get('output_dir'):
                self._output_dir_var.set(cfg['output_dir'])
            if cfg.get('voices_dir'):
                self._voices_dir_var.set(cfg['voices_dir'])
            if cfg.get('model_path'):
                self._model_var.set(cfg['model_path'])
            if cfg.get('model_type'):
                self._model_type_var.set(cfg['model_type'])
            if cfg.get('quality'):
                self._quality_var.set(cfg['quality'])
            if cfg.get('cfg_scale') is not None:
                self._cfg_var.set(cfg['cfg_scale'])
            if cfg.get('parallel_chunks') is not None:
                self._parallel_var.set(cfg['parallel_chunks'])
            if cfg.get('model_instances') is not None:
                self._instances_var.set(cfg['model_instances'])
            if cfg.get('debug') is not None:
                self._debug_var.set(cfg['debug'])
            return cfg.get('disclaimer_accepted', False)
        except (FileNotFoundError, json.JSONDecodeError):
            return False

    def _save_config(self, disclaimer_accepted: bool = True):
        cfg = {
            'output_dir':          self._output_dir_var.get(),
            'voices_dir':          self._voices_dir_var.get(),
            'model_path':          self._model_var.get(),
            'model_type':          self._model_type_var.get(),
            'quality':             self._quality_var.get(),
            'cfg_scale':           self._cfg_var.get(),
            'parallel_chunks':     self._parallel_var.get(),
            'model_instances':     self._instances_var.get(),
            'debug':               self._debug_var.get(),
            'disclaimer_accepted': disclaimer_accepted,
        }
        try:
            with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
                json.dump(cfg, f, indent=2)
        except Exception:
            pass

    def _show_disclaimer(self) -> bool:
        """Show first-launch disclaimer. Returns True if accepted."""
        dlg = tk.Toplevel(self)
        dlg.title('Usage Disclaimer')
        dlg.resizable(False, False)
        dlg.transient(self)
        dlg.grab_set()

        text = (
            "VibeVoice M4B Converter — Personal Use Only\n\n"
            "By using this software you agree to the following:\n\n"
            "1. Personal use only. This tool is intended for converting\n"
            "   books you legally own for your personal listening.\n\n"
            "2. Copyright compliance. Do not use this tool to reproduce\n"
            "   or distribute copyrighted material without authorization\n"
            "   from the rights holder.\n\n"
            "3. No commercial use. Generated audio files may not be sold,\n"
            "   licensed, or redistributed commercially.\n\n"
            "4. Model license. The VibeVoice TTS model is subject to its\n"
            "   own license terms. Use in accordance with those terms.\n\n"
            "This software is provided as-is with no warranty."
        )

        ttk.Label(dlg, text=text, justify='left', padding=(16, 12)).pack()

        accepted = tk.BooleanVar(value=False)

        def _accept():
            accepted.set(True)
            dlg.destroy()

        def _decline():
            dlg.destroy()

        bf = ttk.Frame(dlg)
        bf.pack(pady=(0, 12))
        ttk.Button(bf, text='I Accept', command=_accept, width=12).pack(side='left', padx=8)
        ttk.Button(bf, text='Decline', command=_decline, width=12).pack(side='left', padx=8)

        self.wait_window(dlg)

        if accepted.get():
            self._save_config(disclaimer_accepted=True)
        return accepted.get()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        p = {'padx': 8, 'pady': 3}

        # --- Menubar ---
        menubar = tk.Menu(self)
        options_menu = tk.Menu(menubar, tearoff=0)
        options_menu.add_command(label='Settings...', command=self._open_settings)
        menubar.add_cascade(label='Options', menu=options_menu)
        self.config(menu=menubar)

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

        # --- Voice ---
        vf = ttk.LabelFrame(self, text='Voice')
        vf.grid(row=1, column=0, sticky='ew', padx=10, pady=4)
        vf.columnconfigure(1, weight=1)

        # Hidden vars used by Settings dialog and converter
        self._output_dir_var  = tk.StringVar(value=DEFAULT_OUTPUT_DIR)
        self._model_var       = tk.StringVar(value=DEFAULT_MODEL_PATH)
        self._model_type_var  = tk.StringVar(value='Auto-detect')
        self._voices_dir_var  = tk.StringVar(value=DEFAULT_VOICES_DIR)
        self._quality_var     = tk.StringVar(value='Balanced')
        self._cfg_var         = tk.DoubleVar(value=1.5)
        self._parallel_var    = tk.IntVar(value=2)
        self._debug_var       = tk.BooleanVar(value=False)
        self._instances_var   = tk.IntVar(value=1)

        ttk.Label(vf, text='Voice:').grid(row=0, column=0, sticky='w', **p)
        self._voice_var = tk.StringVar()
        self._voice_combo = ttk.Combobox(vf, textvariable=self._voice_var, width=49, state='readonly')
        self._voice_combo.grid(row=0, column=1, columnspan=2, sticky='ew', **p)

        # --- Actions ---
        af = ttk.Frame(self)
        af.grid(row=2, column=0, sticky='ew', padx=10, pady=6)

        self._convert_btn = ttk.Button(af, text='Convert', command=self._start_convert, width=18)
        self._convert_btn.pack(side='left', padx=(0, 6))

        self._cancel_btn = ttk.Button(af, text='Cancel', command=self._cancel,
                                      width=10, state='disabled')
        self._cancel_btn.pack(side='left', padx=(0, 6))

        self._resume_btn = ttk.Button(af, text='Resume...', command=self._browse_resume, width=12)
        self._resume_btn.pack(side='left', padx=(0, 6))

        self._preview_btn = ttk.Button(af, text='Preview chunk', command=self._preview_chunk,
                                       width=14, state='disabled')
        self._preview_btn.pack(side='left')

        # Progress
        self._progress_var = tk.DoubleVar(value=0)
        ttk.Progressbar(self, variable=self._progress_var, maximum=100, length=580).grid(
            row=3, column=0, sticky='ew', padx=10, pady=(0, 2))

        sf = ttk.Frame(self)
        sf.grid(row=4, column=0, sticky='ew', padx=10)
        sf.columnconfigure(0, weight=1)
        self._status_var = tk.StringVar(value='Ready')
        ttk.Label(sf, textvariable=self._status_var, anchor='w').grid(row=0, column=0, sticky='ew')
        self._timer_var = tk.StringVar(value='')
        ttk.Label(sf, textvariable=self._timer_var, anchor='e').grid(row=0, column=1, sticky='e', padx=(8, 0))

        # Log
        lf = ttk.LabelFrame(self, text='Log')
        lf.grid(row=5, column=0, sticky='nsew', padx=10, pady=(4, 10))
        lf.columnconfigure(0, weight=1)
        lf.rowconfigure(0, weight=1)
        self.rowconfigure(5, weight=1)
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

    def _open_settings(self):
        SettingsDialog(self)

    def _detect_model_type(self):
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
        self._resume_btn.config(state='disabled' if converting else 'normal')
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
            self._output_var.set(str(Path(self._output_dir_var.get()) / (Path(path).stem + '.m4b')))

    def _browse_output(self):
        path = filedialog.asksaveasfilename(
            title='Save M4B as',
            defaultextension='.m4b',
            filetypes=[('M4B Audiobook', '*.m4b'), ('All files', '*.*')]
        )
        if path:
            self._output_var.set(path)

    def _browse_resume(self):
        out = self._output_var.get()
        initialdir = str(Path(out).parent) if out else str(Path.home())
        path = filedialog.askopenfilename(
            title='Select manifest.json from interrupted conversion',
            initialdir=initialdir,
            filetypes=[('JSON', '*.json'), ('All files', '*.*')],
        )
        if not path:
            return
        try:
            with open(path) as f:
                data = json.load(f)
        except Exception as e:
            messagebox.showerror('Error', f'Could not read manifest:\n{e}')
            return

        meta = data.get('_meta')
        if not meta:
            messagebox.showerror('Error',
                'Manifest has no metadata.\n'
                'Fill in the fields manually and click Convert.')
            return

        if meta.get('input_path'):
            self._input_var.set(meta['input_path'])
        if meta.get('output_path'):
            self._output_var.set(meta['output_path'])
        if meta.get('title'):
            self._title_var.set(meta['title'])
        if meta.get('author'):
            self._author_var.set(meta['author'])
        if meta.get('model_path'):
            self._model_var.set(meta['model_path'])
        if meta.get('voices_dir'):
            self._voices_dir_var.set(meta['voices_dir'])
            self._reload_voices()
        if meta.get('voice_name') and meta['voice_name'] in self._voices:
            self._voice_var.set(meta['voice_name'])
        if meta.get('model_type_display'):
            self._model_type_var.set(meta['model_type_display'])
        steps = meta.get('ddpm_steps')
        if steps == 5:
            self._quality_var.set('Fast')
        elif steps == 20:
            self._quality_var.set('Best')
        else:
            self._quality_var.set('Balanced')
        if meta.get('cfg_scale') is not None:
            self._cfg_var.set(meta['cfg_scale'])
        if meta.get('parallel_chunks') is not None:
            self._parallel_var.set(meta['parallel_chunks'])
        if meta.get('model_instances') is not None:
            self._instances_var.set(meta['model_instances'])
        if meta.get('debug') is not None:
            self._debug_var.set(meta['debug'])

        n_done = sum(1 for k in data if not k.startswith('_'))
        self._log(f'Resume manifest loaded: {n_done} chunk(s) already done.')
        self._log('Settings restored. Click Convert to resume.')

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
            messagebox.showerror('Error', 'No voice selected. Check Voices Dir in Options > Settings.')
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
        import traceback
        conv = None
        try:
            from converter import get_converter

            def log_cb(msg):
                self._queue.put(("log", msg))

            def progress_cb(ci, total_c, ki, total_k):
                if total_c > 0 and total_k > 0:
                    pct = (ci / total_c + (ki / total_k) / total_c) * 100
                    status = f"Ch {ci+1}/{total_c}  |  Chunk {ki+1}/{total_k}"
                    self._queue.put(("progress", pct, status))

            def chunk_cb(wav_path):
                self._queue.put(("chunk", wav_path))

            mt_map = {'Streaming 0.5B': 'vibevoice_streaming', 'Full 1.5B': 'vibevoice'}
            mt_arg = mt_map.get(self._model_type_var.get(), '') or None

            meta = {
                'input_path':        self._input_var.get(),
                'output_path':       self._output_var.get(),
                'title':             self._title_var.get(),
                'author':            self._author_var.get(),
                'model_path':        self._model_var.get(),
                'voice_path':        self._voices[self._voice_var.get()],
                'voice_name':        self._voice_var.get(),
                'voices_dir':        self._voices_dir_var.get(),
                'model_type_display': self._model_type_var.get(),
                'ddpm_steps':        self._ddpm_steps(),
                'cfg_scale':         self._cfg_var.get(),
                'parallel_chunks':   self._parallel_var.get(),
                'model_instances':   self._instances_var.get(),
                'debug':             self._debug_var.get(),
            }

            conv = get_converter(
                model_path=self._model_var.get(),
                model_type=mt_arg,
                voice_path=self._voices[self._voice_var.get()],
                ddpm_steps=self._ddpm_steps(),
                cfg_scale=self._cfg_var.get(),
                log_cb=log_cb,
                progress_cb=progress_cb,
                chunk_cb=chunk_cb,
                debug=self._debug_var.get(),
                parallel_chunks=self._parallel_var.get(),
                model_instances=self._instances_var.get(),
            )
            self._conv = conv

            conv.convert(
                input_path=self._input_var.get(),
                output_path=self._output_var.get(),
                title=self._title_var.get(),
                author=self._author_var.get(),
                meta=meta,
            )

            self._conv = None
            self._queue.put(("done", None))

        except Exception as e:
            if conv is not None:
                conv.unload()
            self._conv = None
            if 'Cancelled.' in str(e):
                self._queue.put(("cancelled", None))
                return
            for line in traceback.format_exc().splitlines():
                self._queue.put(("log", line))
            self._queue.put(("error", str(e)))

    def _cancel(self):
        conv = self._conv
        if conv is None:
            return
        self._cancel_btn.config(state='disabled')
        self._status_var.set('Finishing current chunks...')
        self._log('Cancel requested — finishing current chunks before stopping.')
        conv.stop()

    def _on_close(self):
        conv = self._conv
        if conv is not None:
            conv.stop()
            conv.unload()
        self._conv = None
        if self._preview_proc and self._preview_proc.poll() is None:
            self._preview_proc.kill()
        self._save_config()
        self.destroy()

    def _preview_chunk(self):
        if self._preview_proc and self._preview_proc.poll() is None:
            self._preview_proc.kill()
            self._preview_proc = None
            return
        wav = self._last_chunk_wav
        if not wav or not os.path.exists(wav):
            return
        self._preview_proc = subprocess.Popen(
            ['ffplay', '-nodisp', '-autoexit', '-loglevel', 'quiet', wav]
        )

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
                elif kind == 'chunk':
                    self._last_chunk_wav = item[1]
                    self._preview_btn.config(state='normal')
                elif kind == 'progress':
                    self._progress_var.set(item[1])
                    self._status_var.set(item[2])
                elif kind == 'cancelled':
                    self._status_var.set('Cancelled')
                    self._set_converting(False)
                    work_dir = self._output_var.get() + '.work'
                    messagebox.showinfo('Cancelled',
                        f'Stopped after current chunks completed.\n\n'
                        f'Progress saved to:\n{work_dir}\n\n'
                        f'Click Resume to continue from here.')
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
