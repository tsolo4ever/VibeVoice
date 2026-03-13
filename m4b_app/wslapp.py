# wslapp.py (runs inside WSL)

import argparse
import sys
from converter import get_converter

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--voice", required=True)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--cfg", type=float, default=1.5)
    parser.add_argument("--title", default="")
    parser.add_argument("--author", default="")
    parser.add_argument("--model-type", default="")
    parser.add_argument("--debug", action="store_true", default=False)
    args = parser.parse_args()

    def log(msg):
        print(f"[LOG] {msg}", flush=True)

    def progress(ci, total_c, ki, total_k):
        print(f"[PROGRESS] {ci}/{total_c}/{ki}/{total_k}", flush=True)

    conv = get_converter(
        model_path=args.model,
        model_type=args.model_type or None,
        voice_path=args.voice,
        ddpm_steps=args.steps,
        cfg_scale=args.cfg,
        log_cb=log,
        progress_cb=progress,
        debug=args.debug,
    )

    conv.convert(
        input_path=args.input,
        output_path=args.output,
        title=args.title,
        author=args.author,
    )

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        print(f"[LOG] ERROR: {e}", flush=True)
        for line in traceback.format_exc().splitlines():
            print(f"[LOG] {line}", flush=True)
        sys.exit(1)