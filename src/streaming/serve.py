"""Start ONE chosen local ASR model; selecting DeepDML opts into private staging."""
import argparse
import os
from .model_options import MODEL_PRESETS, select_model


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=MODEL_PRESETS)
    parser.add_argument("--model-path", help="Override the preset directory; DeepDML also accepts the parent run")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--dtype", choices=["auto", "fp16", "bf16", "fp32"], default="auto")
    parser.add_argument("--beams", type=int, choices=[1, 3, 5], help="Override decoding: 1 is faster; recognition may change. Default: saved adapter policy / full-model greedy")
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    return args


def main(argv=None):
    args = parse_args(argv)
    path = select_model(args.model, args.model_path)
    os.environ["RECITER_MODEL_PATH"] = str(path)
    os.environ["RECITER_MODEL_PRESET"] = args.model
    os.environ["RECITER_DEVICE"] = args.device
    os.environ["RECITER_DTYPE"] = args.dtype
    if args.beams is None:
        os.environ.pop("RECITER_NUM_BEAMS", None)
    else:
        os.environ["RECITER_NUM_BEAMS"] = str(args.beams)
    os.environ["RECITER_ALLOW_EXPERIMENTAL_ADAPTER"] = "1" if args.model == "deepdml" else "0"
    print(f"Selected {args.model}: {path}. One model / one server worker; no fallback.", flush=True)
    if args.model == "deepdml":
        print("DeepDML is a private experimental recognition model, not certified learner/tajweed grading.", flush=True)
    # Import AFTER setting selection; a string import also keeps startup compatible
    # with uvicorn. Multiple workers would duplicate model memory on this 12GB GPU.
    import uvicorn
    uvicorn.run("src.streaming.server:app", host=args.host, port=args.port,
                workers=1, ws_max_size=16384)


if __name__ == "__main__":
    main()
