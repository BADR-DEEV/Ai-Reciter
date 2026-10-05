"""Start the chosen local ASR models (named engines, one worker); selecting DeepDML opts into private staging."""
import argparse
import os
from .model_options import MODEL_PRESETS, check_model, default_models, default_name, parse_models, model_spec, with_tajweed_slot


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", default=[], metavar="[NAME=]PRESET|PATH",
                        help=f"Repeatable. Presets: {', '.join(MODEL_PRESETS)}. A bare preset is named plain "
                             "(tajweed for a tajweed preset); NAME=PATH or NAME=PRESET:PATH names a directory")
    parser.add_argument("--models", help="Comma-separated NAME=PRESET|PATH list, e.g. plain=rattil-v3,tajweed=runs/my_tajweed. "
                                         "Default: $RECITER_MODELS, else plain=rattil-v3. A tajweed engine is always offered "
                                         "(rattil-tajweed-v1 unless set) and reported unavailable until its directory exists")
    parser.add_argument("--model-path", help="Override the directory of a single --model preset; DeepDML also accepts the parent run")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--dtype", choices=["auto", "fp16", "bf16", "fp32"], default="auto")
    parser.add_argument("--beams", type=int, choices=[1, 3, 5], help="Override decoding: 1 is faster; recognition may change. Default: saved adapter policy / full-model greedy")
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    try:
        specs = parse_models(args.model + ([args.models] if args.models else []))
        if args.model_path:
            if len(args.model) != 1 or args.models or len(specs) != 1:
                parser.error("--model-path overrides exactly one --model preset")
            spec = next(iter(specs.values()))
            if spec.preset is None:
                parser.error("--model-path needs a preset --model")
            specs = {spec.name: model_spec(spec.name, f"{spec.preset}:{args.model_path}")}
        if not specs and os.environ.get("RECITER_MODELS"):
            specs = parse_models([os.environ["RECITER_MODELS"]])
    except ValueError as exc:
        parser.error(str(exc))
    specs = specs or default_models()
    if not specs:
        parser.error("Choose --model: rattil-v3 is not downloaded (python src/deployment/pull_hf_assets.py)")
    args.specs = with_tajweed_slot(specs)
    return args


def main(argv=None):
    args = parse_args(argv)
    default = default_name(args.specs)
    # Fail before starting: every engine must exist except an optional tajweed one.
    paths = {name: check_model(spec, required=name == default) for name, spec in args.specs.items()}
    specs = {name: model_spec(name, f"{spec.preset}:{paths[name]}" if spec.preset else str(paths[name])) if paths[name] else spec
             for name, spec in args.specs.items()}
    os.environ["RECITER_MODELS"] = ",".join(spec.entry() for spec in specs.values())
    os.environ["RECITER_MODEL_PATH"] = str(paths[default])
    os.environ["RECITER_MODEL_PRESET"] = specs[default].label
    os.environ["RECITER_DEVICE"] = args.device
    os.environ["RECITER_DTYPE"] = args.dtype
    if args.beams is None:
        os.environ.pop("RECITER_NUM_BEAMS", None)
    else:
        os.environ["RECITER_NUM_BEAMS"] = str(args.beams)
    deepdml = any(spec.preset == "deepdml" for spec in specs.values())
    os.environ["RECITER_ALLOW_EXPERIMENTAL_ADAPTER"] = "1" if deepdml else "0"
    for name, spec in specs.items():
        state = paths[name] or f"{spec.path} not ready: reported unavailable, plain recognition is used instead"
        print(f"Selected {name}{' (default)' if name == default else ''} = {spec.label}: {state}", flush=True)
    print("One server worker; requests choose an engine by name. No fallback for plain models.", flush=True)
    if deepdml:
        print("DeepDML is a private experimental recognition model, not certified learner/tajweed grading.", flush=True)
    # Import AFTER setting selection; a string import also keeps startup compatible
    # with uvicorn. Multiple workers would duplicate model memory on this 12GB GPU.
    import uvicorn
    uvicorn.run("src.streaming.server:app", host=args.host, port=args.port,
                workers=1, ws_max_size=16384)


if __name__ == "__main__":
    main()
