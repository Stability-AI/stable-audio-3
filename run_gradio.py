import torch
from stable_audio_3.interface.diffusion_cond import create_diffusion_cond_ui
from stable_audio_3.pipeline import StableAudioPipeline
from stable_audio_3.verbose import set_verbose
import sys

# Silence Python warnings (FutureWarning, DeprecationWarning, etc.) unless --verbose.
# Must run before any ML library imports since most warnings fire at import time.
# We keep Gradio chatter, HF/torch progress bars, and generation tqdm intact.
if "--verbose" not in sys.argv:
    import os as _os

    _os.environ.setdefault("PYTHONWARNINGS", "ignore")
    import warnings as _warnings

    _warnings.filterwarnings("ignore")


def main(args):
    set_verbose(getattr(args, "verbose", False))
    torch.manual_seed(42)
    model_half = args.model_half
    pipe = StableAudioPipeline.from_pretrained(args.model, model_half=model_half)
    if args.lora_ckpt_path:
        pipe.load_lora(args.lora_ckpt_path)
    interface = create_diffusion_cond_ui(
        pipe,
        gradio_title=args.title if args.title is not None else "Stable Audio 3",
        default_prompt=args.default_prompt,
    )
    interface.queue()
    interface.launch(
        share=True,
        js=getattr(interface, "_sao_js", None),
        theme=getattr(interface, "_sao_theme", None),
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run gradio interface")
    parser.add_argument(
        "--model", type=str, help="Name of pretrained model", required=False
    )
    parser.add_argument(
        "--model-config", type=str, help="Path to model config", required=False
    )
    parser.add_argument(
        "--ckpt-path", type=str, help="Path to model checkpoint", required=False
    )
    parser.add_argument(
        "--pretransform-ckpt-path",
        type=str,
        help="Optional to model pretransform checkpoint",
        required=False,
    )
    parser.add_argument("--username", type=str, help="Gradio username", required=False)
    parser.add_argument("--password", type=str, help="Gradio password", required=False)
    parser.add_argument(
        "--model-half",
        action="store_true",
        help="Whether to use half precision",
        required=False,
        default=True,
    )
    parser.add_argument(
        "--title", type=str, help="Display Title top of Gradio", required=False
    )
    parser.add_argument(
        "--lora-ckpt-path",
        type=str,
        nargs="*",
        help="Path(s) for LoRA(s) to apply. Can specify multiple.",
        required=False,
    )
    parser.add_argument(
        "--default-prompt",
        type=str,
        default=None,
        help="Default prompt to pre-fill in the textbox",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Print detailed load/generation progress",
    )
    args = parser.parse_args()
    main(args)
