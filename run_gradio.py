import torch
from stable_audio_3.interface.diffusion_cond import create_diffusion_cond_ui
from stable_audio_3.pipeline import StableAudioPipeline


def main(args):
    torch.manual_seed(42)
    model_half = args.model_half
    pipe = StableAudioPipeline.from_pretrained(
        args.model, model_half=model_half
    )
    if args.lora_ckpt_path:
        pipe.load_lora(args.lora_ckpt_path)
    interface = create_diffusion_cond_ui(
        pipe,
        gradio_title=args.title if args.title is not None else "Stable Audio 3",
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
    args = parser.parse_args()
    main(args)
