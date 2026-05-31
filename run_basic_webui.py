import argparse

from stable_audio_3.interface.basic_cli_webui import launch_basic_cli_webui


def main():
    parser = argparse.ArgumentParser(description="Run basic Stable Audio CLI-parity WebUI")
    parser.add_argument(
        "--model",
        default="medium",
        choices=[
            "medium",
            "small-music",
            "small-sfx",
            "medium-base",
            "small-music-base",
            "small-sfx-base",
        ],
        help="Model to load",
    )
    parser.add_argument("--device", default=None, help="cuda/mps/cpu (optional)")
    parser.add_argument("--no-half", action="store_true", help="Disable fp16 on CUDA")

    args = parser.parse_args()
    launch_basic_cli_webui(model_name=args.model, device=args.device, no_half=args.no_half)


if __name__ == "__main__":
    main()
