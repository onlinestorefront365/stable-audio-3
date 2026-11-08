import argparse
import sys

import torch
from stable_audio_3.interface.diffusion_cond import create_diffusion_cond_ui
from stable_audio_3 import StableAudioModel
from stable_audio_3.verbose import set_verbose

# Silence Python warnings (FutureWarning, DeprecationWarning, etc.) unless --verbose.
# Must run before any ML library imports since most warnings fire at import time.
# We keep Gradio chatter, HF/torch progress bars, and generation tqdm intact.
if "--verbose" not in sys.argv:
    import os as _os

    _os.environ.setdefault("PYTHONWARNINGS", "ignore")
    import warnings as _warnings

    _warnings.filterwarnings("ignore")

# Flags that were accepted but never wired to anything. They fail loudly rather than
# silently loading the Hub model instead of the checkpoint the user asked for.
# Loading local checkpoints is tracked as P1.9 in ROADMAP.md.
_UNSUPPORTED_FLAGS = {
    "model_config": "--model-config",
    "ckpt_path": "--ckpt-path",
    "pretransform_ckpt_path": "--pretransform-ckpt-path",
}


def build_parser():
    parser = argparse.ArgumentParser(description="Run gradio interface")
    parser.add_argument(
        "--model", type=str, help="Name of pretrained model", required=True
    )
    parser.add_argument(
        "--model-config",
        type=str,
        help="Not supported yet (see ROADMAP.md P1.9)",
        required=False,
    )
    parser.add_argument(
        "--ckpt-path",
        type=str,
        help="Not supported yet (see ROADMAP.md P1.9)",
        required=False,
    )
    parser.add_argument(
        "--pretransform-ckpt-path",
        type=str,
        help="Not supported yet (see ROADMAP.md P1.9)",
        required=False,
    )
    parser.add_argument(
        "--share",
        action="store_true",
        default=False,
        help="Create a public gradio.live link. Anyone with the link can use your "
        "machine to generate; combine with --username/--password.",
    )
    parser.add_argument(
        "--server-name",
        type=str,
        default=None,
        help="Address to listen on (default: 127.0.0.1). Use 0.0.0.0 to accept "
        "connections from other machines.",
    )
    parser.add_argument(
        "--server-port", type=int, default=None, help="Port to listen on"
    )
    parser.add_argument(
        "--username",
        type=str,
        help="Require this username to open the UI (needs --password)",
        required=False,
    )
    parser.add_argument(
        "--password",
        type=str,
        help="Require this password to open the UI (needs --username)",
        required=False,
    )
    parser.add_argument(
        "--model-half",
        action=argparse.BooleanOptionalAction,
        help="Use half precision on CUDA (default: on; --no-model-half to disable)",
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
    return parser


def parse_args(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    for dest, flag in _UNSUPPORTED_FLAGS.items():
        if getattr(args, dest) is not None:
            parser.error(
                f"{flag} is not supported yet; only Hub models can be loaded "
                "(see ROADMAP.md P1.9)"
            )
    if (args.username is None) != (args.password is None):
        parser.error("--username and --password must be given together")
    return args


def launch_kwargs(args):
    """Keyword arguments for gr.Blocks.launch() derived from the CLI flags."""
    kwargs = {"share": args.share}
    if args.username is not None:
        kwargs["auth"] = (args.username, args.password)
    if args.server_name is not None:
        kwargs["server_name"] = args.server_name
    if args.server_port is not None:
        kwargs["server_port"] = args.server_port
    return kwargs


def main(args):
    set_verbose(getattr(args, "verbose", False))
    if args.share and args.username is None:
        print(
            "Warning: --share without --username/--password makes this UI usable by "
            "anyone who has the link."
        )
    torch.manual_seed(42)
    model_half = args.model_half
    model = StableAudioModel.from_pretrained(args.model, model_half=model_half)
    if args.lora_ckpt_path:
        model.load_lora(args.lora_ckpt_path)
    interface = create_diffusion_cond_ui(
        model,
        gradio_title=args.title if args.title is not None else "Stable Audio 3",
        default_prompt=args.default_prompt,
    )
    interface.queue()
    interface.launch(
        js=getattr(interface, "_sao_js", None),
        theme=getattr(interface, "_sao_theme", None),
        **launch_kwargs(args),
    )


if __name__ == "__main__":
    main(parse_args())
