"""Unit tests for run_gradio.py's flag handling.

The model and UI are mocked, so these run without downloading weights.
"""

import importlib.util
import os
import warnings
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

RUN_GRADIO = Path(__file__).resolve().parent.parent / "run_gradio.py"


@pytest.fixture(scope="module")
def run_gradio():
    # run_gradio.py silences warnings at import time; keep that out of the rest of the suite.
    saved_env = os.environ.get("PYTHONWARNINGS")
    with warnings.catch_warnings():
        spec = importlib.util.spec_from_file_location("run_gradio", RUN_GRADIO)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    if saved_env is None:
        os.environ.pop("PYTHONWARNINGS", None)
    else:
        os.environ["PYTHONWARNINGS"] = saved_env
    return module


def test_default_is_local_only_without_auth(run_gradio):
    args = run_gradio.parse_args(["--model", "small-music"])
    assert run_gradio.launch_kwargs(args) == {"share": False}


def test_share_is_opt_in(run_gradio):
    args = run_gradio.parse_args(["--model", "small-music", "--share"])
    assert run_gradio.launch_kwargs(args)["share"] is True


def test_username_and_password_are_passed_as_auth(run_gradio):
    args = run_gradio.parse_args(
        ["--model", "small-music", "--username", "alice", "--password", "s3cret"]
    )
    assert run_gradio.launch_kwargs(args)["auth"] == ("alice", "s3cret")


@pytest.mark.parametrize("flags", [["--username", "alice"], ["--password", "s3cret"]])
def test_username_and_password_must_be_given_together(run_gradio, flags):
    with pytest.raises(SystemExit):
        run_gradio.parse_args(["--model", "small-music", *flags])


def test_server_name_and_port(run_gradio):
    args = run_gradio.parse_args(
        ["--model", "small-music", "--server-name", "0.0.0.0", "--server-port", "7861"]
    )
    kwargs = run_gradio.launch_kwargs(args)
    assert kwargs["server_name"] == "0.0.0.0"
    assert kwargs["server_port"] == 7861


def test_model_half_defaults_on_and_can_be_disabled(run_gradio):
    assert run_gradio.parse_args(["--model", "medium"]).model_half is True
    assert (
        run_gradio.parse_args(["--model", "medium", "--no-model-half"]).model_half
        is False
    )


@pytest.mark.parametrize(
    "flag", ["--model-config", "--ckpt-path", "--pretransform-ckpt-path"]
)
def test_unsupported_checkpoint_flags_fail_loudly(run_gradio, flag):
    with pytest.raises(SystemExit):
        run_gradio.parse_args(["--model", "medium", flag, "some/path"])


def test_main_launches_with_auth_and_requested_precision(run_gradio):
    args = run_gradio.parse_args(
        [
            "--model",
            "small-music",
            "--no-model-half",
            "--share",
            "--username",
            "alice",
            "--password",
            "s3cret",
        ]
    )
    interface = MagicMock()
    with (
        patch.object(run_gradio, "StableAudioModel") as model_cls,
        patch.object(run_gradio, "create_diffusion_cond_ui", return_value=interface),
    ):
        run_gradio.main(args)

    model_cls.from_pretrained.assert_called_once_with("small-music", model_half=False)
    launch = interface.launch.call_args.kwargs
    assert launch["share"] is True
    assert launch["auth"] == ("alice", "s3cret")
