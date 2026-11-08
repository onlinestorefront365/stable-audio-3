import os
import shutil
from unittest.mock import patch

import pytest
import torch

from stable_audio_3.interface.diffusion_cond import (
    FFMPEG_FORMAT_ARGS,
    condense_prompt,
    export_audio,
    parse_inpaint_regions,
)


def test_parse_single_inpaint_region():
    assert parse_inpaint_regions("4", "8") == (4.0, 8.0)


def test_parse_multiple_inpaint_regions():
    assert parse_inpaint_regions("4, 16", "8, 20") == (
        [4.0, 16.0],
        [8.0, 20.0],
    )


def test_parse_empty_inpaint_regions():
    assert parse_inpaint_regions("", "") == (None, None)


@pytest.mark.parametrize(
    ("starts", "ends"),
    [("4", ""), ("4, 16", "8"), ("8", "4"), ("-1", "4")],
)
def test_parse_invalid_inpaint_regions(starts, ends):
    with pytest.raises(Exception):
        parse_inpaint_regions(starts, ends)


# --- export_audio: output files and ffmpeg conversion ---


SAMPLE_RATE = 44100
HAS_FFMPEG = shutil.which("ffmpeg") is not None


def _silence(seconds=0.1):
    return torch.zeros(2, int(SAMPLE_RATE * seconds), dtype=torch.int16)


@pytest.fixture
def exported_dirs():
    dirs = []
    yield dirs
    for d in dirs:
        shutil.rmtree(d, ignore_errors=True)


def test_export_wav_writes_into_its_own_temp_dir(exported_dirs):
    path, out_dir = export_audio(_silence(), SAMPLE_RATE, "output", "wav")
    exported_dirs.append(out_dir)
    assert os.path.isabs(path)
    assert os.path.dirname(path) == out_dir
    assert os.path.basename(path) == "output.wav"
    assert os.path.isfile(path)


def test_export_concurrent_calls_do_not_share_files(exported_dirs):
    path_a, dir_a = export_audio(_silence(), SAMPLE_RATE, "output", "wav")
    path_b, dir_b = export_audio(_silence(), SAMPLE_RATE, "output", "wav")
    exported_dirs.extend([dir_a, dir_b])
    assert dir_a != dir_b
    assert path_a != path_b


def test_export_unknown_format_falls_back_to_wav(exported_dirs):
    with patch("stable_audio_3.interface.diffusion_cond.subprocess.run") as run:
        path, out_dir = export_audio(_silence(), SAMPLE_RATE, "output", "ogg")
    exported_dirs.append(out_dir)
    run.assert_not_called()
    assert path.endswith("output.wav") and os.path.isfile(path)


@pytest.mark.parametrize("file_format", sorted(FFMPEG_FORMAT_ARGS))
def test_export_runs_ffmpeg_without_a_shell(file_format, exported_dirs):
    basename = "x $(touch pwned) `id` ; rm -rf ~"
    with patch("stable_audio_3.interface.diffusion_cond.subprocess.run") as run:
        path, out_dir = export_audio(_silence(), SAMPLE_RATE, basename, file_format)
    exported_dirs.append(out_dir)
    run.assert_called_once()
    args, kwargs = run.call_args
    cmd = args[0]
    assert isinstance(cmd, list), "ffmpeg must be run with an argument list"
    assert not kwargs.get("shell", False)
    # The prompt-derived names are single, absolute arguments.
    assert cmd[-1] == path and os.path.isabs(path)
    assert os.path.basename(path).startswith(basename)
    assert os.path.join(out_dir, basename + ".wav") in cmd


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg not installed")
def test_export_prompt_with_shell_syntax_is_not_executed(
    tmp_path, exported_dirs, monkeypatch
):
    # Under a shell, either substitution would create ./pwned in the working directory.
    monkeypatch.chdir(tmp_path)
    marker = tmp_path / "pwned"
    basename = condense_prompt("beat $(touch pwned) `touch pwned`")
    path, out_dir = export_audio(_silence(), SAMPLE_RATE, basename, "flac")
    exported_dirs.append(out_dir)
    assert not marker.exists()
    assert path.endswith(".flac") and os.path.isfile(path)


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg not installed")
def test_export_basename_starting_with_dash(exported_dirs):
    path, out_dir = export_audio(_silence(), SAMPLE_RATE, "-version", "mp3 128k")
    exported_dirs.append(out_dir)
    assert os.path.basename(path) == "-version.mp3"
    assert os.path.getsize(path) > 0
