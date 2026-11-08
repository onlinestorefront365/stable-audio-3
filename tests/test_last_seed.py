"""StableAudioModel.generate() records the seed it used in last_seed.

The diffusion model and sampler are mocked, so this runs without weights.
"""

from unittest.mock import MagicMock, patch

import pytest
import torch

from stable_audio_3.model import StableAudioModel

SAMPLE_RATE = 44100


@pytest.fixture
def model():
    inner = MagicMock()
    inner.pretransform = None
    inner.io_channels = 2
    inner.sample_rate = SAMPLE_RATE
    inner.model.parameters.side_effect = lambda: iter([torch.zeros(1)])
    inner.get_conditioning_inputs.return_value = {}
    sam = StableAudioModel.__new__(StableAudioModel)
    sam.model = inner
    sam.model_config = {}
    sam.device = "cpu"
    sam.model_half = False
    sam.last_seed = None
    return sam


def _generate(model, seed):
    with patch(
        "stable_audio_3.model.sample_diffusion",
        side_effect=lambda **kw: torch.zeros(1, 2, SAMPLE_RATE),
    ) as sampler:
        model.generate(
            prompt="test", duration=1, seed=seed, sample_size=SAMPLE_RATE * 8
        )
    return sampler.call_args.kwargs["noise"]


def test_explicit_seed_is_recorded(model):
    _generate(model, seed=123)
    assert model.last_seed == 123


def test_random_seed_is_recorded_and_reproduces_the_noise(model):
    noise_random = _generate(model, seed=-1)
    seed = model.last_seed
    assert isinstance(seed, int) and 0 <= seed < 99999

    noise_again = _generate(model, seed=seed)
    assert torch.equal(noise_random, noise_again)
