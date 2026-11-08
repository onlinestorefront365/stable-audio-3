"""Model-name validation in StableAudioModel.from_pretrained (no downloads)."""

from unittest.mock import patch

import pytest

from stable_audio_3.model import StableAudioModel


@pytest.mark.parametrize("name", ["same-s", "same-l"])
def test_autoencoder_names_are_rejected_before_downloading(name):
    with patch("stable_audio_3.model_configs.hf_hub_download") as download:
        with pytest.raises(ValueError, match="AutoencoderModel.from_pretrained"):
            StableAudioModel.from_pretrained(name, device="cpu")
    download.assert_not_called()


def test_unknown_name_lists_generation_models_only():
    with pytest.raises(ValueError, match="Unknown model") as err:
        StableAudioModel.from_pretrained("not-a-model", device="cpu")
    assert "medium" in str(err.value)
    assert "same-s" not in str(err.value)
