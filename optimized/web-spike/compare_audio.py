"""Side-by-side mel spectrograms and basic stats for the native and browser renders.

    python compare_audio.py parity/audio.bin out/browser_audio_t4.f32 out/spectrograms.png
Both inputs are float32 planar [2, n] (native_ref.py's audio.bin, the page's upload).
"""

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torchaudio  # noqa: E402

SR = 44100


def load(path):
    a = np.fromfile(path, dtype="<f4")
    return a.reshape(2, -1)


def stats(a):
    mono = a.mean(0)
    spec = np.abs(np.fft.rfft(mono)) ** 2
    freqs = np.fft.rfftfreq(mono.size, 1 / SR)
    return {
        "peak": float(np.abs(a).max()),
        "rms_db": float(20 * np.log10(np.sqrt((a**2).mean()) + 1e-12)),
        "centroid_hz": float((freqs * spec).sum() / spec.sum()),
        "clipped_pct": float((np.abs(a) >= 0.999).mean() * 100),
        "nan": bool(np.isnan(a).any()),
    }


def mel_db(a):
    mel = torchaudio.transforms.MelSpectrogram(SR, n_fft=2048, hop_length=512, n_mels=128)(torch.from_numpy(a.mean(0)))
    return 10 * torch.log10(mel + 1e-10).numpy()


def main(native_path, browser_path, out_png):
    native, browser = load(native_path), load(browser_path)
    s_n, s_b = stats(native), stats(browser)
    m_n, m_b = mel_db(native), mel_db(browser)
    band_n, band_b = m_n.mean(1), m_b.mean(1)  # average energy per mel band
    print("native ", s_n)
    print("browser", s_b)
    print("mean |mel-band energy diff| dB:", float(np.abs(band_n - band_b).mean()))

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    vmin, vmax = np.percentile(np.concatenate([m_n, m_b]), [5, 99.5])
    for ax, m, title in [(axes[0], m_n, "native TFLite (Python, int8)"), (axes[1], m_b, "browser (LiteRT.js WASM, int8)")]:
        ax.imshow(m, origin="lower", aspect="auto", vmin=vmin, vmax=vmax, cmap="magma",
                  extent=[0, m.shape[1] * 512 / SR, 0, m.shape[0]])
        ax.set_title(title)
        ax.set_ylabel("mel bin")
    axes[1].set_xlabel("seconds")
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    print("wrote", out_png)


if __name__ == "__main__":
    main(*sys.argv[1:4])
