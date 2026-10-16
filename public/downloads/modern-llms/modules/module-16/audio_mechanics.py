"""Audio framing, temporal pooling and a tiny residual quantizer, all on CPU.

The spectra and hand-written codebooks are teaching examples, not a speech model
or a trained audio codec. Codec-rate accounting is a separate declared scenario.
"""
from numbers import Integral
import math
import numpy as np


def count(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(name + ' must be a positive integer')
    return int(value)


def frame_signal(waveform, window=400, hop=160):
    """Start a window at 0, hop, ... < length; right-pad the last windows."""
    signal = np.asarray(waveform)
    window, hop = count(window, 'window'), count(hop, 'hop')
    if signal.ndim != 1 or signal.size == 0 or signal.dtype.kind not in 'fiu':
        raise ValueError('waveform must be a nonempty real numeric vector')
    if not np.isfinite(signal).all():
        raise ValueError('waveform must be finite')
    starts = np.arange(0, len(signal), hop)
    indices = starts[:, None] + np.arange(window)[None, :]
    valid = indices < len(signal)
    frames = np.zeros(indices.shape, dtype=np.float64)
    frames[valid] = signal[indices[valid]]
    return frames, valid


def pool_frames(features, factor=4):
    """Average consecutive frame vectors; the final short group uses its own size."""
    x = np.asarray(features, dtype=np.float64)
    factor = count(factor, 'factor')
    if x.ndim != 2 or 0 in x.shape or not np.isfinite(x).all():
        raise ValueError('features must be a finite nonempty matrix')
    return np.stack([x[i:i+factor].mean(axis=0) for i in range(0, len(x), factor)])


def residual_quantize(features, codebooks):
    """Greedily encode residuals; books have shape [books, entries, width]."""
    x = np.asarray(features, dtype=np.float64)
    books = np.asarray(codebooks, dtype=np.float64)
    if x.ndim != 2 or 0 in x.shape or books.ndim != 3 or 0 in books.shape:
        raise ValueError('nonempty feature matrix and codebook tensor required')
    if x.shape[1] != books.shape[2] or not np.isfinite(x).all() or not np.isfinite(books).all():
        raise ValueError('widths must agree and values must be finite')
    residual = x.copy()
    reconstructed = np.zeros_like(x)
    ids, squared_errors = [], [np.square(residual).sum(axis=1)]
    for book in books:
        distances = np.square(residual[:, None, :] - book[None, :, :]).sum(axis=-1)
        chosen = distances.argmin(axis=1)
        reconstructed += book[chosen]
        residual -= book[chosen]
        ids.append(chosen)
        squared_errors.append(np.square(residual).sum(axis=1))
    return np.stack(ids, axis=1), reconstructed, np.stack(squared_errors, axis=1)


def audio_budget(n_samples=96000, sample_rate=16000, hop=160, pooling=4,
                 codec_rate=75, books=8, entries=1024):
    """Two separate paths: acoustic features and illustrative discrete codec codes."""
    n_samples, sample_rate, hop, pooling, codec_rate, books, entries = (
        count(v,n) for v,n in zip((n_samples,sample_rate,hop,pooling,codec_rate,books,entries),
                                ('samples','sample_rate','hop','pooling','codec_rate','books','entries')))
    if entries < 2 or entries & (entries-1):
        raise ValueError('entries must be a power of two >= 2 for fixed-width packing')
    frames = (n_samples+hop-1)//hop
    codec_frames = (n_samples*codec_rate+sample_rate-1)//sample_rate
    ids = codec_frames*books
    return dict(samples=n_samples, seconds=n_samples/sample_rate,
                acoustic_frames=frames, pooled_positions=(frames+pooling-1)//pooling,
                codec_frames=codec_frames, codec_ids=ids,
                codec_payload_bits=ids*(entries.bit_length()-1),
                codec_bits_per_second=codec_rate*books*(entries.bit_length()-1))


def run_audio_example():
    t = np.arange(96000)/16000
    wave = .6*np.sin(2*np.pi*220*t) + .2*np.sin(2*np.pi*660*t)
    frames, valid = frame_signal(wave)
    # Simple magnitude spectra: not log-mel features or a trained audio encoder.
    features = np.abs(np.fft.rfft(frames*np.hanning(400), axis=1))
    pooled = pool_frames(features)
    assert frames.shape == (600,400) and features.shape == (600,201)
    assert pooled.shape == (150,201) and valid[-1].sum() == 160
    budget = audio_budget()
    assert budget['acoustic_frames'] == len(frames)
    assert budget['pooled_positions'] == len(pooled)
    books = np.array([[[0,0],[1,0],[0,-1]], [[0,0],[.5,0],[.5,-.5]]], dtype=float)
    ids, reconstructed, errors = residual_quantize([[1.6,-.6]], books)
    assert ids.tolist() == [[1,2]]
    np.testing.assert_allclose(reconstructed, [[1.5,-.5]], atol=1e-15)
    np.testing.assert_allclose(errors, [[2.92,.72,.02]], atol=1e-15)
    # Zero is available in each book, so this toy can never increase residual norm.
    assert np.all(np.diff(errors, axis=1) <= 0)
    # Illustrative timings; no speech decoder is measured by this script.
    rtf = 2.4/6
    assert math.isclose(rtf,.4)
    result = dict(budget=budget, spectral_shape=list(features.shape),
                  pooled_shape=list(pooled.shape), rvq_ids=ids.tolist(),
                  rvq_squared_errors=errors.tolist(), illustrative_rtf=rtf,
                  illustrative_first_audio_seconds=.18)
    return result


if __name__ == '__main__':
    import json
    print(json.dumps(run_audio_example(),indent=2))
