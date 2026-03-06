"""Tests for the preference model bootstrap heuristic."""

import numpy as np
import pytest


def _librosa_features_available() -> bool:
    """Check if librosa spectral features work (may fail with numba/coverage conflict)."""
    try:
        import librosa
        import numpy as _np

        y = _np.zeros(4096, dtype=_np.float32)
        librosa.feature.spectral_contrast(y=y, sr=32000)
        return True
    except Exception:
        return False


_skip_librosa = pytest.mark.skipif(
    not _librosa_features_available(),
    reason="librosa features unavailable (numba/coverage conflict)",
)


@pytest.fixture
def model():
    from tuneforge.scoring.preference_model import PreferenceModel

    return PreferenceModel(model_path=None)  # bootstrap mode


class TestPreferenceModelBootstrap:
    """Verify the model runs in bootstrap mode when no model path is given."""

    def test_bootstrap_flag_set(self, model):
        assert model._bootstrap is True

    def test_no_head_loaded(self, model):
        assert model._head is None


@_skip_librosa
class TestHeuristicScoreRange:
    """All audio types produce scores in [0, 1]."""

    def test_sine_in_range(self, model, sample_audio_sine, sample_rate):
        score = model.score(sample_audio_sine, sample_rate)
        assert 0.0 <= score <= 1.0

    def test_complex_in_range(self, model, sample_audio_complex, sample_rate):
        score = model.score(sample_audio_complex, sample_rate)
        assert 0.0 <= score <= 1.0

    def test_noise_in_range(self, model, sample_audio_noise, sample_rate):
        score = model.score(sample_audio_noise, sample_rate)
        assert 0.0 <= score <= 1.0

    def test_silence_in_range(self, model, sample_audio_silence, sample_rate):
        score = model.score(sample_audio_silence, sample_rate)
        assert 0.0 <= score <= 1.0


class TestBootstrapReturnsNeutral:
    """Bootstrap mode returns 0.5 for all inputs (no double-counting with other scorers)."""

    def test_silence_returns_neutral(self, model, sample_audio_silence, sample_rate):
        score = model.score(sample_audio_silence, sample_rate)
        assert score == 0.5

    def test_complex_returns_neutral(self, model, sample_audio_complex, sample_rate):
        score = model.score(sample_audio_complex, sample_rate)
        assert score == 0.5

    def test_noise_returns_neutral(self, model, sample_audio_noise, sample_rate):
        score = model.score(sample_audio_noise, sample_rate)
        assert score == 0.5


class TestDualPreferenceHead:
    """Tests for the DualPreferenceHead architecture."""

    def test_input_output_shapes(self):
        import torch
        from tuneforge.scoring.preference_model import DualPreferenceHead

        head = DualPreferenceHead()
        x = torch.randn(4, 1280)  # batch of 4, 1280-dim input
        out = head(x)
        assert out.shape == (4, 1)
        # Output is raw logit (no Sigmoid — applied at inference in PreferenceModel)
        assert out.requires_grad

    def test_single_sample(self):
        import torch
        from tuneforge.scoring.preference_model import DualPreferenceHead

        head = DualPreferenceHead()
        x = torch.randn(1, 1280)
        out = head(x)
        assert out.shape == (1, 1)

    def test_wrong_input_dim_raises(self):
        import torch
        from tuneforge.scoring.preference_model import DualPreferenceHead

        head = DualPreferenceHead()
        with pytest.raises(RuntimeError):
            head(torch.randn(1, 512))  # Wrong input size


class TestPreferenceWeightScaler:
    """Tests for PreferenceWeightScaler."""

    def test_none_accuracy_returns_min_weight(self):
        from tuneforge.scoring.preference_model import PreferenceWeightScaler

        scaler = PreferenceWeightScaler(min_weight=0.02, max_weight=0.10)
        assert scaler.get_scaled_weight() == pytest.approx(0.02)

    def test_min_accuracy_returns_min_weight(self):
        from tuneforge.scoring.preference_model import PreferenceWeightScaler

        scaler = PreferenceWeightScaler(min_weight=0.02, max_weight=0.10, min_accuracy=0.55)
        scaler.update_accuracy(0.55)
        assert scaler.get_scaled_weight() == pytest.approx(0.02)

    def test_max_accuracy_returns_max_weight(self):
        from tuneforge.scoring.preference_model import PreferenceWeightScaler

        scaler = PreferenceWeightScaler(min_weight=0.02, max_weight=0.10, max_accuracy=0.80)
        scaler.update_accuracy(0.80)
        assert scaler.get_scaled_weight() == pytest.approx(0.10)

    def test_mid_accuracy_interpolates(self):
        from tuneforge.scoring.preference_model import PreferenceWeightScaler

        scaler = PreferenceWeightScaler(
            min_weight=0.0, max_weight=1.0, min_accuracy=0.0, max_accuracy=1.0
        )
        scaler.update_accuracy(0.5)
        assert scaler.get_scaled_weight() == pytest.approx(0.5, abs=0.01)

    def test_below_min_accuracy_clamps(self):
        from tuneforge.scoring.preference_model import PreferenceWeightScaler

        scaler = PreferenceWeightScaler(min_weight=0.02, max_weight=0.10, min_accuracy=0.55)
        scaler.update_accuracy(0.40)
        assert scaler.get_scaled_weight() == pytest.approx(0.02)

    def test_above_max_accuracy_clamps(self):
        from tuneforge.scoring.preference_model import PreferenceWeightScaler

        scaler = PreferenceWeightScaler(min_weight=0.02, max_weight=0.10, max_accuracy=0.80)
        scaler.update_accuracy(0.99)
        assert scaler.get_scaled_weight() == pytest.approx(0.10)


class TestCheckpointFormatDetection:
    """Tests for auto-detecting checkpoint format (dual vs single)."""

    def test_legacy_single_checkpoint(self, tmp_path, monkeypatch):
        """Legacy raw state_dict with 512-dim input loads as single mode."""
        import torch
        from tuneforge.scoring.preference_model import PreferenceHead, PreferenceModel

        # Prevent CUDA OOM by forcing CPU
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

        head = PreferenceHead()
        path = str(tmp_path / "legacy.pt")
        torch.save(head.state_dict(), path)

        m = PreferenceModel(model_path=path)
        assert m._bootstrap is False
        assert m._dual_mode is False
        assert isinstance(m._head, PreferenceHead)

    def test_new_format_single_checkpoint(self, tmp_path, monkeypatch):
        """New checkpoint format with 512-dim loads as single mode."""
        import torch
        from tuneforge.scoring.preference_model import PreferenceHead, PreferenceModel

        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

        head = PreferenceHead()
        path = str(tmp_path / "new_single.pt")
        torch.save(
            {"state_dict": head.state_dict(), "val_accuracy": 0.72, "embedding_dim": 512},
            path,
        )

        m = PreferenceModel(model_path=path)
        assert m._bootstrap is False
        assert m._dual_mode is False
        assert m.get_scaled_weight() > 0.02  # accuracy loaded

    def test_dual_checkpoint(self, tmp_path, monkeypatch):
        """Checkpoint with 1280-dim input loads as dual mode."""
        import torch
        from tuneforge.scoring.preference_model import DualPreferenceHead, PreferenceModel

        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

        head = DualPreferenceHead()
        path = str(tmp_path / "dual.pt")
        torch.save(
            {"state_dict": head.state_dict(), "val_accuracy": 0.75, "embedding_dim": 1280},
            path,
        )

        m = PreferenceModel(model_path=path)
        assert m._bootstrap is False
        assert m._dual_mode is True
        assert isinstance(m._head, DualPreferenceHead)

    def test_get_scaled_weight_bootstrap(self):
        """Bootstrap model returns min weight."""
        from tuneforge.scoring.preference_model import PreferenceModel

        m = PreferenceModel(model_path=None)
        # No accuracy loaded, should return min_weight (0.02)
        assert m.get_scaled_weight() == pytest.approx(0.02)
