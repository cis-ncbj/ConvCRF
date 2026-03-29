"""
Smoke tests for ConvCRF forward pass reproducibility.

These tests compare model output against saved reference tensors to detect
regressions when migrating to newer PyTorch versions. Reference tensors were
generated with PyTorch 2.6.0a0+df5bbc09d1.nv24.12 on CPU using fixed seeds.

Run with:
    cd contrib/ConvCRF && python -m pytest tests/ -v
"""

import math
import os
import sys
import warnings

import numpy as np
import pytest
import torch
import torch.nn.functional as F

# Ensure the ConvCRF package root is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from convcrf import convcrf
from utils import test_utils

REFERENCE_DIR = os.path.join(os.path.dirname(__file__))

# Tolerance: operations are float32 on CPU, deterministic with fixed seeds.
# 1e-6 is tight but appropriate for identical code paths.
ATOL = 1e-6
RTOL = 1e-5


def _prepare_inputs(nclasses=2, shape=(10, 10)):
    """Return (img_tensor, unary_tensor) with batch dim, deterministic."""
    torch.manual_seed(0)
    np.random.seed(0)

    img_np = test_utils._get_simple_img()  # (10, 10, 3) uint8
    img_t = torch.from_numpy(
        img_np.transpose(2, 0, 1).astype(np.float32)
    ).unsqueeze(0)  # (1, 3, H, W)

    unary_np = test_utils._get_simple_unary()  # (nclasses, H*W)
    unary_t = torch.from_numpy(
        unary_np.reshape(nclasses, shape[0], shape[1]).astype(np.float32)
    ).unsqueeze(0)  # (1, nclasses, H, W)

    return img_t, unary_t


def _run_forward(conf, nclasses=2, shape=(10, 10), num_iter=5):
    """Build GaussCRF with conf, run forward on test data, return output numpy array."""
    img_t, unary_t = _prepare_inputs(nclasses, shape)
    model = convcrf.GaussCRF(conf=conf, shape=shape, nclasses=nclasses, use_gpu=False)
    model.eval()
    with torch.no_grad():
        out = model.forward(unary=unary_t, img=img_t, num_iter=num_iter)
    return out.numpy()


def _load_reference(name):
    path = os.path.join(REFERENCE_DIR, f"reference_{name}.npy")
    assert os.path.exists(path), f"Reference file not found: {path}"
    return np.load(path)


# ---------------------------------------------------------------------------
# Test 1: test_config (blur=1, merge=False, norm=sym)
#   Exercises: ConvCRF inference, symmetric normalization, F.unfold im2col,
#   log_softmax, softmax, scalar weight.
# ---------------------------------------------------------------------------
class TestTestConfig:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.conf = convcrf.get_test_conf()

        self.conf["final_softmax"] = False
        self.reference = _load_reference("test_config")

    def test_output_shape(self):
        out = _run_forward(self.conf)
        assert out.shape == (1, 2, 10, 10)

    def test_output_matches_reference(self):
        out = _run_forward(self.conf)
        np.testing.assert_allclose(out, self.reference, atol=ATOL, rtol=RTOL)

    def test_deterministic(self):
        """Two consecutive runs must produce identical output."""
        out1 = _run_forward(self.conf)
        out2 = _run_forward(self.conf)
        np.testing.assert_array_equal(out1, out2)

    def test_output_finite(self):
        out = _run_forward(self.conf)
        assert np.all(np.isfinite(out))


# ---------------------------------------------------------------------------
# Test 2: test_config with blur=2 (exercises upsample/interpolate path)
#   This is the critical path for the upsample -> interpolate migration.
#   The blur>1 branch in _compute_gaussian calls F.upsample (bilinear).
# ---------------------------------------------------------------------------
class TestBlur2Config:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.conf = convcrf.get_test_conf()

        self.conf["final_softmax"] = False
        self.conf["blur"] = 2
        self.reference = _load_reference("test_config_blur2")

    def test_output_shape(self):
        out = _run_forward(self.conf)
        assert out.shape == (1, 2, 10, 10)

    def test_output_matches_reference(self):
        out = _run_forward(self.conf)
        np.testing.assert_allclose(out, self.reference, atol=ATOL, rtol=RTOL)

    def test_deterministic(self):
        out1 = _run_forward(self.conf)
        out2 = _run_forward(self.conf)
        np.testing.assert_array_equal(out1, out2)

    def test_output_finite(self):
        out = _run_forward(self.conf)
        assert np.all(np.isfinite(out))


# ---------------------------------------------------------------------------
# Test 3: default_config (merge=True, blur=4, vector weight)
#   Exercises: merged gaussian filters, larger blur with bilinear upsample,
#   vector-valued weight, no normalization.
# ---------------------------------------------------------------------------
class TestDefaultConfig:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.conf = convcrf.get_default_conf()

        self.conf["filter_size"] = 5  # 11 is too large for 10x10 test input
        self.reference = _load_reference("default_config")

    def test_output_shape(self):
        out = _run_forward(self.conf)
        assert out.shape == (1, 2, 10, 10)

    def test_output_matches_reference(self):
        out = _run_forward(self.conf)
        np.testing.assert_allclose(out, self.reference, atol=ATOL, rtol=RTOL)

    def test_deterministic(self):
        out1 = _run_forward(self.conf)
        out2 = _run_forward(self.conf)
        np.testing.assert_array_equal(out1, out2)

    def test_output_finite(self):
        out = _run_forward(self.conf)
        assert np.all(np.isfinite(out))


# ---------------------------------------------------------------------------
# Test 4: Structural / smoke tests (not config-specific)
# ---------------------------------------------------------------------------
class TestStructural:
    def test_gausscrf_is_nn_module(self):
        conf = convcrf.get_test_conf()

        conf["final_softmax"] = False
        model = convcrf.GaussCRF(conf=conf, shape=(10, 10), nclasses=2, use_gpu=False)
        assert isinstance(model, torch.nn.Module)

    def test_convcrf_is_nn_module(self):
        conf = convcrf.get_test_conf()

        conf["final_softmax"] = False
        model = convcrf.GaussCRF(conf=conf, shape=(10, 10), nclasses=2, use_gpu=False)
        assert isinstance(model.CRF, torch.nn.Module)

    def test_num_iter_affects_output(self):
        """More iterations should produce a different (more refined) output."""
        conf = convcrf.get_test_conf()

        conf["final_softmax"] = False
        out_1 = _run_forward(conf, num_iter=1)
        out_5 = _run_forward(conf, num_iter=5)
        assert not np.allclose(out_1, out_5, atol=1e-3)

    def test_clean_filters_resets_kernel(self):
        conf = convcrf.get_test_conf()

        conf["final_softmax"] = False
        model = convcrf.GaussCRF(conf=conf, shape=(10, 10), nclasses=2, use_gpu=False)
        img_t, unary_t = _prepare_inputs()
        model.eval()
        with torch.no_grad():
            model.forward(unary=unary_t, img=img_t)
        # After forward, clean_filters should have been called
        assert model.CRF.kernel is None

    def test_argmax_matches_expected_pattern(self):
        """On the simple test image (center square), the CRF should roughly
        assign class 1 to the center and class 0 to the border."""
        conf = convcrf.get_test_conf()

        conf["final_softmax"] = False
        out = _run_forward(conf)
        pred = np.argmax(out[0], axis=0)  # (10, 10)
        # Center pixel should be class 1 (the center square class)
        assert pred[5, 5] == 1
        # Corner pixel should be class 0 (the border class)
        assert pred[0, 0] == 0


# ---------------------------------------------------------------------------
# Test 5: Interpolation equivalence tests
#   Verify that F.interpolate(align_corners=False) produces identical output
#   to the deprecated F.upsample for the exact tensor shapes and parameters
#   used inside MessagePassingCol._compute_gaussian.
#
#   This is the critical guard for the upsample -> interpolate migration
#   (plan step 7). If these tests pass after the swap, numerical
#   reproducibility is guaranteed.
# ---------------------------------------------------------------------------
class TestInterpolationEquivalence:
    """Directly compare F.upsample vs F.interpolate on tensors that match
    the shapes and values flowing through _compute_gaussian with blur>1."""

    @staticmethod
    def _build_message_tensor(blur, npixels, nclasses=2, bs=1):
        """Simulate the 'message' tensor at the point where upsample is called
        inside _compute_gaussian. Returns (message, pad_0, pad_1)."""
        torch.manual_seed(42)
        off_0 = (blur - npixels[0] % blur) % blur
        off_1 = (blur - npixels[1] % blur) % blur
        pad_0 = math.ceil(off_0 / 2)
        pad_1 = math.ceil(off_1 / 2)
        downsampled_h = math.ceil(npixels[0] / blur)
        downsampled_w = math.ceil(npixels[1] / blur)
        message = torch.randn(bs, nclasses, downsampled_h, downsampled_w)
        return message, pad_0, pad_1, npixels

    @staticmethod
    def _upsample_old(message, blur, pad_0, pad_1, npixels):
        """Replicate the exact upsample code path from convcrf.py:463-471."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            up = torch.nn.functional.upsample(
                message, scale_factor=blur, mode='bilinear')
        up = up[:, :, pad_0:pad_0 + npixels[0], pad_1:npixels[1] + pad_1]
        return up.contiguous()

    @staticmethod
    def _interpolate_new(message, blur, pad_0, pad_1, npixels):
        """The proposed replacement using F.interpolate with explicit
        align_corners=False (matches the default used by F.upsample)."""
        up = F.interpolate(
            message, scale_factor=blur, mode='bilinear', align_corners=False)
        up = up[:, :, pad_0:pad_0 + npixels[0], pad_1:npixels[1] + pad_1]
        return up.contiguous()

    def test_blur2_10x10(self):
        """blur=2 on 10x10 — used by TestBlur2Config."""
        msg, p0, p1, npix = self._build_message_tensor(blur=2, npixels=(10, 10))
        old = self._upsample_old(msg, 2, p0, p1, npix)
        new = self._interpolate_new(msg, 2, p0, p1, npix)
        torch.testing.assert_close(old, new, atol=0, rtol=0)

    def test_blur4_10x10(self):
        """blur=4 on 10x10 — used by TestDefaultConfig."""
        msg, p0, p1, npix = self._build_message_tensor(blur=4, npixels=(10, 10))
        old = self._upsample_old(msg, 4, p0, p1, npix)
        new = self._interpolate_new(msg, 4, p0, p1, npix)
        torch.testing.assert_close(old, new, atol=0, rtol=0)

    def test_blur2_odd_dimensions(self):
        """blur=2 on 15x13 — odd spatial dims test the padding logic."""
        msg, p0, p1, npix = self._build_message_tensor(blur=2, npixels=(15, 13))
        old = self._upsample_old(msg, 2, p0, p1, npix)
        new = self._interpolate_new(msg, 2, p0, p1, npix)
        torch.testing.assert_close(old, new, atol=0, rtol=0)

    def test_blur4_realistic_resolution(self):
        """blur=4 on 64x64 — closer to a realistic image patch size."""
        msg, p0, p1, npix = self._build_message_tensor(
            blur=4, npixels=(64, 64), nclasses=21, bs=1)
        old = self._upsample_old(msg, 4, p0, p1, npix)
        new = self._interpolate_new(msg, 4, p0, p1, npix)
        torch.testing.assert_close(old, new, atol=0, rtol=0)

    def test_blur2_multichannel_batched(self):
        """blur=2, 21 classes, batch size 4 — stress test."""
        msg, p0, p1, npix = self._build_message_tensor(
            blur=2, npixels=(32, 48), nclasses=21, bs=4)
        old = self._upsample_old(msg, 2, p0, p1, npix)
        new = self._interpolate_new(msg, 2, p0, p1, npix)
        torch.testing.assert_close(old, new, atol=0, rtol=0)

    def test_align_corners_true_differs(self):
        """Confirm that align_corners=True produces DIFFERENT results,
        so we know the parameter actually matters."""
        msg, p0, p1, npix = self._build_message_tensor(blur=2, npixels=(10, 10))
        old = self._upsample_old(msg, 2, p0, p1, npix)
        wrong = F.interpolate(
            msg, scale_factor=2, mode='bilinear', align_corners=True)
        wrong = wrong[:, :, p0:p0 + npix[0], p1:npix[1] + p1].contiguous()
        assert not torch.allclose(old, wrong, atol=1e-6), \
            "align_corners=True should differ from the default (False)"
