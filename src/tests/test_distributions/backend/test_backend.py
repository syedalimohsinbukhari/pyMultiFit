"""Created on Dec 11 11:24:19 2024"""

from typing import Any, Dict

import numpy as np
import pytest

from ....pymultifit.distributions.backend import BaseDistribution


class MockDistribution(BaseDistribution):
    """A mock distribution for testing purposes. Implements a simple uniform distribution in the range [0, 1]."""

    def pdf(self, x: np.ndarray) -> np.ndarray:
        return np.where((x >= 0) & (x <= 1), 1.0, 0.0)

    def cdf(self, x: np.ndarray) -> np.ndarray:
        return np.where(x < 0, 0.0, np.where(x > 1, 1.0, x))

    def stats(self) -> Dict[str, float]:
        return {"mean": 0.5, "variance": 1 / 12}


class TestMockDistribution:

    @staticmethod
    def test_base_distribution_instantiation():
        """Test that instantiating the BaseDistribution directly raises a NotImplementedError."""
        base_dist = BaseDistribution()
        x = np.array([0.5])

        # Check for NotImplementedError in all abstract methods
        with pytest.raises(NotImplementedError):
            base_dist.pdf(x)

        with pytest.raises(NotImplementedError):
            base_dist.cdf(x)

        with pytest.raises(NotImplementedError):
            base_dist.stats()

    @staticmethod
    def test_mock_distribution_pdf():
        """Test the PDF of the MockDistribution."""
        mock_dist = MockDistribution()
        x = np.array([-1.0, 0.0, 0.5, 1.0, 2.0])
        expected_pdf = np.array([0.0, 1.0, 1.0, 1.0, 0.0])

        assert np.allclose(mock_dist.pdf(x), expected_pdf)

    @staticmethod
    def test_mock_distribution_cdf():
        """Test the CDF of the MockDistribution."""
        mock_dist = MockDistribution()
        x = np.array([-1.0, 0.0, 0.5, 1.0, 2.0])
        expected_cdf = np.array([0.0, 0.0, 0.5, 1.0, 1.0])

        assert np.allclose(mock_dist.cdf(x), expected_cdf)

    @staticmethod
    def test_mock_distribution_stats():
        """Test the statistics of the MockDistribution."""
        mock_dist = MockDistribution()
        stats = mock_dist.stats()

        assert stats["mean"] == 0.5
        assert stats["variance"] == pytest.approx(1 / 12, rel=1e-5)
