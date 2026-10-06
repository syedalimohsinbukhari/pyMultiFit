"""Created on Oct 06 2026"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass

import numpy as np

from .typing import NDArray


@dataclass(frozen=True)
class Component:
    """A single additive component of a fitted model.

    Parameters
    ----------
    label :
        Display name of the component, e.g. ``"Gaussian"``.
    func :
        The component's model function, called as ``func(x, params)``.
    n_par :
        Number of parameters the component consumes from the flat parameter vector.
    """

    label: str
    func: Callable
    n_par: int


@dataclass(frozen=True)
class FitResult:
    """An immutable description of a (possibly not yet performed) fit.

    This is everything the plotting and confidence-interval code needs to know about a fitter, so that those modules
    never have to import or inspect a fitter object.

    Parameters
    ----------
    x, y :
        The data the fit was (or will be) performed on.
    params :
        The flat vector of fitted parameters, ``None`` before the fit is performed.
    covariance :
        The covariance matrix of ``params``, ``None`` before the fit is performed.
    components :
        The additive components of the model, in the order their parameters appear in ``params``.
    param_labels :
        One display label per entry of ``params``.
    """

    x: NDArray
    y: NDArray
    params: NDArray | None
    covariance: NDArray | None
    components: tuple[Component, ...]
    param_labels: tuple[str, ...]

    @property
    def n_fits(self) -> int:
        """Number of components in the model."""
        return len(self.components)

    @property
    def is_fitted(self) -> bool:
        """Whether the fit has been performed."""
        return self.params is not None

    @property
    def errors(self) -> NDArray:
        """Standard errors of the fitted parameters."""
        self.require_fit()
        return np.sqrt(np.diag(self.covariance))

    def require_fit(self) -> None:
        """Raise ``RuntimeError`` if the fit has not been performed yet."""
        if self.params is None or self.covariance is None:
            raise RuntimeError("Fit not performed yet. Call fit() first.")

    def slices(self) -> list[slice]:
        """Slices selecting each component's parameters from the flat parameter vector."""
        out, start = [], 0
        for comp in self.components:
            out.append(slice(start, start + comp.n_par))
            start += comp.n_par
        return out

    def split(self, params: NDArray) -> Iterator[tuple[Component, NDArray]]:
        """Yield ``(component, its_parameters)`` for a flat parameter vector."""
        for comp, sl in zip(self.components, self.slices()):
            yield comp, params[sl]

    def component_curve(self, index: int, x: NDArray | None = None, params: NDArray | None = None) -> NDArray:
        """Evaluate a single component.

        Parameters
        ----------
        index :
            Zero-based component index.
        x :
            Evaluation points. Defaults to the fitted data's ``x``.
        params :
            A flat parameter vector. Defaults to the fitted parameters.
        """
        x, params = self._resolve(x, params)
        comp, sl = self.components[index], self.slices()[index]
        return np.asarray(comp.func(x, list(params[sl])))

    def model(self, x: NDArray | None = None, params: NDArray | None = None) -> NDArray:
        """Evaluate the total model, i.e. the sum of all components.

        Parameters
        ----------
        x :
            Evaluation points. Defaults to the fitted data's ``x``.
        params :
            A flat parameter vector. Defaults to the fitted parameters.
        """
        x, params = self._resolve(x, params)
        y = np.zeros_like(x, dtype=float)
        for comp, pars in self.split(params):
            y += comp.func(x, list(pars))
        return y

    def residuals(self) -> NDArray:
        """Data minus the fitted model."""
        self.require_fit()
        return self.y - self.model()

    def _resolve(self, x: NDArray | None, params: NDArray | None) -> tuple[NDArray, NDArray]:
        if params is None:
            self.require_fit()
            params = self.params
        return (self.x if x is None else np.asarray(x)), np.asarray(params)
