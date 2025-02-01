Mathematical formulae
---------------------

.. |m_log_cdf| replace:: :math:`\mathcal{L}(y)`

.. py:currentmodule:: pymultifit.distributions

This page lists all the formulae used along with their derivations.
The **PDF** function is given by :math:`f(y)`, with :math:`\ell(y)` representing the **logPDF**.
Similarly, the **CDF** functions is given by :math:`F(y)`, with |m_log_cdf| representing the **logCDF**.

ArcSineDistribution
===================

The PDF of :class:`~arcSine_d.ArcSineDistribution` is as follows,

.. math::
   f(y) = \dfrac{1}{\pi\sqrt{y(1-y)}}

The logPDF can be derived as follows,

.. math::
    \begin{gather}
       \ell(y) = \ln\left(\dfrac{1}{\pi\sqrt{y(1-y)}}\right)\\
       \ell(y) = \ln(1) - \ln\left(\pi\sqrt{y - y^2}\right)\\
       \ell(y) = -\ln(\pi) - \ln(\sqrt{y-y^2})
    \end{gather}

The CDF of :class:`~arcSine_d.ArcSineDistribution` is as follows,

.. math::
   F(y) = \left(\dfrac{2}{\pi}\right)\arcsin(\sqrt{y})

The logCDF can be derived as follows,

.. math::
    \begin{gather}
        \mathcal{L}(y) = \ln\left[\left(\dfrac{2}{\pi}\right)\arcsin(\sqrt{y})\right]\\
        \mathcal{L}(y) = \ln\left(\dfrac{2}{\pi}\right) + \ln\left[\arcsin(\sqrt{y})\right]
    \end{gather}