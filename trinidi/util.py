"""Some util functions."""

import numpy as np
SPEED_OF_LIGHT = 299792458 # m/s
MASS_OF_NEUTRON = 939.56542052 * 1e6 / (SPEED_OF_LIGHT) ** 2  # [eV s²/m²]

TOF_LABEL = "Time-of-flight [μs]"
ENERGY_LABEL = "Energy [eV]"


def time2energy(time, flight_path_length):
    r"""Convert time-of-flight to energy of the neutron. 

    .. math::
        E = \left( \gamma - 1 \right) m c^2 \; ,
        \gamma = \frac{1}{\sqrt{1 - \left(\frac{L}{c \cdot t} \right)^2}} \; ,

    where :math:`E` is the energy, :math:`m` is the mass, :math:`c` is 
    the speed of light, :math:`t` is the time-of-flight of the neutron, 
    and :math:`L` is the flight path length.

    Args:
        time: Time-of-flight in :math:`\mathrm{μs}`.
        flight_path_length: flight path length in :math:`\mathrm{m}`.

    Returns:
        Energy of the neutron in :math:`\mathrm{eV}`.
    """
    m = MASS_OF_NEUTRON  # [eV s²/m²]
    c = SPEED_OF_LIGHT # m/s
    L = flight_path_length  # m
    t = time / 1e6  # s
    γ = 1 / np.sqrt (1 - (L / t) ** 2 / c ** 2 )
    return ( γ - 1 ) * m * c ** 2  # eV


def energy2time(energy, flight_path_length):
    r"""Convert energy to time-of-flight of the neutron.

    .. math::
        t = \frac{L}{c} \sqrt{ \frac{\gamma^2}{\gamma^2 - 1 }} \; ,
        \gamma = 1 + \frac{E}{mc^2}
        
    where :math:`E` is the energy, :math:`m` is the mass, :math:`c` 
    is the speed of light, :math:`t` is the time-of-flight of the neutron, 
    and :math:`L` is the flight path length.

    Args:
        energy:  Energy of the neutron in :math:`\mathrm{eV}`.
        flight_path_length: flight path length in :math:`\mathrm{m}`.

    Returns:
        Time-of-flight in :math:`\mathrm{μs}`.

    """
    L = flight_path_length  # m
    m = MASS_OF_NEUTRON  # eV s²/m²
    c = SPEED_OF_LIGHT # m/s
    E = energy  # eV
    γ = 1 + E / m / c ** 2
    t = L / c * np.sqrt(γ ** 2 / ( γ ** 2 - 1 ) ) # s 
    return t * 1e6  # μs


def no_nan_divide(x, y):
    """Return `x/y`, with 0 instead of NaN where `y` is 0.

    Args:
        x: Numerator.
        y: Denominator.

    Returns:
        `x / y` with 0 wherever `y == 0`.
    """
    return np.where(y != 0, np.divide(x, np.where(y != 0, y, 1)), 0)
