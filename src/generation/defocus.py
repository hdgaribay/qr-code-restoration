import numpy as np

def disk_psf(radius):
    # Coordinates centered on zero.
    coordinates = np.arange(-radius, radius + 1)
    x, y = np.meshgrid(coordinates, coordinates)

    # Select positions inside the circle.
    disk = (x**2 + y**2) <= radius**2

    # Convert the mask into normalized floating-point weights.
    psf = disk.astype(np.float64)
    psf /= psf.sum()

    return psf