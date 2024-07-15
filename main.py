import numpy as np
import matplotlib.pyplot as plt

class PSF:
    def __init__(self, x0, y0, z0, sigma_xy, sigma_z):
        self.x0 = x0
        self.y0 = y0
        self.z0 = z0
        self.sigma_xy = sigma_xy
        self.sigma_z = sigma_z
        self.image = np.zeros((255, 255, 3), dtype=np.uint8)

    def compute_psf(self):
        x = np.linspace(-127, 127, 255) - self.x0
        y = np.linspace(-127, 127, 255) - self.y0
        x, y = np.meshgrid(x, y)
        z = self.z0  # Since we are working with a 2D slice, z is constant
        gaussian_2d = np.exp(-((x**2 + y**2) / (2 * self.sigma_xy**2) + (z**2) / (2 * self.sigma_z**2)))
        return gaussian_2d

    def generate_image(self):
        psf = self.compute_psf()
        plt.imshow(psf, cmap='gray', vmin=0, vmax=1)
        plt.colorbar()
        plt.axis('off')  # Hide the axes
        plt.show()

x0 = float(input("Enter x0: "))
y0 = float(input("Enter y0: "))
z0 = float(input("Enter z0: "))
sigma_xy = float(input("Enter sigma_xy: "))
sigma_z = float(input("Enter sigma_z: "))

psf_instance = PSF(x0, y0, z0, sigma_xy, sigma_z)
psf_instance.generate_image()
