import numpy as np, matplotlib.pyplot as plt, cv2


def add_noise(img, D0, amplitude):
    h, w = img.shape
    p = h // 2
    q = w // 2

    ft = np.fft.fft2(img)
    ft_shift = np.fft.fftshift(ft)

    magnitude_specturm_ac = np.abs(ft_shift)
    ang = np.angle(ft_shift)

    peak_strength = amplitude * h * w / 2

    noisy_magnitude = magnitude_specturm_ac.copy()

    noisy_magnitude[p, q + D0] += peak_strength
    noisy_magnitude[p, q - D0] += peak_strength
    noisy_magnitude[p + D0, q] += peak_strength
    noisy_magnitude[p - D0, q] += peak_strength

    noisy_spectrum = np.multiply(noisy_magnitude, np.exp(1j * ang))

    img_noisy = np.real(np.fft.ifft2(np.fft.ifftshift(noisy_spectrum)))

    return img_noisy

img = cv2.imread("boat.jpg", 0)
img_noisy = add_noise(img, D0=100, amplitude=50)


h, w = img_noisy.shape
D0 = 100
W = 30
n = 2

ft = np.fft.fft2(img_noisy)
ft_shift = np.fft.fftshift(ft)

magnitude_specturm_ac = np.abs(ft_shift)
ang = np.angle(ft_shift)

H_ideal = np.ones_like(magnitude_specturm_ac)
H_butterworth = np.ones_like(magnitude_specturm_ac)
H_gaussian = np.ones_like(magnitude_specturm_ac)

for i in range(h):
    for j in range(w):
        D = ((i - h//2)**2 + (j - w//2)**2) ** 0.5

        if D0 - W/2 <= D <= D0 + W/2:
            H_ideal[i, j] = 0

        if D**2 == D0**2:
            H_butterworth[i, j] = 0
        else:
            H_butterworth[i, j] = 1 / (1 + ((D * W) / (D**2 - D0**2))**(2 * n))

        if D != 0:
            H_gaussian[i, j] = 1 - np.exp(-((D**2 - D0**2) / (D * W))**2)

ideal_result = ft_shift * H_ideal
ideal_img_back = np.real(np.fft.ifft2(np.fft.ifftshift(ideal_result)))

butterworth_result = ft_shift * H_butterworth
butterworth_img_back = np.real(np.fft.ifft2(np.fft.ifftshift(butterworth_result)))

gaussian_result = ft_shift * H_gaussian
gaussian_img_back = np.real(np.fft.ifft2(np.fft.ifftshift(gaussian_result)))

fig = plt.figure(figsize=(12, 10))


plt.subplot(3, 3, 1)
plt.imshow(img_noisy, cmap="gray", vmin=0, vmax=255)
plt.title("Input Image with Noise")
plt.axis("off")

plt.subplot(3, 3, 2)
plt.imshow(H_ideal, cmap="gray", vmin=0, vmax=1)
plt.title("Ideal Reject Filter")
plt.axis("off")

plt.subplot(3, 3, 3)
plt.imshow(ideal_img_back, cmap="gray", vmin=0, vmax=255)
plt.title("Reconstructed Image")
plt.axis("off")

plt.subplot(3, 3, 4)
plt.imshow(img_noisy, cmap="gray", vmin=0, vmax=255)
plt.title("Input Image with Noise")
plt.axis("off")

plt.subplot(3, 3, 5)
plt.imshow(H_butterworth, cmap="gray", vmin=0, vmax=1)
plt.title("Butterworth Reject Filter")
plt.axis("off")

plt.subplot(3, 3, 6)
plt.imshow(butterworth_img_back, cmap="gray", vmin=0, vmax=255)
plt.title("Reconstructed Image")
plt.axis("off")

plt.subplot(3, 3, 7)
plt.imshow(img_noisy, cmap="gray", vmin=0, vmax=255)
plt.title("Input Image with Noise")
plt.axis("off")

plt.subplot(3, 3, 8)
plt.imshow(H_gaussian, cmap="gray", vmin=0, vmax=1)
plt.title("Gaussian Band Reject Filter")
plt.axis("off")

plt.subplot(3, 3, 9)
plt.imshow(gaussian_img_back, cmap="gray", vmin=0, vmax=255)
plt.title("Reco" \
"nstructed Image")
plt.axis("off")

plt.suptitle("Band Reject Filtering", fontsize=20, y=0.97)
plt.subplots_adjust(left=0.05, right=0.98, bottom=0.05, top=0.90, wspace=0.08, hspace=0.15)
fig.text(0.45, 0.01, "Roll: 2107052", fontsize=12)



plt.show()