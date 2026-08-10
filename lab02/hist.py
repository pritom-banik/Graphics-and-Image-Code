import numpy as np 
import matplotlib.pyplot as plt
import cv2

def compute_cdf(pdf):
    cdf=np.empty_like(pdf)
    cdf[0]=pdf[0]
    for i in range(1, len(pdf)):
        cdf[i]=cdf[i-1]+pdf[i]
    return cdf


#====================================================
img1=cv2.imread('boat.jpg')

source=cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)

lab = cv2.cvtColor(source, cv2.COLOR_RGB2LAB)

L, a, b = cv2.split(lab)

hist, bins = np.histogram(L.ravel(), bins=256, range=(0, 256))

pdf=hist/np.sum(hist)

cdf=compute_cdf(pdf)




#====================================================
img2=cv2.imread('power_plant.jpg')

reference=cv2.cvtColor(img2,cv2.COLOR_BGR2RGB)

ref_lab=cv2.cvtColor(reference, cv2.COLOR_RGB2LAB)

L_ref, a_ref, b_ref = cv2.split(ref_lab)

ref_hist, ref_bins = np.histogram(L_ref.ravel(), bins=256, range=(0, 256))

ref_pdf=ref_hist/np.sum(ref_hist)

ref_cdf=compute_cdf(ref_pdf)




#====================================================
mapping = np.zeros(256, dtype=np.uint8)

for r in range(256):
    z = np.argmin(np.abs(ref_cdf - cdf[r]))
    mapping[r] = z


L_matched = mapping[L]

matched_lab = cv2.merge((L_matched, a, b))

matched_rgb = cv2.cvtColor(matched_lab, cv2.COLOR_LAB2RGB)

matched_hist, matched_bins = np.histogram(L_matched.ravel(), bins=256, range=(0, 256))

matched_pdf=matched_hist/np.sum(matched_hist)

matched_cdf=compute_cdf(matched_pdf)




#==================================================== extra==========
plt.figure(figsize=(15, 8))

plt.subplot(3, 3, 1)
plt.imshow(img1)
plt.title("Input Image")
plt.axis("off")

plt.subplot(3, 3, 2)
plt.plot(bins[:-1], pdf, color='red', linewidth=2)
plt.title("PDF of source image")


plt.subplot(3, 3, 3)
plt.plot(cdf, color='red', linewidth=2)
plt.title("CDF of source image")


plt.subplot(3, 3, 4)
plt.imshow(img2)
plt.title("Reference Image")
plt.axis("off")

plt.subplot(3, 3, 5)
plt.plot(bins[:-1], ref_pdf, color='blue', linewidth=2)
plt.title("PDF of reference image")


plt.subplot(3, 3, 6)
plt.plot(ref_cdf, color='blue', linewidth=2)
plt.title("CDF of reference image")



plt.subplot(3, 3, 7)
plt.imshow(matched_rgb)
plt.title("Matched Image")
plt.axis("off")

plt.subplot(3, 3, 8)
plt.plot(matched_bins[:-1], matched_pdf, color='green', linewidth=2)
plt.title("PDF of matched image")


plt.subplot(3, 3, 9)
plt.plot(matched_cdf, color='green', linewidth=2)
plt.title("CDF of matched image")



plt.subplots_adjust(
    hspace=0.3,
    bottom=0.18
)


plt.figtext(
    0.5, 0.04,
    "Equalize an RGB image by equalizing only the L channel in Lab color space",
    ha="center",
    fontsize=12,
    fontweight="bold"
)


plt.figtext(
    0.98, 0.01,
    "Name: Pritom Banik" \
    "\nRoll: 2107052",
    ha="right",
    va="bottom",
    fontsize=11
)

plt.show()
