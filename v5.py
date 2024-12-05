from skimage import io
from skimage.transform import resize, rescale
import numpy as np
import matplotlib.pyplot as plt

# Primer 5.1
# Odrediti koliko piksela slike zelda.png ima vrednost 50.

# img = io.imread('test_slike_omv/zelda.png')
# counter = (img==50).sum()
# print(counter)


#  Primer 5.2
# Odrediti koliko piksela slike undercurrent.jpg ima vrednost 0, 1, . . . 255. Dobijeni rezultat prikazati grafiˇcki.

# img = io.imread('test_slike_omv/boat.png')
#
# plt.figure(figsize=(15, 5))
# plt.subplot(1, 4, 1)
# plt.imshow(img, cmap='gray')
#
# cnt = np.zeros(256)
# for i in range(256):
#     cnt[i] = (img == i).sum()
#
# plt.subplot(1, 4, 2)
# plt.plot(np.arange(0, 256), cnt)
#
# h = np.zeros(256)
# for pixel_intensity in img.ravel():
#     h[pixel_intensity] += 1
#
# plt.subplot(1, 4, 3)
# plt.plot(np.arange(0, 256), h)
#
# h_norm = h / h.sum()
#
# plt.subplot(1, 4, 4)
# plt.plot(np.arange(0, 256), h_norm)
# plt.show()


# Primer 5.3
# Koriste´ci python pakete numpy i skimage odrediti i prikazati histograme slike boat.png. Smatrati
# da histogram odredujemo u broju binova definisan opsegom mogu´cih vrednosti za dati tip (kao u ¯
# prethodnom primeru).

# from skimage import exposure
#
# img = io.imread('test_slike_omv/boat.png')
#
# plt.figure(figsize=(15, 5))
# plt.subplot(1, 3, 1)
# plt.imshow(img, cmap='gray')
#
# numpy_hist, nbin_edges = np.histogram(img, bins=np.arange(257))
# skimage_hist, sbin_centers = exposure.histogram(img, source_range='dtype')
#
# plt.subplot(1, 3, 2)
# plt.plot(np.arange(256), numpy_hist)
# plt.subplot(1, 3, 3)
# plt.plot(sbin_centers, skimage_hist)
# plt.show()


#  Primer 5.4
# Za sliku cat.png odrediti udeo piksela koji imaju intenzitet 50 ili manji.

# img = io.imread('test_slike_omv/cat.png')
# N, M =img.shape
# perc = round((img <= 50).sum() / N / M * 100, 4)
# print(f'{perc}%')
#
# hist,bins = np.histogram(img, bins=np.arange(257), density=True)
# perc2 = round(hist[:51].sum() * 100, 4)
# print(f'{perc2}%')
#
# plt.plot(np.arange(256), hist)
# plt.show()
# def udeo(img, i):
#     hist, bins = np.histogram(img, bins=np.arange(257), density=True)
#     perc = round(hist[:i + 1].sum() * 100, 2)
#     return perc
#
# img = io.imread('test_slike_omv/lena.png')
# print(f'{udeo(img, 25)}%')


# Primer 5.5
# Za sliku lena.png odrediti udeo piksela koji imaju intenzitet i ili manji za i = 0, 1, ..., 255.

# img = io.imread('test_slike_omv/lena.png')
# hist, bin_edges = np.histogram(img, bins = np.arange(257), density=True)
# udeli = np.zeros(256)
# for i in range(256):
#     udeli[i] = hist[:i+1].sum()
# plt.figure(figsize=(10,5))
# plt.subplot(1,2,1)
# plt.plot(np.arange(256), hist)
# plt.subplot(1,2,2)
# plt.plot(np.arange(256), udeli)
# plt.show()


# Primer 5.6
# Napisati funkciju imageHistEq koja ekvalizuje histogram ulazne slike. Pretpostaviti da je opseg
# vrednosti ulazne slike [0, 255]. Transformacija koja odreduje ekvalizaciju histograma je data slede- ¯
# ´cim izrazom:
# 7
# T(rk) = (L − 1) ·
# k
# ∑
# i=0
# hi
# n
# , k = 0, 1, ..., L − 1
# gde k predstavlja nivo intenziteta u opsegu [0, L − 1], gde je L ukupni broj niva intenziteta koji
# piksel može imati (zavisno od tipa), rk
# je intenzitet piksela na nivou k, T(·) je transformacija koja
# definiše ekvalizaciju histograma, hi
# je broj piksela koji imaju intenzitet i, a n je ukupan broj piskela
# u slici.
# Kao rezultat funkcije vratiti sliku sa ekvalizovanim histogramom i transformaciju koja je to omogu´cila.
# Funkciju primeniti nad slikama dark.png i light.png. Prikazati sliku i njen histogram pre i posle
# poziva funkcije. Prikazati kako izgleda funkcija T za svaku sliku.

# def imageHistEq(img):
#     hist = np.histogram(img, bins=np.arange(257), density=True)[0]
#     T = 255 * np.cumsum(hist)
#     T = np.round(T).astype('uint8')  # ako zelimo rezultat u tipu uint8
#     return T[img], T
#
# img = io.imread('test_slike_omv/cat.png')
# cat_hist = np.histogram(img, bins=np.arange(257), density=True)[0]
# cat_eq, T_cat = imageHistEq(img)
# cat_eq_hist = np.histogram(cat_eq, bins=np.arange(257), density=True)[0]
#
# plt.figure(figsize=(10,3))
# plt.subplot(1,5,1)
# plt.imshow(img, cmap='gray')
# plt.subplot(1,5,2)
# plt.plot(np.arange(256), cat_hist)
# plt.subplot(1,5,4)
# plt.imshow(cat_eq, cmap='gray')
# plt.subplot(1,5,3)
# plt.plot(np.arange(256), T_cat)
# plt.subplot(1,5,5)
# plt.plot(np.arange(256), cat_eq_hist)
# plt.show()

# from skimage import exposure
# from skimage.util import img_as_ubyte
#
# dark = io.imread('test_slike_omv/zelda.png')
# hist, bin_centers = exposure.histogram(dark, source_range='dtype', normalize=True)
# dark_eq = img_as_ubyte(exposure.equalize_hist(dark))
# hist_eq, bin_centers = exposure.histogram(dark_eq, source_range='dtype', normalize=True)
# plt.figure(figsize=(10, 10))
# plt.subplot(2, 2, 1)
# plt.imshow(dark, vmin=0, vmax=255, cmap='gray')
# plt.subplot(2, 2, 2)
# plt.plot(np.arange(256), hist)
# plt.subplot(2, 2, 3)
# plt.imshow(dark_eq, vmin=0, vmax=255, cmap='gray')
# plt.subplot(2, 2, 4)
# plt.plot(np.arange(256), hist_eq)
# plt.show()


# Zadaci za samostalnu vežbu