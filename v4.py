import numpy as np
from skimage import io
from skimage.transform import resize, rescale
import matplotlib.pyplot as plt
import plotly.express as px

# Primer 4.1
# U promenljivu img uˇcitati lena.png. Odrediti sliku koja nastaje odredivanjem kvadratnog korena  ̄
# nad vrednostima piksela uˇcitane slike. Slike prikazati u opsegu tonske skale [0,255].

# img = io.imread('test_slike_omv/lena.png')
# img_sqrt = np.sqrt(img.astype('float'))
# plt.figure(figsize=(10,5))
# plt.subplot(1,2,1)
# plt.imshow(img, cmap='gray')
# plt.subplot(1,2,2)
# plt.imshow(img_sqrt, cmap='gray', vmin=0, vmax=255)
# plt.show()

# LUT = np.sqrt(np.arange(0,256))
#
# fig = px.line(x=np.arange(256), y=LUT)
# fig.update_layout(xaxis_title="ulazni intenzitet", yaxis_title="izlazni intenzitet")
# fig.show(config={'modeBarButtonsToAdd':['drawline',
# 'eraseshape']})
#
# img_sqrt2 = LUT[img]
# fig = px.imshow(img_sqrt2,zmin = 0, zmax = 255, color_continuous_scale='gray')
# fig.show()


# Primer 4.2
#
# Napisati funkciju gammaCorrection koja implementira stepenu transformaciju nad slikama sa ska-
# liranjem na opseg [0, 255], odredenu parametrom  ̄ gamma. Smatrati da je tip ulazne i izlazne slike
# uint8. U implementaciji koristiti LUT tabelu. Testirati funkciju nad slikom undercurrent.jpg i
# parametrom gamma od 0.5.

# def gammaCorrection(input, gamma):
#     c = 255 ** (1 - gamma)
#     LUT = c * np.arange(0, 256) ** gamma
#     LUT = np.round(LUT).astype('uint8')
#     return LUT[input]
#
# img = io.imread('test_slike_omv/lena.png')
# img_gc = gammaCorrection(img,0.5)
# plt.figure(figsize=(10,5))
# plt.subplot(1,2,1)
# plt.imshow(img, cmap='gray')
# plt.subplot(1,2,2)
# plt.imshow(img_gc, cmap='gray', vmin=0, vmax=255)
# plt.show()


# 2.3 Primer 4.3
# Odrediti negativ slike boat.png koriste ́ci LUT.

# def negativ(img):
#     LUT = (255-np.arange(0,256)).astype('uint8')
#     return LUT[img]
#
# plt.plot(np.arange(256),negativ(np.arange(256)))
# plt.xlabel('ulazni intenziteti')
# plt.ylabel('izlazni intenziteti')
# plt.show()
#
# img = io.imread('test_slike_omv/boat.png')
# img_n = negativ(img)
# plt.figure(figsize=(10,5))
# plt.subplot(1,2,1)
# plt.imshow(img, cmap='gray')
# plt.subplot(1,2,2)
# plt.imshow(img_n, cmap='gray', vmin=0, vmax=255)
# plt.show()


# 2.4 Primer 4.4
# Nad slikom cat.png upotrebiti deo-po-deo linearnu transformaciju. Transformacija je odredena  ̄
# parovima (ulaz,izlaz).
# (0,0), (100,50), (200,250), (255,255)

# lut1 = np.linspace(0,50,101)
# lut2 = np.linspace(50,250, 101)
# lut3 = np.linspace(250, 255, 56)
# LUT = np.concatenate((lut1[:-1],lut2[:-1],lut3), axis=0)
# # plt.plot(LUT)
# # plt.show()
#
# img = io.imread('test_slike_omv/cat.png')
# img_2 = LUT[img]
# plt.figure(figsize=(10,5))
# plt.subplot(1,2,1)
# plt.imshow(img, cmap='gray')
# plt.subplot(1,2,2)
# plt.imshow(img_2, cmap='gray', vmin=0, vmax=255)
# plt.show()

# 3 Zadaci za samostalnu vežbu
# Zadatak 4.2

# Linearnu deo-po-deo transformaciju mogu ́ce je upotrebiti da bi se na mamogramima naglasile
# strukture odredene gustine. Napisati funkciju  ̄ tissueEmphasis koja implementira linearnu deo-
# po-deo transformaciju prikazanu na slici. Funkcija prima parametre p1 i p2 koji definišu njen iz-
# gled.

# def tissueEmphasis(img, p1, p2):
#     lut1 = np.linspace(0,5,p1+1)
#     lut2 = np.linspace(5, 245, p2-p1+1)
#     lut3 = np.linspace(245,255,256-p2)
#     LUT = np.concatenate((lut1[:-1],lut2[:-1],lut3), axis=0)
#     return LUT[img], LUT
#
# img=io.imread('test_slike_omv/boat.png')
# img_2,LUT=tissueEmphasis(img,40,200)
# plt.plot(LUT)
# plt.show()
#
# plt.figure(figsize=(10,5))
# plt.subplot(1,2,1)
# plt.imshow(img, cmap='gray')
# plt.subplot(1,2,2)
# plt.imshow(img_2, cmap='gray', vmin=0, vmax=255)
# plt.show()

# Zadatak 4.3
# Prepraviti funkciju gammaCorrection tako da može da se koristi za obradu snimaka uint16 tipa.

# def gammaCorrection(input, gamma):
#     c = 65535 ** (1 - gamma)
#     LUT = c * np.arange(0, 65535) ** gamma
#     LUT = np.round(LUT).astype('uint16')
#     return LUT[input]