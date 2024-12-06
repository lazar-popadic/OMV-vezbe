from skimage import io
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal  # za koriscenje scipy.signal.convolve2d funkcije
from scipy import ndimage  # za koriscenje scipy.ndimage.convolve funkcije


# 3.1 Primer 6.1
# Nad slikom fabio.tif izvršiti proširivanje:
# 1. sa svih strana za po 200 piksela vrednosti 100
# 2. sa gornje i donje strane za po 200 piksela, a sa leve i desne za po 100 piksela vrednosti 0
# 3. sa gornje i donje strane za po 150 piksela ponavljanjem vrednosti iviˇcnih piksela
# 4. sa leve i desne strane za po 250 piksela simetriˇcnim ponavljanjem vrednosti
# 5. sa donje strane za po 100 piksela cirkularnim ponavljanjem piksela

# img = io.imread('test_slike_omv/barbara.png')
# plt.figure(figsize=(12,8))
# plt.subplot(2,3,1)
# plt.imshow(img, cmap='gray')
# # 1. constant padding
# img1 = np.pad(img, pad_width=200, mode='constant', constant_values=100)
# plt.subplot(2,3,2)
# plt.imshow(img1, cmap='gray')
# # 2. zero padding
# img2 = np.pad(img, ((200,),(100,)))
# plt.subplot(2,3,3)
# plt.imshow(img2, cmap='gray')
# # 3. edge padding
# img3 = np.pad(img, ((150,),(0,)), mode='edge')
# plt.subplot(2,3,4)
# plt.imshow(img3, cmap='gray')
# # 4. symmetric padding
# img4 = np.pad(img, ((0,),(250,)), mode='symmetric')
# plt.subplot(2,3,5)
# plt.imshow(img4, cmap='gray')
# # 5 - "wrap" padding
# img5 = np.pad(img, ((0,100),(0,0)), mode='wrap')
# plt.subplot(2,3,6)
# plt.imshow(img5, cmap='gray')
# plt.show()

# Primer 6.2
# Nad slikom lena.png primeniti filtar aritmetiˇcki usrednjivaˇc: 1. veliˇcine 101x101, raˇcunaju´ci tzv.
# full konvoluciju 2. veliˇcine 11x11, sa zadržavanjem originalne dimenzije slike i koriš´cenjem zero
# padding-a 4. veliˇcine 7x7, sa zadržavanjem originalne dimenzije slike i koriš´cenjem simetriˇcnog
# proširenja slike 5. veliˇcine 3x3, sa zadržavanjem originalne dimenzije slike i koriš´cenjem proširenja
# slike ponavljanjem iviˇcnih piksela

# from scipy import signal # za koriscenje scipy.signal.convolve2d funkcije
# from scipy import ndimage # za koriscenje scipy.ndimage.convolve funkcije

# img = io.imread('test_slike_omv/lena.png')
# plt.imshow(img, cmap='gray')
# # usrednjivac dimenzije 101x101
# w = np.ones((101,101))/101**2
# # "full" konvolucija
# img1 = signal.convolve2d(img, w, mode='full')
# print(img1.shape)
# plt.imshow(img1, cmap='gray')
# plt.show()

# # usrednjivac dimenzije 11x11
# w = np.ones((11,11))/11**2
# # "same" konvolucija
# img2 = signal.convolve2d(img, w, mode='same')
# print(img2.shape)
# plt.imshow(img2, cmap='gray')
# plt.show()

# # usrednjivac dimenzije 7x7
# w = np.ones((7,7))/49
# # "same" symmetric konvolucija
# img3 = signal.convolve2d(img, w, mode='same', boundary='symmetric')
# print(img3.shape)
# plt.imshow(img3, cmap='gray')
# plt.show()

# # usrednjivac dimenzije 3x3
# w = np.ones((3,3))/9
# # rucno prosirenje sa 'edge' parametrom
# img_p = np.pad(img,(1,1),mode='edge')
# # "valid" konvolucija
# img4 = signal.convolve2d(img_p, w, mode='valid')
# print(img4.shape)
# plt.imshow(img4, cmap='gray')
# plt.show()

# # usrednjivac dimenzije 3x3
# w = np.ones((3,3))/9
# # konvolucija sa 'nearest' prosirenjem - 'nearest' u okviry ndimage.convolve odgovara 'edge' u okviru np.pad
# img5 = ndimage.convolve(img, w, mode='nearest', output='float')
# print(img5.shape)
# plt.imshow(img5, cmap='gray')
# plt.show()

# Primer 6.3
# Nad slikom boat.png upotrebiti Laplasijan filtar koji posmatra 4 suseda. Sliku izoštriti upotrebom
# Laplasijan filtra. Prikazati dobijene rezultate.
# Laplasijan za izdvajanje detalja je dat kao:
#
#     |  0 −1 0 |
# L = | −1  4−1 |
#     |  0 −1 0 |

# img = io.imread('test_slike_omv/boat.png')
# plt.imshow(img, cmap='gray')
# plt.show()
# L = np.array([[0,-1,0],[-1,4,-1],[0,-1,0]])
# img_lap = ndimage.convolve(img.astype('float'), L, mode='reflect')
# # plt.imshow(img_lap, cmap='gray')
# # plt.show()
# img_detail_enhanced = img + img_lap
# plt.imshow(img_detail_enhanced, vmin=0, vmax=255, cmap='gray')
# plt.show()
# L_izostravanje = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
# img_izostren = ndimage.convolve(img.astype('float'), L_izostravanje,mode='reflect')
# plt.imshow(img_izostren, vmin=0, vmax=255, cmap='gray')
# plt.show()

# Primer 6.4
# Odrediti sliku gradijenata na slici deda_mraz.png upotrebom upotrebom Robertsovog krosgradijentnog operatora (engl. Roberts cross operator). Odrediti sliku ivica iz dobijene slike binarizovanjem sa pragom T=30.
# Robertsov kross operator je dat kao par kernela:
# r1 =
# 
# −1 0
# 0 +1
# 
# i r2 =
# 
# 0 −1
# +1 0 
# a gradijentna slika se formira kao:
# G =
# q
# G2
# 1 + G2
# 2
# gde su G1 i G2 slike nastale filtriranjem ulazne slike sa kernelima r1 i r2.

# img = io.imread('test_slike_omv/peppers.png')
# plt.imshow(img, cmap='gray')
# plt.show()
#
# r1 = np.array([[1, 0], [0, -1]])
# r2 = np.array([[0, 1], [-1, 0]])
# img = img.astype('float')
# G1 = ndimage.convolve(img, r1, mode='reflect')
# plt.imshow(G1, cmap='gray')
# plt.show()
# G2 = ndimage.convolve(img, r2, mode='reflect')
# plt.imshow(G2, cmap='gray')
# plt.show()
# G = np.sqrt(G1**2+G2**2)
# plt.imshow(G, cmap='gray')
# plt.show()
# # binarizacija pragovanjem
# T = 30
# G_edge = G > T
# plt.imshow(G_edge, cmap='gray')
# plt.show()

# Primer 6.5 preskocen


# Zadaci za samostalnu vežbu

# Zadatak 6.1
# Napisati funkciju lapFilt koja implementira izoštravanje slike Laplasijan filtrom koji uzima u obzir 8 suseda bez koriš´cenja ugradenih funkcija za filtriranje. Funkciju upotrebiti nad proizvoljnom ¯
# slikom.

# img = io.imread('test_slike_omv/peppers.png')
# plt.imshow(img, cmap='gray')
# plt.show()
#
# def lapFilt(img):
#     L = np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])
#     img_filt = ndimage.convolve(img,L,mode='reflect')
#     return img_filt
#
# plt.imshow(lapFilt(img), cmap='gray')
# plt.show()

# Zadatak 6.3
# Napisati funkciju detectEdge koja pronalazi piksele kojima odgovaraju jaki gradijenti. Funkcija
# treba da ima dva ulazna argumenta. Prvi ulazni argument je slika, dok drugi predstavlja prag.
# Izlaz funkcije treba da bude binarna maska koja ima vrednost 1 na mestima gde je intenzitet gradijenta bio ve´ci od praga, a 0 na ostalim mestima. Za procenu gradijenta koristiti Sobelov (engl.
# Sobel) gradijentni operator definisan slede´cim maskama:
# Sy =
# 
# 
# −1 −2 −1
# 0 0 0
# 1 2 1
# 
#  i Sx =
# 
# 
# −1 0 1
# −2 0 2
# −1 0 1
# 
# 

def detectEdge(img, T):
    img = img.astype('float')
    Sy = np.array(([-1, -2, -1], [0, 0, 0], [1, 2, 1]))
    Sx = np.array(([-1, 0, 1], [-2, 0, 2], [-1, 0, 1]))
    Gy = ndimage.convolve(img,Sy,mode='reflect')
    plt.imshow(Gy, cmap='gray')
    plt.show()
    Gx = ndimage.convolve(img, Sx, mode='reflect')
    plt.imshow(Gx, cmap='gray')
    plt.show()
    G = np.sqrt(Gy**2+Gx**2)
    plt.imshow(G, cmap='gray')
    plt.show()
    img_edges = G > T
    return img_edges

img = io.imread('test_slike_omv/peppers.png')
plt.imshow(detectEdge(img,100), cmap='gray')
plt.show()