import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import matplotlib.pyplot as plt
from skimage import io

img = np.full((256, 256), fill_value=63)  # moze i kao np.zeros()+63 / np.ones()*63
img[32:32 + 192, 32:32 + 192] = 127
img[64:64 + 128, 64:64 + 128] = 191
# hist, bin_edges = np.histogram(img, bins=np.arange(257))
# fig = make_subplots(rows=1, cols=2)
# fig.add_trace(go.Image(z=np.stack((img, img, img), axis=2)), row=1, col=1)
# fig.add_trace(go.Scatter(x=np.arange(256), y=hist, mode='lines',name='hist',line=dict(color='blue')), row=1, col=2)
# fig.show()

import plotly.express as px


# rng = np.random.default_rng()
# noise_im = rng.standard_normal(size=img.shape)
# fig = px.imshow(noise_im, color_continuous_scale='gray')
# fig.show()
# img_noisy = img + noise_im * np.sqrt(100) + 0
# img_noisy[img_noisy > 255] = 255
# img_noisy[img_noisy < 0] = 0
# hist_n, bin_edges = np.histogram(img_noisy, bins=np.arange(257))
# fig = make_subplots(rows=1, cols=2)
# fig.add_trace(go.Image(z=np.stack((img_noisy, img_noisy, img_noisy), axis=2)), row=1, col=1)
# fig.add_trace(go.Scatter(x=np.arange(256), y=hist_n, mode='lines', name='hist_noisy', line=dict(color='blue')), row=1,
#               col=2)
# # fig.show()
# psnr = 10*np.log10(255**2/((img - img_noisy)**2).mean())
# print(psnr)

# Primer 8.2
# Koriste´ci zašumljenu sliku iz prethodnog primera, ukloniti šum upotrebom aritmetiˇckog usrednjivaˇca veliˇcina: * 3x3 * 5x5 * 11x11
# Koriste´ci PSNR odrediti koji filtar je dao objektivno najbolje otklanjanje.

# noise = np.random.default_rng().standard_normal(img.shape)
# img_noisy = img + noise * np.sqrt(100) + 0
# img_noisy[img_noisy > 255] = 255
# img_noisy[img_noisy < 0] = 0
# plt.figure(figsize=(15,5))
# plt.subplot(1,4,1)
# plt.imshow(img_noisy, cmap='gray')

def psnr(img, img_n):
    return 10 * np.log10(255 ** 2 / ((img - img_n) ** 2).mean())


from scipy import ndimage

# w3 = np.ones((3,3))/3**2
# w5 = np.ones((5,5))/5**2
# w11 = np.ones((11,11))/11**2
#
# img_3=ndimage.convolve(img_noisy.astype('float'), w3, mode='mirror')
# img_5=ndimage.convolve(img_noisy.astype('float'), w5, mode='mirror')
# img_11=ndimage.convolve(img_noisy.astype('float'), w11, mode='mirror')
# plt.subplot(1,4,2)
# plt.imshow(img_3, cmap='gray')
# plt.subplot(1,4,3)
# plt.imshow(img_5, cmap='gray')
# plt.subplot(1,4,4)
# plt.imshow(img_11, cmap='gray')
# plt.show()
# print(psnr(img,img_3))
# print(psnr(img,img_5))
# print(psnr(img,img_11))

# Primer 8.4
# Napraviti demo sliku dimenzije 256x256 sa dva kvadrata u sredini (jedan u drugom) koji su sa
# nijansama 127 i 191. Postaviti pozadinu na 63. Prikazati njen histogram.
# Na ovu sliku dodati impulsni: * so * biber * so i biber
# 11
# šum gustine 20%. Prikazati zašumljene slike i njihove histograme.

p = np.random.default_rng().uniform(size=img.shape)
p0 = 0.2

img_so = img.copy()
img_so[p < p0] = 255
hist, bin_edges = np.histogram(img_so, bins=np.arange(257))
img_biber = img.copy()
img_biber[p < p0] = 0
hist_b = np.histogram(img_biber, bins=np.arange(257))[0]
img_sb = img.copy()
img_sb[p<p0/2] = 255
img_sb[(p>p0/2) & (p<p0)] = 0
hist_sb = np.histogram(img_sb, bins=np.arange(257))[0]

# plt.figure(figsize=(8, 8))
# plt.subplot(3, 2, 1)
# plt.imshow(img_so, cmap='gray', vmin=0, vmax=255)
# plt.subplot(3, 2, 2)
# plt.plot(np.arange(256), hist)
# plt.subplot(3, 2, 3)
# plt.imshow(img_biber, cmap='gray', vmin=0, vmax=255)
# plt.subplot(3, 2, 4)
# plt.plot(np.arange(256), hist_b)
# plt.subplot(3, 2, 5)
# plt.imshow(img_sb, cmap='gray', vmin=0, vmax=255)
# plt.subplot(3, 2, 6)
# plt.plot(np.arange(256), hist_sb)
# plt.show()
#
# print(psnr(img,img_so))
# print(psnr(img,img_biber))
# print(psnr(img,img_sb))

# Primer 8.6
# Sliku img_sb iz primera 8.4 filtrirati median filtrom statistike poretka dimenzije 3x3.
# Filtri statistike poretka (engl. order-statistics filters) predstavljaju grupu nelinearnih filtara koji sortiraju elemente iz posmatranog regiona datim nekom maskom u rastu´cem poretku i vra´caju vrednost odredenu zahtevanim indeksom u okviru ovog poretka. ¯
# U zavisnosti od vrednosti indeksa ovim naˇcinom se mogu definisati 3 osnovne vrste:
# 20
# • Min filtar ˆ
# f(x, y) = min(s,t)∈Sx,y
# g(s, t) - indeks je 0, tj. najmanja vrednost u poretku.
# • Max filtar ˆ
# f(x, y) = max(s,t)∈Sx,y
# g(s, t) - indeks je indeks poslednjeg elementa, tj. najve´ca
# vrednost u poretku.
# • Median filtar ˆ
# f(x, y) = median(s,t)∈Sx,y
# g(s, t) - indeks je indeks centralnog elementa, tj. vrednost koja je ve´ca od 50% preostalih piksela i 50% manja od vrednosti preostalih piksela.
# Opšta funkcija za filtre poretka je implementirana u scipy.signal.order_filter. Ulazni parametri su
# N-dimenzioni niz koji se filtrira, domain parametar koji oznaˇcava masku po kojoj se posmatraju
# elementi u regionu (sa vrednostima 0 i 1) i rank koji obeležava indeks iz poretka od interesa.
# Ukoliko posmatramo sve elemente 3x3 regiona, domain je potrebno definisati kao matricu sa svim
# jedinicama, a za odredivanje median vrednosti, indeks od interesa bi´ce 4 (centralni u opsegu od 0 ¯
# do 8).
# Shodno tome, Min filtar je definisan sa indeksom 0 a Max sa indeksom 8.
# Funkcija uvek radi zero-padding.

from scipy import signal

# region = np.ones((3,3))
# # img_f = signal.order_filter(img_sb,domain=region,rank=4)
# img_f = ndimage.median_filter(img_sb,footprint=region,mode='reflect')
# plt.imshow(img_f, cmap='gray')
# plt.show()
# print(psnr(img,img_f))

# Primer 8.7
# Uˇcitati sliku lena.png i dodati joj AWGN varijanse 100 i srednje vrednosti 0. Nad ovom zašumljenom slikom dodati impulsni so i biber šum gustine 5%.
# Za potrebe uklanjanja ovog kombinovanog šuma potrebno je napisati funkciju
# alphaTrimmedMeanFilt koja implementira alfa-trimovani usrednjivaˇc (engl. alpha-trimmed
# mean) dat formulom:
# ˆ
# f(x, y) = 1
# mn − α ∑
# (s,t)∈Sx,y
# gr(s, t)
# gde (s, t) predstavljaju prostorne koordinate regiona Sx,y posmatran oko centra pozicije (x, y), gr
# su ulazni intenziteti nakon odbacivanja vrednosti a m, n predstavljaju dimenzije regiona koji se
# posmatra. α predstavlja parametar koliko se ukupno najmanjih i najve´cih vrednosti odbacuje.
# Sliku filtrirati sa posmatranim regionom 3x3 i alfa parametrom od 4. Prikazati rezulat i uporediti
# PSNR pre i posle filtriranja.
# 23
# Filtar kombinuje prednosti aritmetiˇckog usrednjivaˇca i median filtra. Ako posmatramo region postavljen u okolini nekog piksela, vrši se prvo sortiranje vrednosti u rastu´cem poretku (kao i kod
# filtara statistike poretka), nakon ˇcega odbacujemo α/2 vrednosti sa leve i desne strane (α/2 najmanjih i najve´cih). Ovin naˇcinom se borimo protiv impulsnog šuma. Od preostalih vrednosti tražimo
# srednju vrednost, ˇcime pokušavamo da potisnemo Gausov šum.

img = io.imread('test_slike_omv/lena.png')
noise = np.random.default_rng().standard_normal(img.shape) * np.sqrt(100) + 0
img_n = img.copy()
img_n = img_n + noise
img_n[img_n>255] = 255
img_n[img_n<0] = 0
p = np.random.default_rng().uniform(size=img_n.shape)
p0 = 0.05
img_n[p<p0/2] = 255
img_n[(p>p0/2)&(p<p0)] = 0
plt.figure(figsize=(16,8))
plt.subplot(1,3,1)
plt.imshow(img,cmap='gray')
plt.subplot(1,3,2)
plt.imshow(img_n,cmap='gray')

import math
def alphaTrimmedMeanFilt(g, S_shape, alpha):
    f = np.zeros(g.shape)
    pad_rows = math.floor(S_shape[0]/2)
    pad_cols = math.floor(S_shape[1]/2)
    img_p = np.pad(g, ((pad_rows,),(pad_cols,)), mode='symmetric')
    for row in range(f.shape[0]):
        for col in range(f.shape[1]):
            region = img_p[row:row+S_shape[0], col:col+S_shape[1]]
            region = np.sort(region, axis = None)
            res = region[alpha//2:-alpha//2].mean()
            f[row, col] = res
    return f

S_shape = (3,3)
img_f = alphaTrimmedMeanFilt(img_n, S_shape, alpha=4)
plt.subplot(1,3,3)
plt.imshow(img_f,cmap='gray')

print(psnr(img,img_n))
print(psnr(img,img_f))
plt.show()