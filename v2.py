print("Vezbe 2")

import plotly.express as px
from skimage import io
import matplotlib.pyplot as plt
import numpy as np

# Primer 1

# img = io.imread('test_slike_omv/lena.png')

# fig = px.imshow(img, zmin=0, zmax=255, color_continuous_scale='gray')
# # opciono
# # fig.update_layout(coloraxis_showscale=False)
# # fig.update_xaxes(showticklabels=False)
# # fig.update_yaxes(showticklabels=False)
# # prikaz
# fig.show()

# plt.imshow(img, cmap='gray', vmin=0, vmax=255)
# plt.show()


# Primer 2

# img = io.imread('test_slike_omv/baboon.png')
#
# plt.imshow(img, cmap='gray', vmin=0, vmax=255)
# plt.show()
#
# img_2 = img.astype('float') + 100
#
# plt.imshow(img_2, cmap='gray', vmin=0, vmax=255)
# plt.show()
#
# img_3 = img.astype('float') - 100
#
# plt.imshow(img_3, cmap='gray', vmin=0, vmax=255)
# plt.show()


# Primer 3
# img = io.imread('test_slike_omv/zelda.png')
#
# plt.imshow(img, cmap='gray', vmin=0, vmax=255)
# plt.show()
#
# img_norm = (img - img.min()) / (img.max() - img.min())  # sa [img.min, img.max] na [0, 1]
#
# plt.imshow(img_norm, cmap='gray', vmin=0, vmax=255)
# plt.show()


# Primer 4
# img = io.imread('test_slike_omv/peppers.png')
# plt.imshow(img, cmap='gray')
# plt.show()

# io.imsave('test_slike_omv/peppers2.png', img)


# Primer 5

# img = io.imread('test_slike_omv/boat.png')
#
# plt.figure(figsize=(10,5))
# plt.subplot(1,2,1)
# plt.imshow(img, cmap='gray')
#
# N, M = img.shape
#
# img_2 = img.copy()
# img_2 = img_2[round(N * 0.25):round(N * 0.75),
#         round(M * 0.25):round(M * 0.75)]
#
# plt.subplot(1,2,2)
# plt.imshow(img_2, cmap='gray')
# plt.show()


# Primer 6

# img = io.imread('test_slike_omv/cat.png')
# img_flipped = img.copy()
# img_flipped_2 = img.copy()
#
# img_flipped = img_flipped[-1:0:-1,:]
# img_flipped_2 = img_flipped_2[:,::-1]
# img_flipud = np.flipud(img)
# img_fliplr = np.fliplr(img)
#
# plt.figure(figsize=(16,4))
# plt.subplot(1,5,1)
# plt.imshow(img, cmap='gray')
# plt.subplot(1,5,2)
# plt.imshow(img_flipped, cmap='gray')
# plt.subplot(1,5,3)
# plt.imshow(img_flipped_2, cmap='gray')
# plt.subplot(1,5,4)
# plt.imshow(img_flipud, cmap='gray')
# plt.subplot(1,5,5)
# plt.imshow(img_fliplr, cmap='gray')
#
# plt.show()


# 8 Zadaci za samostalnu vežbu (strana 29)
# Zadatak 2.1
#

# Uˇcitati i prikazati CT (engl. computerized tomography) snimak ct_1.png. Slika je tipa uint16 (neo-
# znaˇceni celobrojni tip dužine 16 bita) što znaˇci da je opseg vrednosti piksela od 0 do 65535. Ovaj
#
# opseg se preslikava u više nijansi sive boje nego što ljudsko oko može da razlikuje. Medutim,  ̄
# preko podešavanja granica palete mogu ́ce je prikazati samo odredene delove dinamiˇckog opsega  ̄
# slike ˇcime se naglašavaju odredene vrste tkiva prilikom prikaza. Prikazati sliku sa naglašenim ko-  ̄
# štanim tkivima podešavanjem granica palete na 3000 i 4000 i sa naglašenim mekim tkivima putem
# podešavanja granica palete na 2800 i 3150.

# img = io.imread('test_slike_omv/cat.png')
# plt.imshow(img, cmap='gray', vmin=100, vmax=255)
# plt.show()


# Zadatak 2.2
# Jedan od standardnih naˇcina za interaktivno podešavanje kontrasta kod medicinskih snimaka je
# podešavanje dva parametra: prozora i nivoa (engl. window and level). Parametar nivo odredu-  ̄
# je vrednost koja  ́ce se prikazati kao srednja nijansa sive (na polovini palete). Parametar prozor
# odreduje širinu opsega koji  ́ce biti obuhva ́cen paletom sivih tonova. Ako se vrednost parame-  ̄
# tra nivo obeleži sa L, a vrednost parametra prozor obeleži sa W, intenzitet koji predstavlja donju
# granicu palete  ́ce biti L-W/2, a nivo koji odreduje gornju granicu  ́ce biti  ̄ L+W/2. Napisati funkciju
# setWindowLevel koja prikazuje sliku sa paletom sivih tonova podešenom prema nivou i prozoru.
# Funkciju pozvati nad MRI (engl. magnetic resonance imaging) snimkom mr_1.png. Sliku prikazati
# sa parametrima koji istiˇcu guste strukture, gde je W = 3000, a L = 2500, zatim sa parametrima koji
# istiˇcu mekša tkiva, gde je W = 2100, a L = 1450.

# def set_windows_level(img,l,w):
#     plt.imshow(img, cmap='gray', vmin = l - w / 2, vmax = l + w / 2)
#     plt.show()
#
# img = io.imread('test_slike_omv/cat.png')
# set_windows_level(img, 100,100)


# Zadatak 2.3
# Polje zraˇcenja se prilikom rendgenskih snimanja ograniˇcava samo na anatomiju koju je potrebno
# snimiti, što za rezultat ima pojavu tamnih regiona u snimku. Ovi regioni predstavljaju suvišan
# deo snimka koji može da ometa dijagnostiku, te ih je nekada potrebno odstraniti. Uˇcitati i pri-
# kazati neobradene digitalne rendgenske snimke  ̄ rtg_1.png, rtg_2.png i rtg_3.png. Sa uˇcitanih
# snimaka ruˇcno odstraniti delove koji ne predstavljaju direktno ozraˇcenu anatomiju. Iseˇcene delo-
# ve prikazati i saˇcuvati kao nove slike sa proizvoljnim imenima.

# img = io.imread('test_slike_omv/rtg_4.png')
# img_cp = img.copy()
# N, M = img_cp.shape
# img_cp = img_cp[:round(N*0.2),
#          round(M*0.05):round(M*0.95)]
# plt.imshow(img_cp,cmap='gray')
# # plt.show()
# io.imsave('test_slike_omv/rtg_41.png',img_cp)

# Zadatak 2.4
# Razliˇciti proizvodaˇci medicinskih ure  ̄ daja nude snimke koji imaju razliˇcite dinamiˇcke opsege. Da  ̄
# bi se omogu ́cila unificirana obrada snimaka, potrebno ih je dovesti na neki unapred poznati opseg.
# Napisati funkciju normalizeImRange koja opseg ulazne slike preraˇcunava na [0, 1]. Napisati funk-
# ciju setRange koja kao ulazne argumente prima minimalnu i maksimalnu vrednost opsega na koji
# treba preraˇcunati opseg ulazne slike. Izlaz funkcije treba da bude slika sa podešenim opsegom. U
# razvoju funkcije setRange iskoristiti funkciju normalizeImRange. Ispravnost funkcije proveriti na
# nekoliko proizvoljnih slika (proveriti njihove nove min i max vrednosti).

# def normalize_im_range(img):
#     img_n = (img - img.min()) / (img.max() - img.min())
#     return img_n
#
#
# img = io.imread('test_slike_omv/zelda.png')
# plt.figure(figsize=(10, 5))
# plt.subplot(1, 2, 1)
# plt.imshow(img, cmap='gray')
# img = normalize_im_range(img)
# plt.subplot(1, 2, 2)
# plt.imshow(img, cmap='gray')
# plt.show()

# Zadatak 2.5
# Prilikom akvizicije rendgenskih snimaka plu ́ca pacijenti se pozicioniraju ledima prema izvoru  ̄
# zraˇcenja i licem prema detektoru. Na ovako prikupljenim snimcima  ́ce desna strana pacijenta biti
# prikazana na desnoj strani snimka. Pošto je uobiˇcajeno da je na desnoj strani snimka prikazana
# leva strana pacijenta, snimak je potrebno preslikati oko vertikalne ose simetrije. Napisati program
# koji ostvaruje navedeno preslikavanje bez upotrebe funkcije fliplr. Funkcionalnost programa
# isprobati na slici rtg_4.png.

# img = io.imread('test_slike_omv/rtg_4.png')
# img_fl = img.copy()
# img_fl = img_fl[:, ::-1]
#
# plt.figure(figsize=(10,5))
# plt.subplot(1,2,1)
# plt.imshow(img, cmap='gray')
# plt.subplot(1,2,2)
# plt.imshow(img_fl, cmap = 'gray')
# plt.show()

# Zadatak 2.6
# Uˇcitati sliku boat.png. Iz uˇcitane slike potrebno je izdvojiti centralnih 70% piksela kao novu sliku,
# pove ́cati njihov intenzitet 2 puta i potom vratiti segment na originalnu poziciju. Vrednosti koji
# prelaze maksimalnu vrednost definisanu tipom originalne slike pragovati na nju.

img = io.imread('test_slike_omv/boat.png')
img_cent = img.copy()
N, M = img_cent.shape
img_cent = img_cent[round(N * 0.15):round(N * 0.85),
           round(M * 0.15):round(M * 0.85)]

img_cent = img_cent.astype('float') * 2.0
img_cent = np.minimum(img_cent, img.max())
img[round(N * 0.15):round(N * 0.85),
           round(M * 0.15):round(M * 0.85)] = img_cent

plt.imshow(img, cmap='gray')
plt.show()
