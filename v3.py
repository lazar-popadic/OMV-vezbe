import numpy as np
from skimage import io
from skimage.transform import rescale, resize
import matplotlib.pyplot as plt

# Primer 3.1
# U promenljivu img uˇcitati head.png. Prikazati uˇcitanu sliku sa automatskim podešavanjem tonske
# skale. Odrediti kvantizovanu sliku koja predstavlja originalnu sliku sa smanjenim brojem mogu-
#  ́cih nivoa intenziteta. Koristiti 3 bita informacije za odredivanje kvantizovane slike i uniformno  ̄
# odredene nivoe.  ̄

# img = io.imread('test_slike_omv/boat.png')
# plt.imshow(img, cmap='gray')
# plt.show()
#
# bit_num = 3
# bin_string = '1' * bit_num + '0' * (1 - bit_num)
# quant_mask = int(bin_string, 2)
# Q = img & quant_mask
# print('Unikatne vrednosti u okviru kvantizovane slike: {}'.format(np.unique(Q)))
# Q = np.round(Q/quant_mask*255).astype('uint8')
# print('Unikatne vrednosti u okviru kvantizovane slike: {}'.format(np.unique(Q)))
#
# plt.imshow(img, cmap='gray')
# plt.show()

# Primer 3.2
# U promenljivu img uˇcitati sliku baboon.png. Upotrebom funkcije rescale iz skimage.transform
# modula veliˇcinu slike umanjiti 4, 8 i 16 puta. Rezultat snimiti u promenljive img_2a, img_3a i
#
# img_4a. Koriš ́cenjem funkcije resize iz skimage.transform uve ́cati ove slike na dimenziju origi-
# nalne slike img i rezultate smestiti u img_2b, img_3b i img_4b. Prilikom koriš ́cenja funkcije koristiti
#
# bilinearnu interpolaciju. Ispisati pojedinaˇcne veliˇcine dobijenih slika i prikazati finalne.

# img = io.imread('test_slike_omv/baboon.png')
# print('Dimenzija slike img: {}'.format(img.shape))
#
# img_2a = rescale(img, 1 / 4, order=1, preserve_range=True)
# print('Dimenzija slike img_2a: {}'.format(img_2a.shape))
# img_2b = resize(img_2a, img.shape, order=1, preserve_range=True)
# print('Dimenzija slike img_2b: {}'.format(img_2b.shape))
#
# plt.figure(figsize=(10, 5))
# plt.subplot(1, 2, 1)
# plt.title('Originalna slika')
# plt.imshow(img, cmap='gray')
# plt.subplot(1, 2, 2)
# plt.title('Slika umanjena pa uvecana 4x')
# plt.imshow(img_2b, cmap='gray')
# plt.show()


# 4.1 Primer 3.3
#
# Ilustrovati Euklidska rastojanja piksela u odnosu na centralni piksel u slici dimenzije 41x41. Kori-
# stiti numpy funkciju meshgrid za odredivanje pojedinaˇcnih koordinata u matrici.

# promene_po_row = np.arange(-20, 21, 1)
# promene_po_col = np.arange(-20, 21, 1)
# x, y = np.meshgrid(promene_po_col, promene_po_row)
#
# print(x)
# print('---')
# print(y)
# print('---')
# print('Koordinata centralnog piksela je: {}'.format((promene_po_row[20],promene_po_col[20])))
# De = np.sqrt((x-0)**2 + (y-0)**2) # -0 je jer racunamo udaljenost centralnog␣piksela na koordinati (0,0) od svih ostalih
#
# plt.imshow(De, cmap='gray')
# plt.show()

# y = np.arange(-20, 21, 1).reshape(-1, 1) # podesavanje da vektor bude u obliku kolone, drugi parametar je broj kolona, prvi parametar oznacava da se vrednost prolagodi sama
# x = np.arange(-20, 21, 1)
# De = np.sqrt((x-0)**2 + (y-0)**2)
#
# plt.imshow(De, cmap='gray')
# plt.show()

# 4.2 Primer 3.4
#
# Koriste ́ci Euklidsko rastojanje od centralnog piksela slike lena.png kreirati masku gde je to rasto-
# janje manje od poluširine slike. Izvršiti potom maskiranje.

# img = io.imread('test_slike_omv/lena.png')
#
# y = np.arange(0, img.shape[0], 1).reshape(-1, 1) - np.floor(img.shape[0] / 2)
# x = np.arange(0, img.shape[1], 1) - np.floor(img.shape[1] / 2)
# De = np.sqrt((x - 0) ** 2 + (y - 0) ** 2)
# M = De < np.floor(img.shape[0]/2)
#
# plt.imshow(M, cmap='gray')
# plt.show()
#
# img_masked1 = img * M
#
# plt.imshow(img_masked1, cmap='gray')
# plt.show()


# 5 Zadaci za samostalnu vežbu
# Zadatak 3.1
# U promenljivu img uˇcitati sliku baboon.png. Upotrebom funkcije rescale iz skimage.transform
# modula veliˇcinu slike umanjiti 4, 8 i 16 puta. Rezultat snimiti u promenljive img_2a, img_3a i
# img_4a. Koriš ́cenjem funkcije resize iz skimage.transform uve ́cati ove slike na dimenziju origi-
# nalne slike img i rezultate smestiti u img_2b, img_3b i img_4b. Prilikom koriš ́cenja funkcije koristiti
# interpolaciju najbližim susedom. Ispisati pojedinaˇcne veliˇcine dobijenih slika i prikazati finalne.
# Interpolacija najbližim susedom stvar blok efekat koji se ogleda u grupisanju piksela sa istom
# vrednoš ́cu u pravougaone blokove. Blok efekat nastaje jer se za interpoliranu vrednost uzima
# vrednost najbližeg suseda te dolazi do ponavljanja iste vrednosti, to jest stvaranja blokova.

# img = io.imread('test_slike_omv/baboon.png')
# img_a = rescale(img, 1 / 4, order=0, preserve_range='True')
# # io.imsave('test_slike_omv/baboon_a.png', img_a)
# img_b = resize(img_a, img.shape, order=0, preserve_range='True')
# io.imsave('test_slike_omv/baboon_b.png', img_b)


# Zadatak 3.2
# Pojedini proizvodaˇci softvera za obradu i prikazivanje medicinskih snimaka nude mogu ́cnost uve-  ̄
# liˇcanja samo odredenog dela prikazanog snimka. Ovakav softverski alat uve ́cava region slike koji  ̄
# se nalazi na mestu odredenom pokazivaˇcem (kursorom, mišem). Napisati funkciju  ̄ magnifier koja
# kao ulazne argumente prima originalnu sliku i koordinate pokazivaˇca (proslediti kao tuple vred-
# nost). Funkcija kao izlazni argument treba da vrati 4 puta uve ́can region slike veliˇcine 101 x 101
# piksela sa centrom odredenim koordinatama pokazivaˇca.  ̄
# Dodatak: razmisliti kako obezbediti da funkcija radi ˇcak i ako oko pozicije pokazivaˇca ne može
# da se pronade kvadrat koji  ́ce u potpunosti biti ispunjen pikselima, npr. blizu ivica slike.

# def magnifier(img, coords):
#     img_mag = img.copy()
#     img_mag = img[coords[0] - 50:coords[0] + 51, coords[1] - 50:coords[1] + 51]
#     img_mag = rescale(img_mag, 4, order=1, preserve_range='True')
#     return img_mag
#
#
# img = io.imread('test_slike_omv/baboon.png')
# img_2 = magnifier(img, (60, 175))
# plt.figure(figsize=(10, 5))
# plt.subplot(1, 2, 1)
# plt.imshow(img, cmap='gray')
# plt.subplot(1, 2, 2)
# plt.imshow(img_2, cmap='gray')
# plt.show()

# Zadatak 3.3
#
# Interaktivne transformacije slika mogu biti raˇcunarski zahtevne zbog potencijalno velikih dimen-
# zija snimaka. Smanjenjem veliˇcine originalnog snimka pojednostavljuje ovo raˇcunanje. Na smanje-
# nom snimku se primenjuje ta transformacija, nakon ˇcega se veliˇcina snimka vra ́ca na originalnu.
# Uve ́canje slike se ostvaruje interpolacijom najbližim susedom koja nije raˇcunarski zahtevna ope-
# racija. Na ovaj naˇcin je transformisano manji broj piksela, ˇcime je ostvarena ušteda na raˇcunarskim
# operacijama a mi opet dobijamo neku vizuelnu predstavu šta da oˇcekujemo. Slika pune veliˇcine
# se transformiše tek kada smo zadovoljni oˇcekivanim rezultatom transformacije.
# U okviru zadatka potrebno napisati funkciju fastIntensityResPreview koja kao ulazne argu-
# mente prima originalnu sliku i broj bita. Funkcija treba da smanji sliku 5 puta koriste ́ci in-
# terpolaciju sa najbližim susedom, nakon ˇcega je potrebno izvršiti kvantizaciju na broj nivoa koji
# definiše parametar broja bita. Rezultuju ́ca slika se uve ́cava na originalnu dimenziju i prikazuje se.
# Funkciju testirati nad proizvoljnim slikama i brojem bita za kvantizaciju (vrednost od 1 do 8).

# def fastIntensityResPreview(img, bit_num):
#     img_2 = img.copy()
#     img_2 = rescale(img_2, 1 / 5, order=0, preserve_range='True')
#     bin_mask = '1' * bit_num + '0' * (8 - bit_num)
#     bin_mask_int = int(bin_mask, 2)
#     img_2 = img_2 & bin_mask_int
#     img_2 = np.round(img_2/bin_mask_int*255).astype('uint8')
#     return img_2
#
#
# img = io.imread('test_slike_omv/baboon.png')
# img_2 = fastIntensityResPreview(img, 2)
# plt.figure(figsize=(10, 5))
# plt.subplot(1, 2, 1)
# plt.imshow(img, cmap='gray')
# plt.subplot(1, 2, 2)
# plt.imshow(img_2, cmap='gray')
# plt.show()

# Zadatak 3.4
# Vode ́ci se Primerom 3.3, odrediti vizuelan prikaz za City-block i Chessboard rastojanja u slici
# dimenzije 51x51. Za raˇcunanje apsolutne vrednosti koristiti np.abs funkciju, a za raˇcunanje mak-
# simalne vrednosti koristiti np.maximum. Koje figure se mogu uoˇciti za ove tipove rastojanja?

y = np.arange(-25, 26, 1).reshape(-1, 1)
x = np.arange(-25, 26, 1)
Euklidsko = np.sqrt(x ** 2 + y ** 2)
City_block = np.abs(x) + np.abs(y)
Chessboard = np.maximum(np.abs(x), np.abs(y))

plt.figure(figsize=(10, 5))
plt.subplot(1, 3, 1)
plt.imshow(Euklidsko, cmap='gray')
plt.subplot(1, 3, 2)
plt.imshow(City_block, cmap='gray')
plt.subplot(1, 3, 3)
plt.imshow(Chessboard, cmap='gray')
plt.show()
