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

img = io.imread('test_slike_omv/cat.png')
img_flipped = img.copy()
img_flipped_2 = img.copy()

img_flipped = img_flipped[-1:0:-1,:]
img_flipped_2 = img_flipped_2[:,::-1]
img_flipud = np.flipud(img)
img_fliplr = np.fliplr(img)

plt.figure(figsize=(16,4))
plt.subplot(1,5,1)
plt.imshow(img, cmap='gray')
plt.subplot(1,5,2)
plt.imshow(img_flipped, cmap='gray')
plt.subplot(1,5,3)
plt.imshow(img_flipped_2, cmap='gray')
plt.subplot(1,5,4)
plt.imshow(img_flipud, cmap='gray')
plt.subplot(1,5,5)
plt.imshow(img_fliplr, cmap='gray')

plt.show()

# 8 Zadaci za samostalnu vežbu (strana 29)
# Zadatak 2.1