# test_var_1 = 1  # int
# test_var_2 = 2.4  # float
# test_var_3 = "+381 Robotics"  # string
#
# print('Promenljiva: "{0}" je tipa: "{1}"'.format(test_var_3, type(test_var_3)))
# print(f'Promenljiva: "{test_var_2}" je tipa: "{type(test_var_2)}"')
#
# zbir = test_var_1 + test_var_2
#
# print(zbir)
# print(type(zbir))

cnt = 0

# if cnt < 10:
#     cnt += 1
# else:
#     cnt -= 1
# print(cnt)

# uslov = True
#
# if uslov and cnt != 0:
#     print('Zadovoljeno')
# elif uslov or cnt != 0:
#     print('Polovicno')
# else:
#     print('Nije zadovoljeno')


# def minimum(a, b):
#     if a < b:
#         return a
#     else:
#         return b
#
#
# def minimum2(a, b):
#     if a < b:
#         return a
#     return b
#
#
# va = 5
# vb = 4
#
# vc = minimum(va, vb)
# vd = minimum2(va, vb)
# print(vc, vd)
#
#
# def plus_minus(a, b):
#     return a + b, a - b
#
# print(plus_minus(va,vb))

# for i in range(0,9):
#     print(i)

# for j in range(0,9,2):
#     print(j)

# for cnt in range(4):
#     print(cnt)

# while cnt < 10:
#     cnt += 1
#     if cnt % 2 != 0:
#         continue
#     print(cnt)

# a = [0,2,4,6,8]
# print(a)
#
# a[0] = 10
# a[-2] = 5
# print (a)

# a=[0,0,0,0,0,0,0,0,0,0,0]
#
# for cnt in range(11):
#     a[cnt] = cnt**2
# print(a[1:5])
# print(a[:3])
# print(a[:-3])
# print(a[-3:])
# print(a[::3])

# a = [1, 2, 3]
# b = [4, 5, 6]
# c = a + b
# c = a * 4
# a.append(4)
# a.extend(b)
# print(a)
# a.pop(-1)
# print(a)
# a.pop(-1)
# print(a)
# a.pop(-1)
# print(a)
# a.pop(-1)
# print(a)
# while len(a) != 0:
#     print(a[-1])
#     a.pop(-1)


# tup = (1, 'a', 2, 'b')
# print (len(tup))
# print(tup)

# def saberi_liste(a,b):
#     c = []
#     for ai,bi in zip(a,b):
#         c.append(ai+bi)
#     return c
#
# a = [1,2,3]
# b = [4,5,6]
# c = saberi_liste(a,b)
# # print(list(zip(a,b)))
#
# c1 = [ai+bi for (ai,bi) in zip(a,b)]
# print(c1)

import numpy as np

# a = np.array([1, 2, 3])
# b = np.array([4, 5, 6])
# c = a + b
# print(c)
#
# print(np.maximum([1, 0, 2], [3, -1, 4]))

import matplotlib.pyplot as plt

# t = np.arange(0, 1, 1 * 10 **(-6))
# func = 1 - 1 * np.e ** (-8*t)
# plt.figure()
# plt.plot(t, func, linewidth=1)
# plt.title('Signal')
# plt.show()

print("4.1 Zadatak 1.1")
# Napisati funkciju koja raˇcuna sumu kvadrata prosledene liste. Napraviti listu sa elemenatima  ̄
# [1,2,3] i testirati funkciju

def suma_kvadrata(a):
    suma = 0
    for i in range(len(a)):
        suma+=a[i]
    return suma
print("4.2 Zadatak 1.2")
# Ponoviti prethodni zadatak, ali koriste ́ci numpy niz umesto liste. Iskoristiti mogu ́cnosti NumPy
# paketa.

def suma_kvadrata_np(a):
    return np.sum(a)

a1 = [1,2,3]
print(suma_kvadrata(a1))
a2 = np.array([1,2,3])
print(suma_kvadrata_np(a2))

print("4.3 Zadatak 1.3")
# Napisati funkciju koja proverava da li elementi u listi imaju rastu ́ci trend (svaki naredni element
# je ve ́ci od prethodnog). Funkciju proveriti za slede ́ce liste

a31 = [1, 3, 4, 6, 7, 9]
a32 = [-1, 2, 4, 1, 6]
a33 = [2, 10, 11, 12]
a34 = [3, 2, 1, 4, 5]

def rastuci_trend(a):
    if a[1]<a[0]:
        return False
    for i in range(2,len(a)):
        if a[i]<a[i-1]:
            return False
    return True

print(rastuci_trend(a31))
print(rastuci_trend(a32))
print(rastuci_trend(a33))
print(rastuci_trend(a34))

print("4.4 Zadatak 1.4")
#Napisati funkciju koja proverava da li je uneta reˇc palindrom. Funkciju isprobati nad reˇcima
a41 = "vrata"
a42 = "potop"
a43 = "knjiga"
a44 = "pop"
a45 = "rotor"
a46 = "maam"

def is_palindrom(a):
    for i in range(len(a)//2):
        if a[i]!=a[-i-1]:
            return False
    return True

print(is_palindrom(a41))
print(is_palindrom(a42))
print(is_palindrom(a43))
print(is_palindrom(a44))
print(is_palindrom(a45))
print(is_palindrom(a46))