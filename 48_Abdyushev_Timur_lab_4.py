##################################################################################
#   48 Группа, Абдюшев Тимур, Лабораторная работа 4
#
#   Приближение функций. Вычисление собственных значений и собственных векторов матриц.
#
#   1) По методу наименьших квадратов с использованием ортогональных полиномов Чебышева
#   получить многочлен III степени, аппроксимирующий таблично заданную функцию.
#   Проверить значения функции в точках Xi + h/2.
#   Выписать разложение через полиномы Чебышева.
#
#   2) Найти методом Данилевского собственные значения
#   и собственные векторы симметрических матриц
#
#   3) Используя степенной метод,
#   оценить спектральный радиус данных матриц с погрешность e = 0.001
#
##################################################################################

import numpy as np
from math import cos, sqrt, factorial
from numpy.polynomial import Chebyshev as T


# Вычисление ортогональных полиномов Чебышева
def solve_chebyshev_poly(k, N, t):
    N -= 1
    ns = np.ones(N + 1)
    for i in range(1, N + 1):
        ns[i] = ns[i - 1] * (N - i + 1)

    ts = np.ones(k + 1)
    for i in range(1, k + 1):
        ts[i] = ts[i - 1] * (t - i + 1)

    dl = (2 * k + 1) * ns[k]
    num = 1.0
    stp = (N + k + 1)
    while stp > N:
        num *= stp
        stp -= 1
    k_norm = sqrt(num / dl)

    p = np.zeros(k + 1)
    s = 0

    while s <= k:
        Ck = factorial(k) // factorial(s) // factorial(k - s)
        Cks = factorial(k + s) // factorial(s) // factorial(k)
        p_is = (-1) ** s * Ck * Cks / ns[s]
        p[s] = p_is / k_norm
        s += 1

    return p * ts


# Вычисляем число корней
def count_root(matrix_A, l, r):
    nt, N = matrix_A.shape
    I = np.identity(N)
    H1 = np.zeros(N + 1)
    H2 = np.zeros(N + 1)
    for i in range(1, N + 1):
        H1[i] = np.linalg.det(matrix_A[:i, :i] - l * I[:i, :i])
    for i in range(1, N + 1):
        H2[i] = np.linalg.det(matrix_A[:i, :i] - r * I[:i, :i])
    p_1 = 0
    p_2 = 0
    vct_1 = [1]
    vct_1 += H1
    vct_2 = [1]
    vct_2 += H2
    for i in range(len(vct_1) - 1):
        ft = vct_1[i]
        sd = vct_1[i + 1]
        if (ft > 0 and sd > 0) or (ft < 0 and sd < 0):
            p_1 += 1
    for i in range(len(vct_2) - 1):
        ft = vct_2[i]
        sd = vct_2[i + 1]
        if (ft > 0 and sd > 0) or (ft < 0 and sd < 0):
            p_2 += 1

    return p_1 - p_2


# Вычисление коэффицентов Фурье
def solve_coef(f, N, m):
    N -= 1
    c = np.zeros(m + 1)

    for i in range(m + 1):
        for j in range(N + 1):
            poly = solve_chebyshev_poly(i, N + 1, j)
            c[i] += f[j] * np.sum(poly)
    c[0] = np.sum(f) / (N + 1)
    return c


# Метод Данилевского
def method_Danielskogo(matrix_A, eps = 0.001):
    def c_poly(c, x):
        return x ** 4 - c[0] * x ** 3 - c[1] * x ** 2 - c[2] * x - c[3]

    def c_poly_proizv(c, x):
        return 4 * x ** 3 - 3 * c[0] * x ** 2 - 2 * c[1] * x - c[2]

    def newton(x, c):
        x_prv = x + 1
        while abs(x - x_prv) > eps:
            x_prv = x
            x = x - c_poly(c, x) / c_poly_proizv(c, x)

        return x

    D = np.array(matrix_A)
    nt, N = D.shape
    B = np.identity(N)
    for i in range(N - 1, 0, -1):
        p = D[i, i - 1]
        b = np.identity(N)
        b[i - 1, :] = [-vt / p for vt in D[i, :]]
        b[i - 1, i - 1] /= -p
        b_inv = np.identity(N)
        b_inv[i - 1, :] = D[i, :]
        B = B @ b
        D = b_inv @ D @ b

    c = D[0, :]
    stnd = [-1, 1]
    n_roots = np.linalg.matrix_rank(matrix_A)
    roots = 0
    while roots != n_roots:
        stnd[0] *= 2
        stnd[1] *= 2
        roots = count_root(matrix_A, stnd[0], stnd[1])

    dl = [-3.0, -1.0, -0.1, 2.0, 7.0]
    edges = stnd + dl
    edges.sort()
    eg_values = set()
    for i in range(len(edges)):
        rt = round(newton(edges[i], c), 5)
        eg_values.add(rt)

    eg_values = list(eg_values)
    eg_values.sort()
    eg_vectors = np.identity(N)

    for i, value in enumerate(eg_values):
        t = np.array([[value ** 3], [value ** 2], [value], [1]], float)
        x = B @ t
        x = x.flatten()
        x /= np.sqrt(np.sum(x * x))
        eg_vectors[:, i] = x

    return eg_values, eg_vectors


# Вычисление спектрального радиуса
def solve_radius_spect(matrix_A, Yo, eps=0.001):
    Yk = np.array(Yo)
    l = 1

    while True:
        Yk_prev = Yk
        Yk = matrix_A @ Yk

        l_prev = l
        l = Yk[2] / Yk_prev[2]
        if abs(l - l_prev) <= eps:
            break

    return l


if __name__ == '__main__':
    #   1) По методу наименьших квадратов с использованием ортогональных полиномов Чебышева
    #   получить многочлен III степени, аппроксимирующий таблично заданную функцию.
    N = 1
    m = 3
    x = np.array([0.1 * (3 + i + N) for i in range(0, 10)])
    y = np.array([
        0.5913, 0.63 + (N / 17),
        0.7162, 0.8731,
        0.9574, 1.8 - cos(N / 11),
        1.3561, 1.2738,
        1.1 + (N / 29), 1.1672
    ])

    matrix_A = np.array([
        [1.0, 1.5, 2.5, 3.5],
        [1.5, 1.0, 2.0, 1.6],
        [2.5, 2.0, 1.0, 1.7],
        [3.5, 1.6, 1.7, 1.0]
    ])

    print('X = ', *map('{:.4f}'.format, x))
    print('Y = ', *map('{:.4f}'.format, y))
    print()

    Xo = x[0]
    h = x[1] - x[0]
    y_sum = np.sum(y)
    x_size = x.size

    c = np.zeros(m + 1)
    for i in range(1, m + 1):
        summ = 0
        for x_cr, y_cr in zip(x, y):
            p = solve_chebyshev_poly(i, x_size, (x_cr - Xo) / h)
            summ += y_cr * np.sum(p)
        c[i] = summ

    c[0] = y_sum / x_size

    # Проверим значения функции в точках Xi + h / 2
    Xs = np.array([Xi + h / 2 for Xi in x])
    q = np.zeros(x_size)
    h = x[1] - x[0]
    c[0] = y_sum / x_size
    for i, Xi in enumerate(Xs):
        p_s = np.zeros(m + 1)
        for k in range(m + 1):
            t = (Xi - Xo) / h
            Pi = solve_chebyshev_poly(k, x_size, t)
            p_s[k] = np.sum(Pi)
        q[i] = c[0] + c[1] * p_s[1] + c[2] * p_s[2] + c[3] * p_s[3]
        print(f'Y({Xi:.4f}) = {q[i]:.4f}')

    # Выпишем разложение через полиномы Чебышева
    coef = solve_coef(y, x_size, m)
    print(f'\nQ(t) = {coef[0]:.4f}', end='')
    for i in range(1, m + 1):
        print(f'{coef[i]:+.4f}*P̃{(i,x_size - 1)}', end='')
    print('\n')

    # 2) Найти методом Данилевского собственные значения
    # и собственные векторы симметрических матриц
    n_roots = np.linalg.matrix_rank(matrix_A)
    print(f"Число вещественных корней хар ур-я на (-inf, +inf): {n_roots}")

    values, S = method_Danielskogo(matrix_A)

    print("\nСобственные значения:")
    for Xi in values:
        print(f'{Xi: .4f}')

    # Для проверки 
    # check_eigv = sorted(np.linalg.eigvals(matrix_A))
    # for vct in check_eigv:
    #    print(f'{vct: .4f}')

    print("\nСобственные векторы:")
    nt, N = S.shape
    for i in range(N):
        print(f'\nX{i}:')
        for j in range(N):
            print(f'{S[j, i]: .4f}')

    print('\nA = S * L S^(-1):\n')
    L = np.diag(values)
    check_A = np.round(S @ L @ np.linalg.inv(S), 2)
    for r in check_A:
        print(' ', *map(' {:.4f}'.format, r))

    print('\nX = ', *map('{:.4f}'.format, x))

    # 3) Оценимваем Спектральный радиус с погрешносью eps = 0.001
    Yo = np.array([[1], [1], [1], [0]])
    sr = solve_radius_spect(matrix_A, Yo)
    print(f'\nСпектральный радиус: {sr[0]: .4f}')

