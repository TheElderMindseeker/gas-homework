import numpy as np
import scipy as sp
import matplotlib.pyplot as plt


def pass_through(matrix, delta):
    N = matrix.shape[0]
    P = np.zeros(shape=(N,))
    Q = np.zeros(shape=(N,))
    P[0] = matrix[0, 1] / -matrix[0, 0]
    Q[0] = delta[0] / matrix[0, 0]
    for i in range(1, N - 1):
        P[i] = matrix[i, i + 1] / (-matrix[i, i] - matrix[i, i - 1] * P[i - 1])
        Q[i] = (matrix[i, i - 1] * Q[i - 1] - delta[i]) / (-matrix[i, i] - matrix[i, i - 1] * P[i - 1])

    x = np.zeros(shape=(N,))
    x[N - 1] = (matrix[N - 1, N - 2] * Q[N - 2] - delta[N - 1]) / (-matrix[N - 1, N - 1] - matrix[N - 1, N - 2] * P[N - 2])

    for j in range(N - 1):
        i = N - (j + 2)
        x[i] = P[i] * x[i + 1] + Q[i]
    
    return x


MAP_SIZE = 301
H = 1
H_SQR = H ** 2
TAU = H
Map = np.zeros(shape=(MAP_SIZE, MAP_SIZE))
Map[:, 0] = 1.


TOP = 0
RIGHT = 1
BOTTOM = 2
LEFT = 3
building_mask = np.ones(shape=(MAP_SIZE, MAP_SIZE, 4), dtype=np.int8)


def add_building(building_mask, x0, y0, x1, y1):
    # Top border
    for i in range(x0 + 1, x1):
        building_mask[y0, i, BOTTOM] = 0
    
    # Right border
    for i in range(y0 + 1, y1):
        building_mask[i, x1, LEFT] = 0
    
    # Bottom border
    for i in range(x0 + 1, x1):
        building_mask[y1, i, TOP] = 0
    
    # Left border
    for i in range(y0 + 1, y1):
        building_mask[i, y1, RIGHT] = 0
    
    # Inner
    for i in range(x0 + 1, x1):
        for j in range(y0 + 1, y1):
            building_mask[j, i, :] = 0


add_building(building_mask, 69, 18, 87, 36)


def apply_mask_horizontal(matrix, mask, k):
    new_matrix = np.copy(matrix)
    for i in range(mask.shape[1]):
        if mask[k, i, LEFT] == 0:
            new_matrix[i, i - 1] = 0.
        if mask[k, i, RIGHT] == 0:
            new_matrix[i, i + 1] = 0.
    
    return new_matrix


def apply_mask_vertical(matrix, mask, k):
    new_matrix = np.copy(matrix)
    for i in range(mask.shape[0]):
        if mask[i, k, BOTTOM] == 0:
            new_matrix[i, i + 1] = 0.
        if mask[i, k, TOP] == 0:
            new_matrix[i, i - 1] = 0.
    
    return new_matrix

# plt.matshow(Map, cmap='hot')
# plt.show()
# exit(0)

k1 = - 1 / (2 * H_SQR)
k2 = 1 / TAU + 1 / H_SQR
k3 = - 1 / (2 * H_SQR)

k_matrix = np.zeros(shape=(MAP_SIZE, MAP_SIZE))
k_matrix[0, :2] = (k2, k3)
k_matrix[MAP_SIZE - 1, MAP_SIZE - 2:MAP_SIZE] = (k1, k2)
for i in range(1, MAP_SIZE - 1):
    k_matrix[i, i - 1:i + 2] = (k1, k2, k3)

print(k_matrix)

c1 = 1 / (2 * H_SQR)
c2 = 1 / TAU - 1 / H_SQR
c3 = 1 / (2 * H_SQR)

# c_matrix = np.zeros(shape=(MAP_SIZE, MAP_SIZE))
# c_matrix[0, :2] = (c2, c3)
# c_matrix[MAP_SIZE - 1, MAP_SIZE -2:MAP_SIZE] = (c1, c2)
# for i in range(1, MAP_SIZE - 1):
#     c_matrix[i, i - 1:i + 2] = (c1, c2, c3)

for it in range(1):
    if it % 100 == 0:
        print('Iter:', it)
    # print(pass_through(k_matrix, delta[0, :]))
    u_temp = np.copy(Map)
    for i in range(MAP_SIZE):
        vector_b = np.zeros(shape=(MAP_SIZE,))
        for m in range(MAP_SIZE):
            if i == MAP_SIZE - 1:
                vector_b[m] = c1 * Map[i - 1, m] + c2 * Map[i, m]
            elif i == 0:
                vector_b[m] = c2 * Map[i, m] + c3 * Map[i + 1, m]
            else:
                vector_b[m] = c1 * Map[i - 1, m] + c2 * Map[i, m] + c3 * Map[i + 1, m]
        masked_matrix = apply_mask_horizontal(k_matrix, building_mask, i)
        # print(masked_matrix)
        u_temp[i, 1:] = pass_through(masked_matrix, vector_b)[1:]

    # plt.matshow(u_temp, cmap='hot')
    # plt.show()

    Map = np.copy(u_temp)
    for i in range(1, MAP_SIZE):
        vector_b = np.zeros(shape=(MAP_SIZE,))
        for k in range(MAP_SIZE):
            if i == MAP_SIZE - 1:
                vector_b[k] = c1 * u_temp[k, i - 1] + c2 * u_temp[k, i]
            elif i == 0:
                vector_b[k] = c2 * u_temp[k, i] + c3 * u_temp[k, i + 1]
            else:
                vector_b[k] = c1 * u_temp[k, i - 1] + c2 * u_temp[k, i] + c3 * u_temp[k, i + 1]
        masked_matrix = apply_mask_vertical(k_matrix, building_mask, i)
        # print(masked_matrix)
        Map[:, i] = pass_through(masked_matrix, vector_b)

for i in range(MAP_SIZE):
    for j in range(MAP_SIZE):
        if building_mask[i, j, :].sum() == 0:
            Map[i, j] = 0

plt.matshow(Map, cmap='hot')
plt.show()

# delta = np.matmul(c_matrix, u_temp.T)

# u2_temp = np.copy(u_temp)
# for i in range(1, MAP_SIZE):
    # u2_temp[:, i] = pass_through(k_matrix, delta[:, i])

# plt.matshow(u2_temp, cmap='hot', origin='upper')
# plt.show()