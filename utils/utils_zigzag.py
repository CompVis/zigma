import math
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib


#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2024 abetusk
# ported from https://github.com/jakubcerveny/gilbert/tree/master

import numpy as np


def gilbert_xy2d(x, y, w, h):
    """
    Generalized Hilbert ('gilbert') space-filling curve for arbitrary-sized
    2D rectangular grids. Takes a discrete 2D coordinate and maps it to the
    index position on the gilbert curve.
    """

    if w >= h:
        return gilbert_xy2d_r(0, x, y, 0, 0, w, 0, 0, h)
    return gilbert_xy2d_r(0, x, y, 0, 0, 0, h, w, 0)


def sgn(x):
    return -1 if x < 0 else (1 if x > 0 else 0)


def in_bounds(x, y, x_s, y_s, ax, ay, bx, by):

    dx = ax + bx
    dy = ay + by

    if dx < 0:
        if (x > x_s) or (x <= (x_s + dx)):
            return False
    else:
        if (x < x_s) or (x >= (x_s + dx)):
            return False

    if dy < 0:
        if (y > y_s) or (y <= (y_s + dy)):
            return False
    else:
        if (y < y_s) or (y >= (y_s + dy)):
            return False

    return True


def gilbert_xy2d_r(cur_idx, x_dst, y_dst, x, y, ax, ay, bx, by):

    w = abs(ax + ay)
    h = abs(bx + by)

    (dax, day) = (sgn(ax), sgn(ay))  # unit major direction
    (dbx, dby) = (sgn(bx), sgn(by))  # unit orthogonal direction

    dx = dax + dbx
    dy = day + dby

    if h == 1:
        if dax == 0:
            return cur_idx + (dy * (y_dst - y))
        return cur_idx + (dx * (x_dst - x))

    if w == 1:
        if dbx == 0:
            return cur_idx + (dy * (y_dst - y))
        return cur_idx + (dx * (x_dst - x))

    (ax2, ay2) = (ax // 2, ay // 2)
    (bx2, by2) = (bx // 2, by // 2)

    w2 = abs(ax2 + ay2)
    h2 = abs(bx2 + by2)

    if 2 * w > 3 * h:
        if (w2 % 2) and (w > 2):
            # prefer even steps
            (ax2, ay2) = (ax2 + dax, ay2 + day)

        if in_bounds(x_dst, y_dst, x, y, ax2, ay2, bx, by):
            return gilbert_xy2d_r(cur_idx, x_dst, y_dst, x, y, ax2, ay2, bx, by)

        cur_idx += abs((ax2 + ay2) * (bx + by))
        return gilbert_xy2d_r(
            cur_idx, x_dst, y_dst, x + ax2, y + ay2, ax - ax2, ay - ay2, bx, by
        )

    else:
        if (h2 % 2) and (h > 2):
            # prefer even steps
            (bx2, by2) = (bx2 + dbx, by2 + dby)

        # standard case: one step up, one long horizontal, one step down
        if in_bounds(x_dst, y_dst, x, y, bx2, by2, ax2, ay2):
            return gilbert_xy2d_r(cur_idx, x_dst, y_dst, x, y, bx2, by2, ax2, ay2)
        cur_idx += abs((bx2 + by2) * (ax2 + ay2))

        if in_bounds(x_dst, y_dst, x + bx2, y + by2, ax, ay, bx - bx2, by - by2):
            return gilbert_xy2d_r(
                cur_idx, x_dst, y_dst, x + bx2, y + by2, ax, ay, bx - bx2, by - by2
            )
        cur_idx += abs((ax + ay) * ((bx - bx2) + (by - by2)))

        return gilbert_xy2d_r(
            cur_idx,
            x_dst,
            y_dst,
            x + (ax - dax) + (bx2 - dbx),
            y + (ay - day) + (by2 - dby),
            -bx2,
            -by2,
            -(ax - ax2),
            -(ay - ay2),
        )


def gilbert_zigzag_path(N):
    width = height = N
    order_index = np.zeros((width, height), dtype=int)
    for x in range(width):
        for y in range(height):
            idx = gilbert_xy2d(x, y, width, height)
            order_index[x, y] = idx
    return order_index


##################


def reverse_permut_np(permutation):
    n = len(permutation)
    reverse = np.array([0] * n)
    for i in range(n):
        reverse[permutation[i]] = i
    return reverse


def zigzag_path(N):
    print("zigzag_sub_v1", N)
    assert N % 2 == 0, "N should be even"

    def zigzag_path_lr(N, start_row=0, start_col=0, dir_row=1, dir_col=1):
        path = []
        for i in range(N):
            for j in range(N):
                # If the row number is even, move right; otherwise, move left
                col = j if i % 2 == 0 else N - 1 - j
                path.append((start_row + dir_row * i) * N + start_col + dir_col * col)
        return path

    def zigzag_path_tb(N, start_row=0, start_col=0, dir_row=1, dir_col=1):
        path = []
        for j in range(N):
            for i in range(N):
                # If the column number is even, move down; otherwise, move up
                row = i if j % 2 == 0 else N - 1 - i
                path.append((start_row + dir_row * row) * N + start_col + dir_col * j)
        return path

    paths = []
    for start_row, start_col, dir_row, dir_col in [
        (0, 0, 1, 1),
        (0, N - 1, 1, -1),
        (N - 1, 0, -1, 1),
        (N - 1, N - 1, -1, -1),
    ]:
        paths.append(zigzag_path_lr(N, start_row, start_col, dir_row, dir_col))
        paths.append(zigzag_path_tb(N, start_row, start_col, dir_row, dir_col))

    for _index, _p in enumerate(paths):
        paths[_index] = np.array(_p)
    return paths


def rand_perm(N, num):
    res = []
    for _ in range(num):
        # Generate a list of numbers from 0 to NxN-1
        numbers = list(range(N * N))
        # Randomly shuffle the numbers
        random.shuffle(numbers)
        # print(numbers)
        res.append(numbers)
    return res


def traverse_grid_v1_continuous(N):
    path = []
    direction = "right"  # initial direction
    for i in range(N):
        if direction == "right":
            for j in range(N):
                path.append(i * N + j)
            if i != N - 1:  # if not the last row
                direction = "left"
        elif direction == "left":
            for j in range(N - 1, -1, -1):
                path.append(i * N + j)
            if i != N - 1:  # if not the last row
                direction = "right"
    return [path, list(reversed(path))]


def test_hibert_fig(num_dims=2, img_size_power=3):
    import numpy as np
    import matplotlib.pyplot as plt
    from hilbert import decode

    def draw_curve(ax, num_bits):

        # The maximum Hilbert integer.
        max_h = 2 ** (num_bits * num_dims)

        # Generate a sequence of Hilbert integers.
        hilberts = np.arange(max_h)
        print("image size:", 2**img_size_power)
        order_index = np.zeros((2**img_size_power, 2**img_size_power), dtype=int)

        # Compute the 2-dimensional locations.
        locs = decode(hilberts, num_dims, num_bits)
        for i, loc in enumerate(locs):
            order_index[loc[0], loc[1]] = i
        print(locs.shape, locs)
        print(order_index)
        # Draw
        ax.plot(locs[:, 0], locs[:, 1], ".-")
        ax.set_aspect("equal")
        # ax.set_title("%d bits per dimension" % (num_bits))
        # ax.set_xlabel("dim 1")
        # ax.set_ylabel("dim 2")

    fig = plt.figure(figsize=(16, 4))
    for ii, num_bits in enumerate([img_size_power]):
        ax = fig.add_subplot(1, 4, ii + 1)
        draw_curve(ax, num_bits)
    plt.savefig("example_2d.png", bbox_inches="tight")


def hilbert_path_square(num_dims=2, N=4):
    import numpy as np
    from hilbert import decode

    img_size_power = int(math.sqrt(N))
    assert img_size_power**2 == N, f"{N} should be a square number"
    print("img_size_power", img_size_power)

    def draw_curve(num_bits):
        # The maximum Hilbert integer.
        max_h = 2 ** (num_bits * num_dims)

        # Generate a sequence of Hilbert integers.
        hilberts = np.arange(max_h)
        print("image size:", N)
        order_index = np.zeros((N, N), dtype=int)

        # Compute the 2-dimensional locations.
        locs = decode(hilberts, num_dims, num_bits)
        for i, loc in enumerate(locs):
            order_index[loc[0], loc[1]] = i
        # print(locs.shape, locs)
        print(order_index)
        return order_index

    res = draw_curve(img_size_power)
    res_mirror = np.transpose(res)
    ro90 = np.rot90(res, 1)
    ro90_mirror = np.transpose(ro90)
    ro180 = np.rot90(res, 2)
    ro180_mirror = np.transpose(ro180)
    ro270 = np.rot90(res, 3)
    ro270_mirror = np.transpose(ro270)
    res = [res, res_mirror, ro90, ro90_mirror, ro180, ro180_mirror, ro270, ro270_mirror]
    print("***")
    for _ in res:
        print(_)
    res = [_.flatten() for _ in res]
    for _ in res:
        print(_.shape)
    return res


def hilbert_path(N=16):

    res = gilbert_zigzag_path(N)
    res_mirror = np.transpose(res)
    ro90 = np.rot90(res, 1)
    ro90_mirror = np.transpose(ro90)
    ro180 = np.rot90(res, 2)
    ro180_mirror = np.transpose(ro180)
    ro270 = np.rot90(res, 3)
    ro270_mirror = np.transpose(ro270)
    res = [res, res_mirror, ro90, ro90_mirror, ro180, ro180_mirror, ro270, ro270_mirror]
    print("***")
    for _ in res:
        print(_)
    res = [_.flatten() for _ in res]
    for _ in res:
        print(_.shape)
    return res


def draw_pineao_curve():
    import matplotlib.pyplot as plt

    def peano_curve(level, x, y, dx, dy):
        if level == 0:
            plt.plot([x, x + dx], [y, y + dy], color="black")
        else:
            dx /= 3
            dy /= 3
            peano_curve(level - 1, x, y, dx, dy)
            peano_curve(level - 1, x, y + 2 * dy, dx, dy)
            peano_curve(level - 1, x + dx, y + dy, dx, dy)
            peano_curve(level - 1, x + 2 * dx, y, dx, dy)
            peano_curve(level - 1, x + 2 * dx, y + 2 * dy, dx, dy)

    plt.figure(figsize=(6, 6))
    peano_curve(
        1, 0, 0, 1, 1
    )  # Increase the first argument to increase the complexity of the curve
    plt.gca().invert_yaxis()  # Invert y axis to match the standard mathematical coordinate system
    plt.axis("off")  # Hide axes
    plt.show()
    plt.savefig("peano_curve.png", bbox_inches="tight")


if __name__ == "__main__":
    N = 4
    if False:
        print(zigzag_path(N))
        print(len(zigzag_path(N)))
    elif True:
        # hilbert_path(N=N)
        # test_hibert_fig(num_dims=2, img_size_power=3)
        draw_pineao_curve()
