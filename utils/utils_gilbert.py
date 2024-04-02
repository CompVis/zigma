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
    print(order_index)


if __name__ == "__main__":

    # import argparse

    # parser = argparse.ArgumentParser()
    # parser.add_argument("width", type=int, default=16)
    # parser.add_argument("height", type=int, default=16)
    # args = parser.parse_args()

    # width = args.width
    # height = args.height
    pass
