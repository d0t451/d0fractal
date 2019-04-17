# -*- coding: utf-8 -*-
"""
@author:    d0t451
@date:      2019/4/17
@desc:      fractal utils
"""
import math
import os
import time
from functools import wraps
import cv2 as cv
import numpy as np


# 用时装饰器
def decorator_used_time(f):
    @wraps(f)
    def wrapped_function(*args, **kwargs):
        start = time.perf_counter()
        ret = f(*args, **kwargs)
        print('Used time: {:.2f} s'.format(time.perf_counter() - start))
        return ret

    return wrapped_function


def screen_coordinate_to_fx_coordinate(screen_x, screen_y, screen_width, screen_height, fx_short_axis_length=4):
    """
    屏幕坐标转换为分形坐标
    :param screen_x:
    :param screen_y:
    :param screen_width:
    :param screen_height:
    :param fx_short_axis_length: 分形坐标系的短轴长度，长轴长度根据屏幕坐标系比例计算
    :return: 分形坐标系下的坐标
    """
    if screen_width <= 0 or screen_height <= 0 or fx_short_axis_length <= 0:
        print('Invalid screen_coordinate_to_fx_coordinate params.')
        return 0, 0
    if screen_x < 0 or screen_x >= screen_width or screen_y < 0 or screen_y >= screen_height:
        print('Invalid screen coordinate: {}', (screen_x, screen_y))
        return 0, 0

    if screen_width >= screen_height:
        fx_width, fx_height = fx_short_axis_length * screen_width / screen_height, fx_short_axis_length
    else:
        fx_width, fx_height = fx_short_axis_length, fx_short_axis_length * screen_height / screen_width

    fx_x = screen_x * fx_width / screen_width - fx_width / 2
    fx_y = -(screen_y * fx_height / screen_height - fx_height / 2)

    return fx_x, fx_y


@decorator_used_time
def get_mandelbrot_set(width, height):
    """
    获得Mandelbrot集
    :param width:
    :param height:
    :return: Mandelbrot集
    """
    screen_points = []
    for i in range(width):
        for j in range(height):
            fx_x, fx_y = screen_coordinate_to_fx_coordinate(i, j, width, height, fx_short_axis_length=5)

            is_mandelbrot_point = True
            tmp = 0
            for it in range(100):  # 迭代100次
                if math.isinf(tmp.real) or math.isinf(tmp.imag):
                    is_mandelbrot_point = False
                    break
                tmp = tmp * tmp + complex(fx_x, fx_y)

            if is_mandelbrot_point:
                screen_points.append((i, j))

    print('Got {} Mandelbrot points in area {} * {}.'.format(len(screen_points), width, height))
    return screen_points


@decorator_used_time
def get_mandelbrot_set_with_escape_points(width, height):
    """
    获得带有逃逸点的Mandelbrot集
    :param width:
    :param height:
    :return: Mandelbrot集
    """
    screen_points_with_escape_time = []  # 三维，第三维为逃逸时间
    for i in range(width):
        for j in range(height):
            fx_x, fx_y = screen_coordinate_to_fx_coordinate(i, j, width, height, fx_short_axis_length=4)

            tmp = 0
            escape_time = -1  # 逃逸时间
            for it in range(100):  # 迭代100次
                if abs(tmp) > 2:
                    escape_time = it
                    break
                tmp = tmp * tmp + complex(fx_x, fx_y)

            screen_points_with_escape_time.append((i, j, escape_time))

    print('Got {} Mandelbrot points and escaped points in area {} * {}.'.format(len(screen_points_with_escape_time), width, height))
    return screen_points_with_escape_time


@decorator_used_time
def get_julia_set(c, width, height):
    """
    获得填充Julia集
    :param c: Julia函数常数
    :param width:
    :param height:
    :return: 填充Julia集
    """
    screen_points = []
    for i in range(width):
        for j in range(height):
            fx_x, fx_y = screen_coordinate_to_fx_coordinate(i, j, width, height, fx_short_axis_length=4)

            is_julia_good_point = True
            tmp = complex(fx_x, fx_y)
            for it in range(100):  # 迭代100次
                if math.isinf(tmp.real) or math.isinf(tmp.imag):
                    is_julia_good_point = False
                    break
                tmp = tmp * tmp + c

            if is_julia_good_point:
                screen_points.append((i, j))

    print('Got {} Julia points in area {} * {}.'.format(len(screen_points), width, height))
    return screen_points


@decorator_used_time
def get_julia_set_with_escape_points(c, width, height):
    """
    获得带有逃逸点的Julia集
    :param c: Julia函数常数
    :param width:
    :param height:
    :return: Julia
    """
    screen_points_with_escape_time = []  # 三维，第三维为逃逸时间
    for i in range(width):
        for j in range(height):
            fx_x, fx_y = screen_coordinate_to_fx_coordinate(i, j, width, height, fx_short_axis_length=4)

            tmp = complex(fx_x, fx_y)
            escape_time = -1  # 逃逸时间
            for it in range(100):  # 迭代100次
                if abs(tmp) > 2:
                    escape_time = it
                    break
                tmp = tmp * tmp + c

            screen_points_with_escape_time.append((i, j, escape_time))

    print('Got {} Julia points and escaped points in area {} * {}.'.format(len(screen_points_with_escape_time), width, height))
    return screen_points_with_escape_time


def get_koch_points_from_2points(point1, point2, screen_width, screen_height):
    """
    从两个点递归计算出Koch曲线的所有中间点
    :param point1:
    :param point2:
    :param screen_width:
    :param screen_height:
    :return:
    """
    if not is_screen_point_valid(point1, screen_width, screen_height) or not is_screen_point_valid(point2, screen_width, screen_height):
        return []
    if math.sqrt(math.pow(point1[0] - point2[0], 2) + math.pow(point1[1] - point2[1], 2)) < 4:  # 距离太近
        return []

    p1 = (int(point1[0] + (point2[0] - point1[0]) / 3), int(point1[1] + (point2[1] - point1[1]) / 3))  # 三分之一处点
    p3 = (int(point1[0] + (point2[0] - point1[0]) * 2 / 3), int(point1[1] + (point2[1] - point1[1]) * 2 / 3))  # 三分之二处点
    p2 = get_rotated_point(p1[0], p1[1], p3[0], p3[1], 60)  # 中间凸出处点
    if not is_screen_point_valid(p1, screen_width, screen_height) \
            or not is_screen_point_valid(p2, screen_width, screen_height) \
            or not is_screen_point_valid(p3, screen_width, screen_height):
        return []

    points = [p1, p2, p3]
    points.extend(get_koch_points_from_2points(point1, p1, screen_width, screen_height))
    points.extend(get_koch_points_from_2points(p1, p2, screen_width, screen_height))
    points.extend(get_koch_points_from_2points(p2, p3, screen_width, screen_height))
    points.extend(get_koch_points_from_2points(p3, point2, screen_width, screen_height))
    return points


@decorator_used_time
def get_koch_set(width, height):
    """
    获得Koch曲线点集
    :param width:
    :param height:
    :return: Koch曲线点集
    """
    # 初始化为最大圆的内接三角形的三个顶点
    r = min(width, height) / 2 - 10
    if r <= 0:
        print('Width or height is too small: {} * {}.'.format(width, height))
        return []
    center = (width / 2, height / 2)  # 圆心
    init_point1 = (int(center[0]), int(center[1] - r))
    init_point2 = (int(center[0] - r * math.cos(30 * math.pi / 180)), int(center[1] + r * math.sin(30 * math.pi / 180)))
    init_point3 = (int(center[0] + r * math.cos(30 * math.pi / 180)), int(center[1] + r * math.sin(30 * math.pi / 180)))
    screen_points = [init_point1, init_point2, init_point3]  # 初始化为最大圆的内接三角形的三个顶点

    # 递归计算所有点
    screen_points.extend(get_koch_points_from_2points(init_point1, init_point2, width, height))
    screen_points.extend(get_koch_points_from_2points(init_point2, init_point3, width, height))
    screen_points.extend(get_koch_points_from_2points(init_point3, init_point1, width, height))

    print('Got {} Koch points in area {} * {}.'.format(len(screen_points), width, height))
    return screen_points


@decorator_used_time
def get_sierpinski_carpet_set(width, height):
    """
    获得谢尔宾斯基地毯点集
    :param width:
    :param height:
    :return: 谢尔宾斯基地毯点集
    """

    def get_sierpinski_carpet_points(left, top, right, bottom):
        """
        递归获取谢尔宾斯基地毯的点
        :param left:
        :param top:
        :param right:
        :param bottom:
        :return:
        """
        w, h = right - left + 1, bottom - top + 1
        if w != h or w < 3:
            return []

        sub_len = int(w / 3)  # 小正方形边长
        screen_points = []

        # 递归
        for row in range(3):
            for col in range(3):
                x1, y1 = left + sub_len * row, top + sub_len * col
                x2, y2 = x1 + sub_len - 1, y1 + sub_len - 1
                if row == 1 and col == 1:  # 中间块
                    for i in range(x1, x2 + 1):
                        for j in range(y1, y2 + 1):
                            screen_points.append((i, j))
                else:  # 周围8块
                    screen_points.extend(get_sierpinski_carpet_points(x1, y1, x2, y2))

        return screen_points

    # 递归获取谢尔宾斯基地毯的点
    square_len = int(math.pow(3, int(math.log(min(width, height), 3))))  # 大正方形边长，应为3的指数
    sierpinski_carpet_points = get_sierpinski_carpet_points(0, 0, square_len - 1, square_len - 1)
    print('Got {} Sierpinski carpet points in area {} * {}.'.format(len(sierpinski_carpet_points), width, height))
    return sierpinski_carpet_points


def draw_fx_image(screen_points, width, height, color=(255, 128, 0), bg_color=(0, 0, 0), image_show_time=0):
    """
    绘制分形图像
    :param screen_points: 分形点集，屏幕坐标系
    :param width:
    :param height:
    :param color: 前景色，RGB
    :param bg_color: 背景色，RGB
    :param image_show_time: 图像显示时间，毫秒,0表示无限
    :return:
    """
    if len(screen_points) == 0:
        print('No points to draw.')
        return

    # 填充图像矩阵
    cv_color = [color[2], color[1], color[0]]  # RGB -> BGR
    cv_bg_color = [bg_color[2], bg_color[1], bg_color[0]]  # RGB -> BGR
    image_matrix = [[cv_bg_color for col in range(width)] for row in range(height)]
    for i, j in screen_points:
        image_matrix[j][i] = cv_color  # 列索引在前
    np_mat = np.array(image_matrix, dtype=np.uint8)

    # 绘图
    cv.namedWindow('Fractal image')
    cv.moveWindow('Fractal image', 200, 50)
    cv.imshow('Fractal image', np_mat)
    cv.waitKey(image_show_time)
    # cv.destroyAllWindows()


# RGB颜色转换为HSL颜色
def rgb2hsl(rgb):
    rgb_normal = [[[rgb[0] / 255, rgb[1] / 255, rgb[2] / 255]]]
    hls = cv.cvtColor(np.array(rgb_normal, dtype=np.float32), cv.COLOR_RGB2HLS)
    return hls[0][0][0], hls[0][0][2], hls[0][0][1]  # hls to hsl


# HSL颜色转换为RGB颜色
def hsl2rgb(hsl):
    hls = [[[hsl[0], hsl[2], hsl[1]]]]  # hsl to hls
    rgb_normal = cv.cvtColor(np.array(hls, dtype=np.float32), cv.COLOR_HLS2RGB)
    return int(rgb_normal[0][0][0] * 255), int(rgb_normal[0][0][1] * 255), int(rgb_normal[0][0][2] * 255)


# RGB渐变色
def get_multi_colors_by_rgb(begin_color, end_color, color_count):
    if color_count < 2:
        return []

    colors = []
    steps = [(end_color[i] - begin_color[i]) / (color_count - 1) for i in range(3)]
    for color_index in range(color_count):
        colors.append([int(begin_color[i] + steps[i] * color_index) for i in range(3)])

    return colors


# HSL渐变色
def get_multi_colors_by_hsl(begin_color, end_color, color_count):
    if color_count < 2:
        return []

    colors = []
    hsl1 = rgb2hsl(begin_color)
    hsl2 = rgb2hsl(end_color)
    steps = [(hsl2[i] - hsl1[i]) / (color_count - 1) for i in range(3)]
    for color_index in range(color_count):
        hsl = [hsl1[i] + steps[i] * color_index for i in range(3)]
        colors.append(hsl2rgb(hsl))

    return colors


def draw_multi_color_fx_image(screen_points_with_escape_time, width, height, begin_color, end_color, image_show_time=0):
    """
    绘制渐变色分形图像
    :param screen_points_with_escape_time: 分形点集，屏幕坐标系，第三维为逃逸时间
    :param width:
    :param height:
    :param begin_color: 开始颜色，RGB
    :param end_color: 结束颜色，RGB
    :param image_show_time: 图像显示时间，毫秒,0表示无限
    :return:
    """
    if len(screen_points_with_escape_time) == 0:
        print('No points to draw.')
        return

    # 获取点集中的最大逃逸时间
    max_escape_time = -1
    for i, j, escape_time in screen_points_with_escape_time:
        if escape_time > max_escape_time:
            max_escape_time = escape_time
    color_count = max_escape_time + 2  # 渐变色数量，逃逸时间从-1开始
    colors = get_multi_colors_by_hsl(begin_color, end_color, color_count)
    escape_colors = {}  # 逃逸时间：颜色
    for i in range(color_count - 1):
        escape_colors[i] = [colors[i][2], colors[i][1], colors[i][0]]  # RGB -> BGR
    escape_colors[-1] = [end_color[2], end_color[1], end_color[0]]  # RGB -> BGR

    # 填充图像矩阵
    image_matrix_list = [[[] for col in range(width)] for row in range(height)]
    for i, j, escape_time in screen_points_with_escape_time:
        image_matrix_list[j][i] = escape_colors[escape_time]  # 列索引在前
    image_matrix = np.array(image_matrix_list, dtype=np.uint8)

    # 绘图
    cv.namedWindow('Fractal image')
    cv.moveWindow('Fractal image', 200, 50)
    cv.imshow('Fractal image', image_matrix)
    cv.waitKey(image_show_time)
    # cv.destroyAllWindows()

    return image_matrix


def render_and_save_fx_image(filename, screen_points, width, height, color=(255, 128, 0), bg_color=(0, 0, 0)):
    """
    保存分形图像
    :param filename: 保存文件名，包含路径
    :param screen_points: 分形点集，屏幕坐标系
    :param width:
    :param height:
    :param color: 前景色，RGB
    :param bg_color: 背景色，RGB
    :return:
    """
    # 填充图像矩阵
    cv_color = [color[2], color[1], color[0]]  # RGB -> BGR
    cv_bg_color = [bg_color[2], bg_color[1], bg_color[0]]  # RGB -> BGR
    image_matrix = [[cv_bg_color for col in range(width)] for row in range(height)]
    for i, j in screen_points:
        image_matrix[j][i] = cv_color  # 列索引在前
    np_mat = np.array(image_matrix, dtype=np.uint8)

    # save to file
    save_path = os.path.dirname(filename)
    if save_path != '' and not os.path.exists(save_path):
        os.makedirs(save_path)
    cv.imwrite(filename, np_mat)


def save_fx_image(filename, image_matrix):
    """
    保存分形图像
    :param filename:
    :param image_matrix:
    :return:
    """
    save_path = os.path.dirname(filename)
    if save_path != '' and not os.path.exists(save_path):
        os.makedirs(save_path)
    cv.imwrite(filename, image_matrix)


def get_rotated_point(x1, y1, x2, y2, angle):
    """
    获得点（x2, y2）绕（x1, y1）顺时针旋转angle度后的点（x3, y3），屏幕坐标系
    :param x1:
    :param y1:
    :param x2:
    :param y2:
    :param angle: 度
    :return:
    """
    x3 = int(x1 + (x2 - x1) * math.cos(angle * math.pi / 180) - (y2 - y1) * math.sin(angle * math.pi / 180))
    y3 = int(y1 + (x2 - x1) * math.sin(angle * math.pi / 180) + (y2 - y1) * math.cos(angle * math.pi / 180))
    return x3, y3


# 判断屏幕坐标系下的点坐标是否有效
def is_screen_point_valid(point, screen_width, screen_height):
    if 0 <= point[0] < screen_width and 0 <= point[1] < screen_height:
        return True
    return False
