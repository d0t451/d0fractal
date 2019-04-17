# -*- coding: utf-8 -*-
"""
@author:    d0t451
@date:      2019/4/17
@desc:      draw fractal images
"""
from fractal_utils import *


def draw_julia_image(width, height, is_save_to_file=False):
    """
    绘制Julia分形图像
    :param width:
    :param height:
    :param is_save_to_file: 是否保存为文件
    :return:
    """
    julia_c_list = [0, -0.2 + 0.2j, -0.77 - 0.22j, 0.365 - 0.37j, -0.6358 + 0.682j, -0.55 + 0.64j, -0.52 + 0.62j, -0.51251 - 0.521296j,
                    -0.5264 - 0.5255j, -0.534 - 0.5255j, -0.54 - 0.5255j, -0.62 - 0.44j, -0.69 - 0.31j, -0.691 + 0.312j, -0.6984 + 0.31j, 0.26,
                    0.34 - 0.05j, 0.375 - 0.083j, 0.42413 + 0.20753j, 0.3593 + 0.5103j, 0.338 + 0.489j, 0 + 1j, -1.75488, -1.38, 0.3 - 0.015j,
                    0.47 - 0.1566j, 0.02 - -0.66j, -0.6843 - 0.3944j, -0.0471 - 0.656j, -0.015 - 0.66j, ]  # 几个Julia分形函数

    print('{} Julia fractal images will be shown...'.format(len(julia_c_list)))
    for c in julia_c_list:
        points = get_julia_set(c, width, height)
        draw_fx_image(points, width, height, bg_color=(0, 64, 64), image_show_time=50)
        if is_save_to_file:
            filename = 'tmp/{}.png'.format(c)
            render_and_save_fx_image(filename, points, width, height, bg_color=(0, 64, 64))


def draw_multi_color_julia_image(width, height, begin_color=(0, 0, 0), end_color=(255, 255, 255), is_save_to_file=False):
    """
    绘制渐变色Julia分形图像
    :param width:
    :param height:
    :param begin_color:
    :param end_color:
    :param is_save_to_file: 是否保存为文件
    :return:
    """
    julia_c_list = [0, -0.2 + 0.2j, -0.77 - 0.22j, 0.365 - 0.37j, -0.6358 + 0.682j, -0.55 + 0.64j, -0.52 + 0.62j, -0.51251 - 0.521296j,
                    -0.5264 - 0.5255j, -0.534 - 0.5255j, -0.54 - 0.5255j, -0.62 - 0.44j, -0.69 - 0.31j, -0.691 + 0.312j, -0.6984 + 0.31j, 0.26,
                    0.34 - 0.05j, 0.375 - 0.083j, 0.42413 + 0.20753j, 0.3593 + 0.5103j, 0.338 + 0.489j, 0 + 1j, -1.75488, -1.38, 0.3 - 0.015j,
                    0.47 - 0.1566j, 0.02 - -0.66j, -0.6843 - 0.3944j, -0.0471 - 0.656j, -0.015 - 0.66j, ]  # 几个Julia分形函数

    print('{} Julia fractal images will be shown...'.format(len(julia_c_list)))
    for c in julia_c_list:
        screen_points_with_escape_time = get_julia_set_with_escape_points(c, width, height)
        image_matrix = draw_multi_color_fx_image(screen_points_with_escape_time, width, height, begin_color, end_color, image_show_time=50)
        if is_save_to_file:
            save_fx_image('tmp/{}.png'.format(c), image_matrix)


def draw_mandelbrot_image(width, height, is_save_to_file=False):
    """
    绘制Mandelbrot分形图像
    :param width:
    :param height:
    :param is_save_to_file: 是否保存为文件
    :return:
    """
    points = get_mandelbrot_set(width, height)
    draw_fx_image(points, width, height, bg_color=(0, 64, 64), image_show_time=0)
    if is_save_to_file:
        render_and_save_fx_image('images/mandelbrot.png', points, width, height, bg_color=(0, 64, 64))


def draw_multi_color_mandelbrot_image(width, height, begin_color=(0, 0, 0), end_color=(255, 255, 255), is_save_to_file=False):
    """
    绘制渐变色Mandelbrot分形图像
    :param width:
    :param height:
    :param begin_color:
    :param end_color:
    :param is_save_to_file: 是否保存为文件
    :return:
    """
    screen_points_with_escape_time = get_mandelbrot_set_with_escape_points(width, height)
    image_matrix = draw_multi_color_fx_image(screen_points_with_escape_time, width, height, begin_color, end_color, image_show_time=0)
    if is_save_to_file:
        save_fx_image('images/mandelbrot.png', image_matrix)


def draw_koch_image(width, height, is_save_to_file=False):
    """
    绘制Koch曲线分形图像
    :param width:
    :param height:
    :param is_save_to_file: 是否保存为文件
    :return:
    """
    points = get_koch_set(width, height)
    draw_fx_image(points, width, height, color=(255, 255, 255), bg_color=(0, 64, 64), image_show_time=0)
    if is_save_to_file:
        render_and_save_fx_image('images/koch.png', points, width, height, color=(255, 255, 255), bg_color=(0, 64, 64))


def draw_sierpinski_carpet_image(width, height, is_save_to_file=False):
    """
    绘制谢尔宾斯基地毯分形图像
    :param width:
    :param height:
    :param is_save_to_file: 是否保存为文件
    :return:
    """
    points = get_sierpinski_carpet_set(width, height)
    draw_fx_image(points, width, height, color=(0, 0, 0), bg_color=(255, 255, 255), image_show_time=0)
    if is_save_to_file:
        render_and_save_fx_image('images/sierpinski_carpet.png', points, width, height, color=(0, 0, 0), bg_color=(255, 255, 255), )


if __name__ == '__main__':
    # 绘制Julia分形图像
    # draw_julia_image(800, 800)

    # 绘制渐变色Julia分形图像
    draw_multi_color_julia_image(800, 800, begin_color=(0, 64, 64), end_color=(0, 255, 0), is_save_to_file=False)

    # 绘制Mandelbrot分形图像
    # draw_mandelbrot_image(800, 800)

    # 绘制渐变色Mandelbrot分形图像
    # draw_multi_color_mandelbrot_image(800, 800, begin_color=(0, 0, 0), end_color=(255, 255, 255), is_save_to_file=True)

    # 绘制Koch曲线分形图像
    # draw_koch_image(600, 600, False)

    # 绘制谢尔宾斯基地毯分形图像
    # draw_sierpinski_carpet_image(729, 729, False)

    print('OK.')
