# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 10:51:14 2021

@author: sasmitha
"""


# =============================================================================
# library to read image file
# libraries were used only for reading and writing files
# numpy - used to generate random color array, save image
# pathlib,os - used to get file paths
# cv2 - used to read, write, show images
# =============================================================================
import os
from pathlib import Path
import cv2
import math
import numpy as np


# =============================================================================
# read file
# =============================================================================
def read_file():
    file_name = "dashcam_view_2.png"  //file name
    path = os.path.join(Path().absolute(), file_name)
    image = cv2.imread(path, 0)
    # show_image(image)
    image = image.tolist()  # convert to python list
    write_image(image, "1_read.png")
    show_image(image, "Read image file - dashcam_view_2.png")
    print("read file dashcam_view_1.jpg successfully! \n")
    return image


# =============================================================================
# save image
# =============================================================================
def write_image(image, name):
    cv2.imwrite(name, np.asarray(image))


# =============================================================================
# show images of greay scale
# =============================================================================
def show_image(image, title):
    im_size = cv2.resize(np.asarray(image).astype(np.uint8), (450, 801))
    cv2.imshow(title, im_size)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def show_color_image(image, title):
    im_size = cv2.resize(np.asarray(image) / 255, (300, 534))
    cv2.imshow(title, im_size)  # show the image from python list
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# =============================================================================
# scaling function
# =============================================================================
def scale(image, scale_factor=0.8):
    n_rows = len(image)
    n_cols = len(image[0])

    scale_col_pixels = math.ceil(n_cols * scale_factor)
    col_dist = (n_cols - 1) / (scale_col_pixels - 1)

    scaled_rows_image = []
    # scaling along the rows - linear interpolation
    for row in range(n_rows):
        new_row = [image[row][0]]
        index = col_dist
        while index < n_cols:
            x_floor = math.floor(index)
            x_ceil = math.ceil(index)
            w_floor = index - x_floor
            w_ceil = x_ceil - index

            try:
                # find new interpolated value based on ceil and floor values
                new_value = (image[row][x_ceil] * w_ceil) + (
                    image[row][x_floor] * w_floor
                )
            except IndexError:
                pass
            new_row.append(new_value)
            index += col_dist
        scaled_rows_image.append(new_row)

    new_cols = len(scaled_rows_image[0])
    scale_row_pixels = math.ceil(n_rows * scale_factor)
    row_dist = (n_rows - 1) / (scale_row_pixels - 1)

    # final output to be replaced with coloumn interpolation values
    scaled_final_image = [[0] * scale_col_pixels for i in range(scale_row_pixels)]

    # scaling along the columns - linear interpolation
    for col in range(new_cols):
        scaled_final_image[0][col] = scaled_rows_image[0][col]
        index = row_dist
        row_index = 1
        while index < n_rows:
            y_floor = math.floor(index)
            y_ceil = math.ceil(index)
            w_floor = index - y_floor
            w_ceil = y_ceil - index

            try:
                # find new interpolated value based on ceil and floor values
                new_value = (scaled_rows_image[y_ceil][col] * w_ceil) + (
                    scaled_rows_image[y_floor][col] * w_floor
                )

                scaled_final_image[row_index][col] = new_value
            except IndexError:
                pass

            index += row_dist
            row_index += 1
    show_image(scaled_final_image, "Scaled image")
    # write_image(scaled_final_image, "2_scaled.jpg")
    print("image scaled successfully! \n")
    return scaled_final_image


# =============================================================================
# normalize the image
# logic -
# find : (pixel value - mean of all pixels)/ (max of all pixels - min of all pixels)
# shift each pixel by min of newly computed image
# multiple each pixel by 255 and devided by max pixel in new image
# =============================================================================
def normalize(image):
    n_rows = len(image)
    n_cols = len(image[0])

    # take sum, maximum, minimum, mean
    total = sum([sum(x) for x in image])
    maxi = max([max(x) for x in image])
    mini = min([min(x) for x in image])
    mean = total / (n_rows * n_cols)

    for i in range(n_rows):
        for j in range(n_cols):
            # (pixel value - mean of all pixels)/ (max of all pixels - min of all pixels)
            image[i][j] = (image[i][j] - mean) / (maxi - mini)

    new_mini = min([min(x) for x in image])

    for i in range(n_rows):
        for j in range(n_cols):
            # shift each pixel by min of newly computed image
            image[i][j] = image[i][j] - new_mini

    new_maxi = max([max(x) for x in image])
    for i in range(n_rows):
        for j in range(n_cols):
            image[i][j] = image[i][j] / new_maxi * 255

    show_image(image, "Normalized image")
    # write_image(image, "3_normalized.jpg")
    print("image normalized successfully! \n")
    return image


# =============================================================================
# replicate edges of image (to avoid dimentionality reduction)
# =============================================================================
def replicate_edges(image):
    image.insert(0, image[0] + [])
    image.append(image[-1] + [])
    for k in range(len(image)):
        image[k].insert(0, image[k][0])
        image[k].append(image[k][-1])
    print("replicated edges of the image successfully! \n")
    return image


# =============================================================================
# return gaussian kernel
# =============================================================================
def get_gaussian_filter(kernel_size=3, sigma=1.4):
    g_filter = [[0.0] * kernel_size for i in range(kernel_size)]

    height = kernel_size // 2
    width = kernel_size // 2

    for x in range(-height, height + 1):
        for y in range(-width, width + 1):
            x1 = 2 * math.pi * (sigma ** 2)
            x2 = math.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
            g_filter[x + height][y + width] = (1 / x1) * x2
    return g_filter


# =============================================================================
# linear noise filter
# filter type: Gaussian filter
# window size: 5 by 5
# reasons:
#   -since the image is having less noise, gaussian noise distribution is there.
#   -so used gaussian filter to avoid noise.
#   -also Gaussian filter can remove blur effect on boundaries
#   -when used kernel size = 3 by 3, imaged became dark. so to avoid that, used 5 by 5
# =============================================================================
def gaussian_filter_process(image):
    image = replicate_edges(image)  # wrap the image to avoid dimansionality reduction
    image = replicate_edges(image)  # here wraps twice since kernel size is 5
    kernel = get_gaussian_filter(kernel_size=5)
    height = len(image)
    width = len(image[0])
    kernel_size = len(kernel)

    result = []
    middle = kernel_size // 2
    for y in range(middle, (height - middle)):
        result_row = []
        for x in range(middle, (width - middle)):
            value = 0
            for a in range(kernel_size):
                for b in range(kernel_size):
                    tem_value = kernel[a][b] * image[y - middle + a][x - middle + b]
                    value += tem_value
            result_row.append(value)
        result.append(result_row)

    show_image(result, "Gaussian filtered image")
    # write_image(result, "4_gaussian_filterd.jpg")
    print("image gaussian filtered successfully! \n")
    return result


# =============================================================================
# Non linear noise filter
# filter type: Median filter
# window size: 3 by 3
# reasons:
#   -used to avoid salt and pepper noise in road marks (white and black)
#   -used to avoid blur effect(because this gives existing pixel values insterad of
#                               computing new values)
# =============================================================================
def median_filter(image, kernel_size=3):
    image = replicate_edges(image)  # wrap the image to avoid dimansionality reduction
    height = len(image)
    width = len(image[0])
    kernel_elements = kernel_size ** 2

    result = []

    middle = kernel_size // 2
    for y in range(middle, (height - middle)):
        result_row = []
        for x in range(middle, (width - middle)):
            values = []
            for a in range(kernel_size):
                for b in range(kernel_size):
                    tem_value = image[y - middle + a][x - middle + b]
                    values.append(tem_value)
            sorted_values = sorted(values)
            result_row.append(sorted_values[kernel_elements // 2])
        result.append(result_row)

    show_image(result, "Median filtered image")
    # write_image(result, "5_median_filterd.jpg")
    print("image median filtered successfully! \n")
    return result


# =============================================================================
# sobel gradient finding algorithm (for canny edge)
# return : vertical, horizontal gradients of each pixel
# =============================================================================
def sobel(image):
    # define kernels
    kernel_x = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
    kernel_y = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]

    height = len(image)
    width = len(image[0])
    kernel_size = len(kernel_x)

    gx, gy = [], []
    middle = kernel_size // 2
    for y in range(middle, (height - middle)):
        gx_row = []
        gy_row = []
        for x in range(middle, (width - middle)):
            value_x = 0
            value_y = 0
            # find gradients of sobel kernel
            for a in range(kernel_size):
                for b in range(kernel_size):
                    tem_value_x = kernel_x[a][b] * image[y - middle + a][x - middle + b]
                    tem_value_y = kernel_y[a][b] * image[y - middle + a][x - middle + b]
                    value_x += tem_value_x
                    value_y += tem_value_y

            gx_row.append(value_x)
            gy_row.append(value_y)

        gx.append(gx_row)
        gy.append(gy_row)

    print("sucessfully computed gradients \n")
    return gx, gy


# =============================================================================
# calculate max gradient to get the thresholds in canny edge
# =============================================================================
def get_max_grad(grad_image):
    maximum = 0
    for row in grad_image:
        if max(row) >= maximum:
            maximum = max(row)
    return maximum


# =============================================================================
# calculate gradients and angles in canny edge
# =============================================================================
def gradient_cal(gx, gy, epsilon=0.0000000000001):
    new_grad = []
    new_theta = []
    for row in range(len(gx)):
        new_grad_row = []
        new_theta_row = []
        for col in range(len(gx[row])):
            grad = math.sqrt((gx[row][col] ** 2) + (gy[row][col] ** 2))
            theta = math.degrees(
                math.atan((gy[row][col] * 1.0) / (gx[row][col] + epsilon))
            )
            new_grad_row.append(grad)
            new_theta_row.append(theta)
        new_grad.append(new_grad_row)
        new_theta.append(new_theta_row)
    print("successfully computed gradient magnitudes and angles(in degrees) \n")
    return new_grad, new_theta


# =============================================================================
# canny edge detection
# parameters:
#   -gaussian kernel dim: 5 by 5
#   -sobel gradient finding algorithm
# =============================================================================
def canny_edge_detection(image):
    image_height = len(image)
    image_width = len(image[0])

    gx, gy = sobel(image)
    new_grad, new_theta = gradient_cal(gx, gy)
    new_grad_max = get_max_grad(new_grad)

    low_th = new_grad_max * 0.1
    high_th = new_grad_max * 0.5

    final_output = [[0.0] * image_width for i in range(image_height)]

    # Looping through every pixel of the grayscale
    # image
    for i in range(1, image_height - 1):
        for j in range(1, image_width - 1):
            try:

                l = 255
                r = 255

                #                     get angle
                grad_ang = new_theta[i][j]
                grad_ang = abs(grad_ang - 180) if abs(grad_ang) > 180 else abs(grad_ang)

                if (0 <= grad_ang < 22.5) or (157.5 <= grad_ang <= 180):
                    l = new_grad[i][j + 1]
                    r = new_grad[i][j - 1]

                elif 22.5 <= grad_ang < 67.5:
                    l = new_grad[i + 1][j - 1]
                    r = new_grad[i - 1][j + 1]

                elif 67.5 <= grad_ang < 112.5:
                    l = new_grad[i + 1][j]
                    r = new_grad[i - 1][j]

                elif 112.5 <= grad_ang < 157.5:
                    l = new_grad[i - 1][j - 1]
                    r = new_grad[i + 1][j + 1]

                #                     non max suppressing step
                if (new_grad[i][j] >= l) and (new_grad[i][j] >= r):
                    final_output[i][j] = new_grad[i][j]
                else:
                    final_output[i][j] = 0

            except IndexError as e:
                pass

    ids = [[0.0] * image_width for i in range(image_height)]

    # double thresholding step
    for i in range(image_width):
        for j in range(image_height):
            try:
                grad_mag = final_output[j][i]

                if grad_mag < low_th:
                    final_output[j][i] = 0
                elif high_th > grad_mag >= low_th:
                    ids[j][i] = 1
                else:
                    ids[j][i] = 2
            except IndexError as e:
                pass

    show_image(final_output, "Canny edge detected image")
    # write_image(final_output, "6_canny_edge.jpg")
    print("canny edge detection successful! \n")
    return final_output


# =============================================================================
# hough tranformation
# parametric equation
#   - rho = x*cos(theta) + y*sin(theta)
#           rho - distance from origin to new function, x*cos(theta) + y*sin(theta)
#           theta - angle between x-axis and perpendiculat drawn from origin to function, x*cos(theta) + y*sin(theta)
# =============================================================================
def hough_tranform(image):
    image_height = len(image)
    image_width = len(image[0])
    diagonal = round(math.sqrt(image_height ** 2 + image_width ** 2))  # range of rho

    p_thetha_comb = [[0] * 150 for _ in range(diagonal * 2)]

    for row in range(image_height):
        for col in range(image_width):
            if image[row][col] != 0:
                # range of theta selected in this range to capture edges of road marks
                for theta in range(120, 150):
                    p = round(
                        col * math.cos(math.radians(theta))
                        + row * math.sin(math.radians(theta))
                    )
                    p_thetha_comb[p][theta] += 1

                # for theta in range(30, 60):
                #     p = round(col * math.cos(math.radians(theta)) + row * math.sin(math.radians(theta)))
                #     p_thetha_comb[p][theta]+=1

    # getting p, theta values via the index value of elements
    p_thetha_comb = [enumerate(x) for x in p_thetha_comb]
    new_p_thetha_comb = []

    y = 0
    for array in p_thetha_comb:
        for i in array:
            j = list(i)
            j.insert(0, y)
            new_p_thetha_comb.append(j)

        y += 1

    maximum_p_values = []
    threshold = 90

    # get maximum hit counts of p, theta pairs using the threshold
    maximum_p_values = list(filter(lambda x: x[2] >= threshold, new_p_thetha_comb))

    hough_tranformed_image = [[0] * image_width for _ in range(image_height)]

    #   find the pixels which are hitting in the maximum hit counts
    for maximum_p in maximum_p_values:
        for row in range(image_height):
            for col in range(image_width):
                if image[row][col] != 0 and hough_tranformed_image[row][col] == 0:
                    p = round(
                        col * math.cos(math.radians(maximum_p[1]))
                        + row * math.sin(math.radians(maximum_p[1]))
                    )

                    if p == maximum_p[0]:
                        hough_tranformed_image[row][col] = 255

    show_image(hough_tranformed_image, "Hough transformed image")
    # write_image(hough_tranformed_image, "7_hough_tranformed.jpg")
    print("successfully hough transformed \n")
    return hough_tranformed_image


# =============================================================================
# impose red colors in identified road marks using hough tranformation
# the output image is having red coloured outlines around the lane marks
# since the identified edges are very tiny(thin), need to zoom the image to see the red outlines
# =============================================================================
def impose_red_lane_mark(scaled_image, hough_trans_image):
    new_image = []
    for row in range(len(hough_trans_image)):
        new_row = []
        for index in range(len(hough_trans_image[row])):
            new_row.append(
                [
                    scaled_image[row][index],
                    scaled_image[row][index],
                    scaled_image[row][index] + hough_trans_image[row][index],
                ]
            )
        new_image.append(new_row)

    show_color_image(new_image, "Red imposed lane marks")
    write_image(new_image, "Lane_170094E.jpg")
    print("successfully imposed red in lane marks \n")


# =============================================================================
# image segmentation using region growing
# need to input paramerts: identified pixel values as seeds
# =============================================================================
def segmentation(image, input_seeds):
    label_colors = [
        [232, 71, 66],
        [135, 245, 66],
        [217, 15, 217],
        [18, 15, 217],
        [181, 33, 94],
        [156, 62, 79],
        [86, 219, 146],
        [142, 191, 29],
        [223, 227, 7],
        [217, 166, 165],
        [124, 240, 242],
        [167, 60, 186],
        [221, 155, 232],
        [219, 59, 190],
        [16, 135, 46],
    ]
    image_height = len(image)
    image_width = len(image[0])
    threshold = 3
    segmented_image = [[[index, index, index] for index in row] for row in image]
    test_image = [[0] * image_width for _ in range(image_height)]

    color_index = 0
    for seeds in input_seeds:
        label = label_colors[color_index]
        for seed in seeds:
            i = seed[0]
            i_m_1 = seed[0] - 1
            i_p_1 = seed[0] + 1

            j = seed[1]
            j_m_1 = seed[1] - 1
            j_p_1 = seed[1] + 1

            out = []

            if i_m_1 > 0 and j_m_1 > 0:
                out.append((i_m_1, j_m_1))
            if i_m_1 > 0:
                out.append((i_m_1, j))
            if i_m_1 > 0 and j_p_1 < image_width:
                out.append((i_m_1, j_p_1))
            if j_m_1 > 0:
                out.append((i, j_m_1))
            if j_p_1 < image_width:
                out.append((i, j_p_1))
            if i_p_1 < image_height and j_m_1 > 0:
                out.append((i_p_1, j_m_1))
            if i_p_1 < image_height:
                out.append((i_p_1, j))
            if i_p_1 < image_height and j_p_1 < image_width:
                out.append((i_p_1, j_p_1))

            for coord in out:
                if test_image[coord[0]][coord[1]] == 0:
                    if abs(image[coord[0]][coord[1]] - image[i][j]) <= threshold:
                        segmented_image[coord[0]][coord[1]] = label
                        seeds.append([coord[0], coord[1]])
                        test_image[coord[0]][coord[1]] = -1

        color_index += 1

    show_color_image(segmented_image, "Segmented image")
    write_image(segmented_image, "Segment_170094.jpg")
    print("image segmented successfully !")


# =============================================================================
# main function
# =============================================================================
def main_170094E():
    # read image file as python list of lists
    raw_image = read_file()  # dimensions = 1080 * 1920
    scaled_image = scale(raw_image)  # scale image ; dims = 864 * 1536
    normalized_image = normalize(scaled_image)  # normalize image;
    median_image = median_filter(normalized_image)

    gaussian_image = gaussian_filter_process(median_image)

    canny_edged_image = canny_edge_detection(gaussian_image)
    hough_tranformed_image = hough_tranform(canny_edged_image)
    impose_red_lane_mark(scaled_image, hough_tranformed_image)

    return (
        gaussian_image  # return this normalized, noise filtered image for segmentation
    )


# =============================================================================
# main function to run
# =============================================================================
gaussian_image = main_170094E()


# =============================================================================
# function to do segmentation
# =============================================================================
segmentation(
    gaussian_image,
    [
        [[100, 60]],
        [[100, 500]],
        [[815, 532]],
        [[645, 298]],
        [[646, 538]],
        [[616, 97]],
        [[493, 441]],
        [[610, 377]],
        [[925, 556]],
        [[730, 96]],
        [[793, 581]],
    ],
)
