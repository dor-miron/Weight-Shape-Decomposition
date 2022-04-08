from itertools import product
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

NO_CLUSTER = -1


def cluster_by_gradient_descent(image):
    clustered_image = np.full_like(image, NO_CLUSTER)
    cluster_count = 0
    shuffled_pixel_list = get_shuffled_pixel_list(image)
    for pixel in shuffled_pixel_list:
        cur_pixel_group = [pixel]
        while True:
            cur_pixel_cluster = clustered_image[cur_pixel_group[-1]]
            if cur_pixel_cluster != NO_CLUSTER:
                for pixel in cur_pixel_group:
                    clustered_image[pixel] = cur_pixel_cluster
                break
            direction = get_direction(image, cur_pixel_group[-1])
            if direction == (0, 0):
                if len(cur_pixel_group) != 1:
                    same_level_pixels = get_pixels_on_same_level(image, cur_pixel_group[-1])
                    new_cluster = cluster_count
                    cluster_count += 1
                    for pixel in cur_pixel_group:
                        clustered_image[pixel] = new_cluster
                    for pixel in same_level_pixels:
                        clustered_image[pixel] = new_cluster
                break
            old_pixel = cur_pixel_group[-1]
            cur_pixel_group.append((old_pixel[0] + direction[0], old_pixel[1] + direction[1]))

    return clustered_image

def get_pixels_on_same_level(image, pixel):
    value = image[pixel]
    plateau_list = [pixel]
    unsearched_pixel_list = [pixel]
    while len(unsearched_pixel_list) > 0:
        cur_pixel = unsearched_pixel_list.pop()
        for direction in product([-1, 0, 1], [-1, 0, 1]):
            new_pixel = (cur_pixel[0] + direction[0], cur_pixel[1] + direction[1])
            if new_pixel in plateau_list:
                continue
            try:
                if image[new_pixel] == value:
                    unsearched_pixel_list.append(new_pixel)
                    plateau_list.append(new_pixel)
            except IndexError:
                pass
    return plateau_list

def get_shuffled_pixel_list(image):
    pixels = list(list(zip(*np.ndenumerate(image)))[0])
    # np.random.shuffle(pixels)

    for px in pixels:
        if px[0] < 0 or px[1] < 0:
            raise ValueError
    return pixels


def get_direction(image, pixel):
    # x_grad_array, y_grad_array = np.gradient(image)
    # grad = np.array([x_grad_array[pixel], y_grad_array[pixel]])
    # argument = np.arctan2(*grad)

    best_direction = (0, 0)
    min_delta = 0
    for direction in product([-1, 0, 1], [-1, 0, 1]):
        cur_pixel = (pixel[0] + direction[0], pixel[1] + direction[1])
        if cur_pixel[0] < 0 or cur_pixel[1] < 0:
            continue
        try:
            cur_delta = image[cur_pixel] - image[pixel]
            if cur_delta < min_delta:
                best_direction = direction
                min_delta = cur_delta
        except IndexError:
            pass

    return best_direction


def test_sine():
    x = np.arange(50)
    y = np.arange(50)
    X, Y = np.meshgrid(x, y)
    Z = np.sin(X / 5) * np.sin(Y / 4)

    clustered = cluster_by_gradient_descent(Z)
    fig = make_subplots(2, 1)
    fig.add_trace(go.Heatmap(z=Z), 1, 1)
    fig.add_trace(go.Heatmap(z=clustered), 2, 1)
    fig.show()


def test_polynom():
    n = 50
    x = np.arange(n)
    y = np.arange(n)
    X, Y = np.meshgrid(x, y)
    Z = 0.01 * (X - n/2) ** 2

    clustered = cluster_by_gradient_descent(Z)
    fig = make_subplots(2, 1)
    fig.add_trace(go.Heatmap(z=Z), 1, 1)
    fig.add_trace(go.Heatmap(z=clustered), 2, 1)
    fig.show()


if __name__ == '__main__':
    # test_polynom()
    test_sine()