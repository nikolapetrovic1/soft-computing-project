import cv2
import numpy as np
import scipy.spatial as spatial
import scipy.cluster as cluster
from collections import defaultdict
from statistics import mean


def read_image(img):
    img = cv2.imread(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blur = cv2.blur(gray, (1, 1))
    return img, blur

# Canny edge detection


def canny_edge(img):
    v = np.median(img)
    lower = int(max(0, (1.0 - 0.1) * v))
    upper = int(min(255, (1.0 + 0.1) * v))
    edges = cv2.Canny(img, lower, upper)
    return edges


def hough_line(edges, min_line_length=10, max_line_gap=25):
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 100,
                           min_line_length, max_line_gap)
    linesP = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, None, 50, 10)
    lines = np.reshape(lines, (-1, 2))
    linesP = np.reshape(linesP, (-1, 2))
    return lines


def h_v_lines(lines):
    h_lines, v_lines = [], []
    for rho, theta in lines:
        if theta < np.pi / 4 or theta > np.pi - np.pi / 4:
            v_lines.append([rho, theta])
        else:
            h_lines.append([rho, theta])
    return h_lines, v_lines


def line_intersections(h_lines, v_lines):
    points = []
    for r_h, t_h in h_lines:
        for r_v, t_v in v_lines:
            a = np.array([[np.cos(t_h), np.sin(t_h)],
                         [np.cos(t_v), np.sin(t_v)]])
            b = np.array([r_h, r_v])
            inter_point = np.linalg.solve(a, b)
            points.append(inter_point)
    return np.array(points)


def cluster_points(points):
    dists = spatial.distance.pdist(points)
    single_linkage = cluster.hierarchy.single(dists)
    flat_clusters = cluster.hierarchy.fcluster(single_linkage, 25, 'distance')
    cluster_dict = defaultdict(list)
    for i in range(len(flat_clusters)):
        cluster_dict[flat_clusters[i]].append(points[i])
    cluster_values = cluster_dict.values()
    clusters = map(lambda arr: (np.mean(np.array(arr)[:, 0]), np.mean(
        np.array(arr)[:, 1])), cluster_values)
    return sorted(list(clusters), key=lambda k: [k[1], k[0]])
