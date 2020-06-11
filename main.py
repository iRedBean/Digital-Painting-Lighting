import argparse
import os
import math
import numpy as np
import taichi as ti
from scipy.spatial import ConvexHull
from skimage import io, filters

# 笔画密度
k = None
# 凸包顶点
hull_p = None
# 凸包的面
hull_f = None

# 光线与三角形求交
@ti.func
def intersect_triangle(o, d, a, b, c):
    ok = False
    intersection = o
    norm = (b - a).cross(c - a)
    norm /= ti.sqrt(norm.dot(norm))
    n_dot_d = norm.dot(d)
    if ti.abs(n_dot_d) >= 1e-7:
        n_dot_ao = norm.dot(a - o)
        t = n_dot_ao / n_dot_d
        if t >= 1e-7:
            p = o + d * t
            u = b - a
            v = c - a
            w = p - a
            uu = u.dot(u)
            uv = u.dot(v)
            vv = v.dot(v)
            wu = w.dot(u)
            wv = w.dot(v)
            r = 1.0 / (uv * uv - uu * vv)
            s = (uv * wv - vv * wu) * r
            t = (uv * wu - uu * wv) * r
            if s >= 0 and s <= 1.0 and t >= 0 and (s + t) <= 1.0:
                ok, intersection = True, p
    return ok, intersection

# 光线与凸包求交
@ti.kernel
def intersect_hull(gx: ti.f32, gy: ti.f32, gz: ti.f32, k_len: ti.i32, hull_len: ti.i32):
    g = ti.Vector([gx, gy, gz])
    for i in range(k_len):
        c = hull_p[i]
        d = c - g
        h = c
        for j in range(hull_len):
            ok, p = intersect_triangle(g, d, hull_p[hull_f[j][0]], hull_p[hull_f[j][1]], hull_p[hull_f[j][2]])
            if ok:
                h = p
                break
        k[i] = ti.min(ti.max(ti.sqrt((c - g).dot(c - g) / (h - g).dot(h - g)), 0.0), 1.0)

def triangle_area(a, b, c):
    norm = np.cross(b - a, c - a)
    return 0.5 * np.sqrt(np.dot(norm, norm))

def estimate_stroke_density(img):
    global k, hull_p, hull_f

    h, w, _ = img.shape

    # 计算RGB空间的凸包及重心
    rgb_coords = img.reshape((-1, 3))
    hull = ConvexHull(rgb_coords)
    g = np.zeros(3)
    for i in range(hull.simplices.shape[0]):
        a = hull.points[hull.simplices[i, 0]]
        b = hull.points[hull.simplices[i, 1]]
        c = hull.points[hull.simplices[i, 2]]
        g += triangle_area(a, b, c) / 3.0 * (a + b + c)
    g /= hull.area

    # 与凸包求交并计算笔画密度
    k = ti.var(dt=ti.f32, shape=h*w)
    hull_p = ti.Vector(3, dt=ti.f32, shape=h*w)
    hull_f = ti.Vector(3, dt=ti.i32, shape=hull.simplices.shape[0])
    hull_p.from_numpy(hull.points)
    hull_f.from_numpy(hull.simplices)
    intersect_hull(float(g[0]), float(g[1]), float(g[2]), h*w, hull.simplices.shape[0])
    stroke = k.to_numpy().reshape(h, w)
    return stroke

def generate_lighting_effect(img, light_pos, stroke):
    h, w, _ = img.shape

    # 高斯模糊 + 归一化
    n = [img[:,:,0], img[:,:,1], img[:,:,2]]
    for i in range(3):
        n[i] = filters.gaussian(n[i], sigma=21)
        n[i] = (n[i] - np.min(n[i])) / (np.max(n[i]) - np.min(n[i]))

    # 计算图像上每个位置的光源方向
    coords = np.zeros(img.shape)
    coords[:,:,0] = np.arange(h * w).reshape((h, w)) % w
    coords[:,:,1] = np.arange(h * w).reshape((h, w)) // w
    light_dir = light_pos - coords
    light_dir /= np.sqrt(np.sum(light_dir ** 2, axis=2, keepdims=True))

    # 生成光效
    e = []
    for i in range(3):
        dx = filters.sobel_v(n[i])
        dy = filters.sobel_h(n[i])
        normal = np.stack([-dx, -dy, np.ones((h, w)) * 0.2], axis=2)
        normal /= np.sqrt(np.sum(normal ** 2, axis=2, keepdims=True))
        e.append(np.sum(normal * light_dir, axis=2).clip(0, 1) * stroke)
    return np.stack(e, axis=2)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Digital Painting Lighting')
    parser.add_argument('--img_path', type=str, default='')
    parser.add_argument('--out_path', type=str, default='./output')
    parser.add_argument('--intensity', type=float, default=2.0)
    parser.add_argument('--ambient', type=float, default=0.55)
    parser.add_argument('--light_dist', type=float, default=1000.0)
    parser.add_argument('--light_h', type=float, default=300.0)
    args = parser.parse_args()

    with open(os.path.join(args.out_path, '_config.txt'), 'wt') as f:
        for k, v in vars(args).items():
            f.write('{}: {}\n'.format(k, v))

    ti.init(arch=ti.gpu)

    img = io.imread(args.img_path)[:,:,:3].astype(np.float32) / 255.0
    stroke = estimate_stroke_density(img)
    io.imsave(os.path.join(args.out_path, 'stroke_density.png'), stroke)

    for theta in range(30):
        x = img.shape[1] / 2 + args.light_dist * math.cos(math.radians(theta * 12))
        y = img.shape[0] / 2 + args.light_dist * math.sin(math.radians(theta * 12))
        lighting = generate_lighting_effect(img, np.array((x, y, args.light_h)), stroke)
        io.imsave(os.path.join(args.out_path, 'lighting_%02d.png' % theta), lighting.clip(0, 1))
        result = (lighting * args.intensity + args.ambient) * img
        io.imsave(os.path.join(args.out_path, '%02d.png' % theta), result.clip(0, 1))
