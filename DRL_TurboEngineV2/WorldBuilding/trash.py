import trimesh
import numpy as np
from PIL import Image

# Load STL
mesh = trimesh.load("Trackv1.stl")

# Define grid resolution
res = 1024
bounds = mesh.bounds
x = np.linspace(bounds[0,0], bounds[1,0], res)
y = np.linspace(bounds[0,1], bounds[1,1], res)
xx, yy = np.meshgrid(x, y)

# Query Z heights
zz = np.full(xx.shape, np.nan)
for i in range(res):
    for j in range(res):
        ray_origins = [[xx[i,j], yy[i,j], bounds[1,2]+1]]
        ray_directions = [[0,0,-1]]
        locs, index_ray, index_tri = mesh.ray.intersects_location(ray_origins, ray_directions)
        if len(locs) > 0:
            zz[i,j] = locs[0][2]

# Normalize to 0-255
zz = np.nan_to_num(zz, nan=bounds[0,2])
zz_norm = (255 * (zz - zz.min()) / (zz.max() - zz.min())).astype(np.uint8)

# Save as PNG
Image.fromarray(zz_norm).save("track_height.png")
