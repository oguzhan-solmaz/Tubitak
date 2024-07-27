import trimesh
from mayavi import mlab
import numpy as np

# Mesh'i yükle
mesh = trimesh.load('smoothed_output_model8mra.obj')

# Mesh'i döndürme ve kaydetme fonksiyonu
def save_view(mesh, filename, azimuth, elevation):
    mlab.figure(size=(500, 600), bgcolor=(1, 1, 1))
    vertices = mesh.vertices
    faces = mesh.faces
    x, y, z = vertices.T
    mlab.triangular_mesh(x, y, z, faces, color=(1, 1, 0))  # Sarı renk
    mlab.view(azimuth=azimuth, elevation=elevation, distance='auto')
    mlab.savefig(filename)
    mlab.clf()  # Grafiği temizle

# Mesh'in kopyalarını kullanarak her açıdan ve ters açıdan görselleri kaydet
angles = [0, 45, 90, 135, 180, 225, 270, 315]

for azimuth in angles:
    for elevation in angles:
        filename = f'view_az{azimuth}_el{elevation}.png'
        save_view(mesh.copy(), filename, azimuth=azimuth, elevation=elevation)

# Ters açılardan görselleri kaydet
for azimuth in angles:
    for elevation in angles:
        filename = f'view_az{azimuth}_el{-elevation}.png'
        save_view(mesh.copy(), filename, azimuth=azimuth, elevation=-elevation)

print("Tüm açılardan ve ters açılardan görseller başarıyla kaydedildi.")
