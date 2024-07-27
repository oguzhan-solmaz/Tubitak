# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 14:59:49 2024

@author: oguzh
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 14:42:55 2024

@author: oguzh
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Jun 29 21:01:14 2024

@author: oguzh
"""

import vtk
import cv2
import mediapipe as mp
import csv

def savepng(rw, filename):
    window_to_image_filter = vtk.vtkWindowToImageFilter()
    window_to_image_filter.SetInput(rw)
    window_to_image_filter.SetScale(1)
    window_to_image_filter.SetInputBufferTypeToRGB()
    window_to_image_filter.ReadFrontBufferOff()
    window_to_image_filter.Update()

    writer = vtk.vtkPNGWriter()
    writer.SetFileName(filename)
    writer.SetInputConnection(window_to_image_filter.GetOutputPort())
    writer.Write()

def create_marker(x, y, z, radius=1.5):
    sphereSource = vtk.vtkSphereSource()
    sphereSource.SetCenter(x, y, z)
    sphereSource.SetRadius(radius)
    
    sphereMapper = vtk.vtkPolyDataMapper()
    sphereMapper.SetInputConnection(sphereSource.GetOutputPort())
    
    sphereActor = vtk.vtkActor()
    sphereActor.SetMapper(sphereMapper)
    
    return sphereActor

def save_landmarks_to_csv(landmarks, filename):
    with open(filename, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['2D_X', '2D_Y', '3D_X', '3D_Y', '3D_Z'])
        for lm in landmarks:
            csvwriter.writerow(lm)

stlfile = "mra10.stl"
reader = vtk.vtkSTLReader()
reader.SetFileName(stlfile)
reader.Update()

# Polydata Mapper oluşturma
mapper = vtk.vtkPolyDataMapper()
mapper.SetInputConnection(reader.GetOutputPort())

# Actor oluşturma ve renklendirme
actor = vtk.vtkActor()
actor.SetMapper(mapper)
actor.GetProperty().SetColor(1.0, 0.5, 0.5)

renderer = vtk.vtkRenderer()
renderer.AddActor(actor)
renderer.SetBackground(0.1, 0.1, 0.1)

# Kameranın sahnedeki aktöre odaklanmasını sağlama
center = actor.GetCenter()
camera = renderer.GetActiveCamera()
camera.SetPosition(center[0], center[1] - 600, center[2])  # Kamerayı y ekseninde geri çekme
camera.SetFocalPoint(center[0], center[1], center[2])      # Kameranın odak noktasını aktörün merkezi yapma
camera.SetViewUp(0, 0, 1)                                  # Kameranın yukarı yönünü z ekseni yapma

render_window = vtk.vtkRenderWindow()
render_window.AddRenderer(renderer)
render_window.SetSize(800, 769)

render_window_interactor = vtk.vtkRenderWindowInteractor()
render_window_interactor.SetRenderWindow(render_window)

render_window.Render()

savepng(render_window, "view_az270_el-270.png")

render_window_interactor.Initialize()
camera.SetViewUp(0, 0, -1)  # landmarkları ters yerleştiriyordu o yüzden kamerayı ters çevirdim (z ekseninde)

# picker = vtk.vtkPropPicker()
picker = vtk.vtkCellPicker()
# picker.SetTolerance(0.00005)  # Increase the pick tolerance for more accuracy
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)

image = cv2.imread('view_az270_el-270.png')
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
results = face_mesh.process(rgb_image)

landmarks = []

if results.multi_face_landmarks:
    for face_landmarks in results.multi_face_landmarks:
        for landmark in face_landmarks.landmark:
            x = int(landmark.x * image.shape[1])
            y = int(landmark.y * image.shape[0])
            picker.Pick(x, y, 0, renderer)
            ras_position = picker.GetPickPosition()
            landmarks.append((x, y, *ras_position))
            
            pactor = create_marker(*ras_position)
            renderer.AddActor(pactor)

            cv2.circle(image, (x, y), 3, (0, 0, 255), -1)  # bu satır rgb görüntüyü landmarklı kaydetmek için (silinebilir)

else:
    print("Yüz landmarkları tespit edilemedi.")

# Save landmarks to CSV
save_landmarks_to_csv(landmarks, 'landmarksmr10.csv')

cv2.imwrite('landmarklıkafa.png', image)

# Render the scene
camera.SetViewUp(0, 0, 1)  # landmarkları yerleştirdikten sonra kamerayı tekrar ters çevirdim. (z ekseninde)
renderer.SetBackground(0.1, 0.2, 0.4)
render_window.Render()
render_window_interactor.Start()
