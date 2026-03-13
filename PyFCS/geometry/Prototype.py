import numpy as np
from typing import List
import subprocess
import os

from pyhull import qvoronoi

### my libraries ###
from PyFCS.geometry.Plane import Plane
from PyFCS.geometry.Point import Point
from PyFCS.geometry.Face import Face
from PyFCS.geometry.Volume import Volume
from PyFCS.colorspace.ReferenceDomain import ReferenceDomain


class Prototype:
    false_negatives = [
    (-5,  -140, -140), (-5,  -140,  140), (-5,   140, -140), (-5,   140,  140),
    (105, -140, -140), (105, -140,  140), (105,  140, -140), (105,  140,  140),
    ]
    

    def __init__(self, label, positive, negatives, voronoi_volume=None, add_false=False):
        self.label = label
        self.positive = positive
        self.negatives = negatives
        self.add_false = add_false
        
        if add_false:
            self.negatives = np.vstack((self.negatives, Prototype.false_negatives))

        if voronoi_volume is not None:
            self.voronoi_volume = voronoi_volume
        else:
            # Create Voronoi volume
            self.voronoi_output = self.run_qvoronoi()
            self.voronoi_volume = self.read_from_voronoi_output()


    @staticmethod
    def get_falseNegatives():
        return Prototype.false_negatives


    def run_qvoronoi(self):
        """
        Run qvoronoi.exe to calculate Voronoi volumes for positive and negative points.

        Returns:
            str: File path of the temporary Voronoi output file.
        """
        try:
            # Get concatenated points
            points = np.vstack((self.positive, self.negatives))

            # VERSION PYHULL -> Problem needs Microsoft C++ Build Tools
            output = qvoronoi("Fi Fo p Fv", points)
            return output

        except Exception as e:
            print(f"Error in execution: {e}")



    def read_from_voronoi_output(self):
        """
        Read Voronoi volumes from self.voronoi_output.

        Returns:
            Volume: Voronoi volume of the positive prototype.
        """
        points = np.vstack((self.positive, self.negatives))

        # qvoronoi devuelve una lista de líneas; si alguna vez devuelve string, lo normalizamos
        lines = self.voronoi_output
        if isinstance(lines, str):
            lines = lines.splitlines()

        num_colors = len(points)
        faces = [[None] * num_colors for _ in range(num_colors)]

        cont = 0

        # Read bounded Voronoi regions
        num_planes = int(lines[0].strip())
        cont += 1
        for i in range(1, num_planes + cont):
            parts = lines[i].split()
            index1 = int(parts[1])
            index2 = int(parts[2])
            plane_params = [float(part) for part in parts[3:]]
            plane = Plane(*plane_params)
            faces[index1][index2] = Face(plane, infinity=False)

        # Read unbounded Voronoi regions
        num_unbounded_planes = int(lines[num_planes + cont].strip())
        cont += 1
        for i in range(num_planes + cont, num_planes + num_unbounded_planes + cont):
            parts = lines[i].split()
            index1 = int(parts[1])
            index2 = int(parts[2])
            plane_params = [float(part) for part in parts[3:]]
            plane = Plane(*plane_params)
            faces[index1][index2] = Face(plane, infinity=True)

        # Read vertex coordinates
        num_dimensions = int(lines[num_planes + num_unbounded_planes + cont].strip())
        cont += 1
        num_vertices = int(lines[num_planes + num_unbounded_planes + cont].strip())
        cont += 1

        vertices = []
        for i in range(num_planes + num_unbounded_planes + cont,
                    num_planes + num_unbounded_planes + num_vertices + cont):
            parts = lines[i].split()
            coords = [float(part) for part in parts]
            vertices.append(coords)

        # Read vertices for each face
        num_faces = int(lines[num_planes + num_unbounded_planes + num_vertices + cont].strip())
        cont += 1
        for i in range(num_planes + num_unbounded_planes + num_vertices + cont,
                    num_planes + num_unbounded_planes + num_vertices + num_faces + cont):
            parts = lines[i].split()
            index1 = int(parts[1])
            index2 = int(parts[2])
            face = faces[index1][index2]

            for j in range(3, int(parts[0]) + 1):
                vertex_index = int(parts[j])
                if vertex_index == 0:
                    face.setInfinity()
                else:
                    face.addVertex(vertices[vertex_index - 1])

        volumes = []
        for point in points:
            volume = Volume(Point(*point))
            volumes.append(volume)

        # Add faces to each fuzzy color
        for i in range(num_colors):
            for j in range(num_colors):
                if faces[i][j] is not None:
                    volumes[i].addFace(faces[i][j])
                    volumes[j].addFace(faces[i][j])

        return volumes[0]

            


















