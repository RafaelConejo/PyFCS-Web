import numpy as np
import re

from pyhull import qvoronoi

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
        self.positive = np.asarray(positive, dtype=float).reshape(3,)
        self.negatives = np.asarray(negatives, dtype=float).reshape(-1, 3)
        self.add_false = add_false

        if add_false:
            self.negatives = np.vstack((self.negatives, Prototype.false_negatives)).astype(float)

        if voronoi_volume is not None:
            self.voronoi_volume = voronoi_volume
        else:
            self.voronoi_output = self.run_qvoronoi()
            self.voronoi_volume = self.read_from_voronoi_output()

    @staticmethod
    def get_falseNegatives():
        return Prototype.false_negatives

    def run_qvoronoi(self):
        """
        Run qvoronoi to calculate Voronoi volumes for positive and negative points.

        Returns
        -------
        list[str] | str
            Raw qvoronoi output.

        Raises
        ------
        RuntimeError
            If qvoronoi fails or returns no output.
        """
        try:
            points = np.vstack((self.positive, self.negatives))
            output = qvoronoi("Fi Fo p Fv", points)

            if output is None:
                raise RuntimeError("qvoronoi returned None")

            if isinstance(output, list) and len(output) == 0:
                raise RuntimeError("qvoronoi returned an empty list")

            if isinstance(output, str) and not output.strip():
                raise RuntimeError("qvoronoi returned an empty string")

            return output

        except Exception as e:
            raise RuntimeError(f"Error running qvoronoi for prototype {self.label!r}: {e}") from e

    def _normalize_qvoronoi_lines(self):
        """
        Normalize qvoronoi output into a clean list of stripped lines.

        In some environments qvoronoi output may contain:
        - leading empty lines
        - diagnostic text like 'While executing:' or 'Menu'
        - qhull banner/options text

        We keep all non-empty lines, then parsing starts from the first integer line.
        """
        raw = self.voronoi_output

        if raw is None:
            raise RuntimeError(f"qvoronoi output is None for prototype {self.label!r}")

        if isinstance(raw, str):
            lines = raw.splitlines()
        else:
            lines = [str(x) for x in raw]

        lines = [line.strip() for line in lines if str(line).strip()]
        if not lines:
            raise RuntimeError(f"qvoronoi returned no usable lines for prototype {self.label!r}")

        return lines

    @staticmethod
    def _is_int_line(text: str) -> bool:
        return re.fullmatch(r"-?\d+", text) is not None

    def read_from_voronoi_output(self):
        """
        Read Voronoi volumes from self.voronoi_output.

        Returns
        -------
        Volume
            Voronoi volume of the positive prototype.

        Raises
        ------
        RuntimeError
            If qvoronoi output is malformed or does not match the expected Fi/Fo/p/Fv layout.
        """
        points = np.vstack((self.positive, self.negatives))
        num_colors = len(points)

        lines = self._normalize_qvoronoi_lines()

        # Find the first integer line; this should be the first section count.
        start_idx = None
        for i, line in enumerate(lines):
            if self._is_int_line(line):
                start_idx = i
                break

        if start_idx is None:
            preview = "\n".join(lines[:20])
            raise RuntimeError(
                f"Could not find the first numeric section in qvoronoi output for prototype {self.label!r}.\n"
                f"First lines received:\n{preview}"
            )

        if start_idx > 0:
            # Skip diagnostic/banner lines on server deployments
            lines = lines[start_idx:]

        faces = [[None] * num_colors for _ in range(num_colors)]
        cont = 0

        def _require_line(idx: int, what: str) -> str:
            if idx >= len(lines):
                raise RuntimeError(
                    f"Unexpected end of qvoronoi output while reading {what} "
                    f"for prototype {self.label!r} at line index {idx}."
                )
            return lines[idx]

        def _require_int(idx: int, what: str) -> int:
            text = _require_line(idx, what)
            if not self._is_int_line(text):
                raise RuntimeError(
                    f"Expected integer for {what} in qvoronoi output, got {text!r} "
                    f"for prototype {self.label!r}."
                )
            return int(text)

        # ---- Read bounded Voronoi regions (Fi) ----
        num_planes = _require_int(0, "number of bounded planes")
        cont += 1

        for i in range(1, num_planes + cont):
            parts = _require_line(i, "bounded plane row").split()
            if len(parts) < 7:
                raise RuntimeError(
                    f"Malformed bounded plane row {parts!r} for prototype {self.label!r}."
                )

            index1 = int(parts[1])
            index2 = int(parts[2])
            plane_params = [float(part) for part in parts[3:]]

            plane = Plane(*plane_params)
            faces[index1][index2] = Face(plane, infinity=False)

        # ---- Read unbounded Voronoi regions (Fo) ----
        num_unbounded_planes = _require_int(num_planes + cont, "number of unbounded planes")
        cont += 1

        for i in range(num_planes + cont, num_planes + num_unbounded_planes + cont):
            parts = _require_line(i, "unbounded plane row").split()
            if len(parts) < 7:
                raise RuntimeError(
                    f"Malformed unbounded plane row {parts!r} for prototype {self.label!r}."
                )

            index1 = int(parts[1])
            index2 = int(parts[2])
            plane_params = [float(part) for part in parts[3:]]

            plane = Plane(*plane_params)
            faces[index1][index2] = Face(plane, infinity=True)

        # ---- Read vertex coordinate section (p) ----
        num_dimensions = _require_int(
            num_planes + num_unbounded_planes + cont,
            "number of dimensions"
        )
        cont += 1

        num_vertices = _require_int(
            num_planes + num_unbounded_planes + cont,
            "number of vertices"
        )
        cont += 1

        vertices = []
        for i in range(
            num_planes + num_unbounded_planes + cont,
            num_planes + num_unbounded_planes + num_vertices + cont
        ):
            parts = _require_line(i, "vertex row").split()
            if len(parts) != num_dimensions:
                raise RuntimeError(
                    f"Vertex row has {len(parts)} coords but expected {num_dimensions}: {parts!r} "
                    f"for prototype {self.label!r}."
                )
            coords = [float(part) for part in parts]
            vertices.append(coords)

        # ---- Read face -> vertices section (Fv) ----
        num_faces = _require_int(
            num_planes + num_unbounded_planes + num_vertices + cont,
            "number of faces"
        )
        cont += 1

        for i in range(
            num_planes + num_unbounded_planes + num_vertices + cont,
            num_planes + num_unbounded_planes + num_vertices + num_faces + cont
        ):
            parts = _require_line(i, "face-vertex row").split()
            if len(parts) < 3:
                raise RuntimeError(
                    f"Malformed face-vertex row {parts!r} for prototype {self.label!r}."
                )

            count = int(parts[0])
            index1 = int(parts[1])
            index2 = int(parts[2])

            face = faces[index1][index2]
            if face is None:
                raise RuntimeError(
                    f"Face reference ({index1}, {index2}) not found while reading vertices "
                    f"for prototype {self.label!r}."
                )

            max_j = min(count + 1, len(parts))
            for j in range(3, max_j):
                vertex_index = int(parts[j])
                if vertex_index == 0:
                    face.setInfinity()
                else:
                    if vertex_index - 1 < 0 or vertex_index - 1 >= len(vertices):
                        raise RuntimeError(
                            f"Vertex index {vertex_index} out of bounds for prototype {self.label!r}."
                        )
                    face.addVertex(vertices[vertex_index - 1])

        volumes = []
        for point in points:
            volume = Volume(Point(*point))
            volumes.append(volume)

        for i in range(num_colors):
            for j in range(num_colors):
                if faces[i][j] is not None:
                    volumes[i].addFace(faces[i][j])
                    volumes[j].addFace(faces[i][j])

        return volumes[0]















