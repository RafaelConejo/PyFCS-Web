import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from PyFCS.geometry.Point import Point


class VisualManager:

    @staticmethod
    def _build_axis_config(title_text, axis_range=None):
        axis = dict(
            title=dict(text=title_text, font=dict(size=12)),
            showgrid=True,
            gridcolor='lightgray',
            gridwidth=1,
            zeroline=True,
            zerolinecolor='gray',
            showline=True,
            linecolor='gray',
            showbackground=False,
            ticks='outside'
        )
        if axis_range is not None:
            axis["range"] = axis_range
        return axis

    @staticmethod
    def plot_more_combined_3D(
        filename,
        color_data,
        core,
        alpha,
        support,
        volume_limits,
        hex_color,
        selected_options,
        filtered_points=None
    ):
        """Generates a 3D figure in Plotly combining centroids, prototypes, and filtered points."""
        fig = go.Figure()

        # Estado local para evitar problemas en entorno web
        show_legends = True
        has_filtered = bool(
            filtered_points and any(len(pts) > 0 for pts in filtered_points.values())
        )

        # Mapa inverso LAB -> HEX para evitar búsquedas repetidas
        lab_to_hex = {
            tuple(np.asarray(v).tolist()): k
            for k, v in hex_color.items()
        }

        # ---------- Helper: triangulate polygonal faces ----------
        def triangulate_face(vertices):
            triangles = []
            for i in range(1, len(vertices) - 1):
                triangles.append([vertices[0], vertices[i], vertices[i + 1]])
            return triangles

        # ---------- Plot centroids ----------
        def plot_centroids():
            if not color_data:
                return

            lab_values = [v['positive_prototype'] for v in color_data.values()]
            lab_array = np.array(lab_values)
            L, A, B = lab_array[:, 0], lab_array[:, 1], lab_array[:, 2]

            colors = [
                lab_to_hex.get(tuple(np.asarray(lab).tolist()), "#000000")
                for lab in lab_values
            ]

            fig.add_trace(go.Scatter3d(
                x=A,
                y=B,
                z=L,
                mode='markers',
                marker=dict(
                    size=5,
                    color=colors,
                    opacity=0.8,
                    line=dict(color='black', width=1)
                ),
                name="Centroids"
            ))

        # ---------- Plot prototypes ----------
        def plot_prototypes(prototypes):
            nonlocal show_legends

            if not prototypes:
                return

            for prototype in prototypes:
                color = lab_to_hex.get(
                    tuple(np.asarray(prototype.positive).tolist()),
                    "#000000"
                )

                vertices, faces = [], []

                for face in prototype.voronoi_volume.faces:
                    if face.infinity:
                        continue

                    clipped = VisualManager.clip_face_to_volume(
                        np.array(face.vertex),
                        volume_limits
                    )

                    if len(clipped) < 3:
                        continue

                    # reorder to a*, b*, L*
                    clipped = clipped[:, [1, 2, 0]]
                    triangles = triangulate_face(clipped)

                    for tri in triangles:
                        idx0 = len(vertices)
                        vertices.extend(tri)
                        faces.append([idx0, idx0 + 1, idx0 + 2])

                if vertices:
                    vertices = np.array(vertices)
                    current_show_legend = (not has_filtered) and show_legends

                    fig.add_trace(go.Mesh3d(
                        x=vertices[:, 0],
                        y=vertices[:, 1],
                        z=vertices[:, 2],
                        i=[f[0] for f in faces],
                        j=[f[1] for f in faces],
                        k=[f[2] for f in faces],
                        color=color,
                        opacity=0.5,
                        name=prototype.label,
                        showlegend=current_show_legend,
                        legendgroup=prototype.label,
                    ))

            show_legends = False

        # ---------- Plot filtered points ----------
        def plot_filtered_points(prototypes, point_color='black'):
            """Plot filtered points only if they are inside the actual prototype volume."""
            if not filtered_points or not prototypes or volume_limits is None:
                return

            border_colors = px.colors.qualitative.Dark24

            for proto_name, points in filtered_points.items():
                try:
                    idx = int(str(proto_name).split("_")[-1])  # e.g., "Volume_3" -> 3
                except (ValueError, TypeError, AttributeError):
                    continue

                if idx < 0 or idx >= len(prototypes):
                    continue

                prototype = prototypes[idx]

                # Filtrar puntos realmente dentro del volumen
                points_inside = []
                for p in points:
                    try:
                        if len(p) == 3 and prototype.voronoi_volume.isInside(Point(*p)):
                            points_inside.append(p)
                    except Exception:
                        continue

                if not points_inside:
                    continue

                pts = np.array(points_inside)
                L, A, B = pts[:, 0], pts[:, 1], pts[:, 2]

                point_size = max(4, min(10, int(300 / len(points_inside))))
                border_color = border_colors[idx % len(border_colors)]

                # 1) Trazado real
                fig.add_trace(go.Scatter3d(
                    x=A,
                    y=B,
                    z=L,
                    mode='markers',
                    marker=dict(
                        size=point_size,
                        color=point_color,
                        opacity=0.7,
                        line=dict(color=border_color, width=0.8)
                    ),
                    name=prototype.label,
                    legendgroup=prototype.label,
                    showlegend=False,
                ))

                # 2) Trazado dummy solo para la leyenda
                x_dummy = [volume_limits.comp2[0] - 10]
                y_dummy = [volume_limits.comp3[0] - 10]
                z_dummy = [volume_limits.comp1[0] - 10]

                fig.add_trace(go.Scatter3d(
                    x=x_dummy,
                    y=y_dummy,
                    z=z_dummy,
                    mode='markers',
                    marker=dict(
                        size=8,
                        color=border_color,
                        opacity=1.0,
                        symbol='circle'
                    ),
                    name=prototype.label,
                    legendgroup=prototype.label,
                    showlegend=True,
                ))

        # ---------- Mapping of user-selected options ----------
        options_map = {
            "Representative": plot_centroids,
            "0.5-cut": lambda: plot_prototypes(alpha),
            "Core": lambda: plot_prototypes(core),
            "Support": lambda: plot_prototypes(support),
        }

        # Ejecutar trazados principales
        for option in selected_options:
            plot_fn = options_map.get(option)
            if plot_fn is not None:
                plot_fn()

        # Llamar solo una vez a los puntos filtrados
        plot_filtered_points(core or alpha or support)

        # ---------- Configure axes and layout ----------
        if volume_limits:
            scene_axes = dict(
                xaxis=VisualManager._build_axis_config('a* (Red-Green)', volume_limits.comp2),
                yaxis=VisualManager._build_axis_config('b* (Blue-Yellow)', volume_limits.comp3),
                zaxis=VisualManager._build_axis_config('L* (Luminosity)', volume_limits.comp1),
            )
        else:
            scene_axes = dict(
                xaxis=VisualManager._build_axis_config('a* (Red-Green)'),
                yaxis=VisualManager._build_axis_config('b* (Blue-Yellow)'),
                zaxis=VisualManager._build_axis_config('L* (Luminosity)'),
            )

        fig.update_layout(
            scene=dict(
                aspectmode='cube',
                camera=dict(
                    eye=dict(x=1.55, y=1.55, z=1.2)
                ),
                **scene_axes
            ),
            margin=dict(l=10, r=10, b=10, t=35),
            title=dict(
                text=filename,
                font=dict(size=12),
                x=0.5,
                xanchor='center',
                y=0.97
            ),
            legend=dict(
                x=0.01,
                y=0.99,
                bgcolor="rgba(255,255,255,0.65)"
            ),
            paper_bgcolor='white'
        )

        return fig

    @staticmethod
    def get_intersection_with_cube(A, B, C, D, volume_limits):
        intersections = []

        x_min, x_max = volume_limits.comp1
        y_min, y_max = volume_limits.comp2
        z_min, z_max = volume_limits.comp3

        def solve_plane_for_x(y, z):
            if A != 0:
                return -(B * y + C * z + D) / A
            return None

        def solve_plane_for_y(x, z):
            if B != 0:
                return -(A * x + C * z + D) / B
            return None

        def solve_plane_for_z(x, y):
            if C != 0:
                return -(A * x + B * y + D) / C
            return None

        # Intersections with the Z = constant faces (XY planes)
        for z in [z_min, z_max]:
            for y in [y_min, y_max]:
                x = solve_plane_for_x(y, z)
                if x is not None and x_min <= x <= x_max:
                    intersections.append((x, y, z))

        # Intersections with the Y = constant faces (XZ planes)
        for y in [y_min, y_max]:
            for z in [z_min, z_max]:
                x = solve_plane_for_x(y, z)
                if x is not None and x_min <= x <= x_max:
                    intersections.append((x, y, z))

        # Intersections with the X = constant faces (YZ planes)
        for x in [x_min, x_max]:
            for z in [z_min, z_max]:
                y = solve_plane_for_y(x, z)
                if y is not None and y_min <= y <= y_max:
                    intersections.append((x, y, z))

        return np.array(intersections)

    @staticmethod
    def order_points_by_angle(points):
        centroid = np.mean(points, axis=0)
        angles = np.arctan2(points[:, 1] - centroid[1], points[:, 0] - centroid[0])
        ordered_indices = np.argsort(angles)
        return points[ordered_indices]

    @staticmethod
    def clip_face_to_volume(vertices, volume_limits):
        """
        Adjusts a face to the specified volume limits.
        """
        adjusted_vertices = []

        for vertex in vertices:
            if isinstance(vertex, Point):
                vertex = vertex.get_double_point()

            adjusted_vertex = np.array([
                np.clip(vertex[0], volume_limits.comp1[0], volume_limits.comp1[1]),
                np.clip(vertex[1], volume_limits.comp2[0], volume_limits.comp2[1]),
                np.clip(vertex[2], volume_limits.comp3[0], volume_limits.comp3[1]),
            ])
            adjusted_vertices.append(adjusted_vertex)

        return np.array(adjusted_vertices)