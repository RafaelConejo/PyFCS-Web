import os
from nicegui import ui
import tempfile
import sys
import numpy as np
from skimage import color as skcolor
import base64
import io
import asyncio
from PIL import Image
from nicegui import context
import matplotlib.pyplot as plt
import io
from PIL import Image, ImageDraw, ImageFont




### current path ###
current_dir = os.path.dirname(__file__)
pyfcs_dir = os.path.abspath(os.path.join(current_dir, '..', '..'))

# Add the PyFCS path to sys.path
sys.path.append(pyfcs_dir)

from PyFCS import Input, VisualManager, ReferenceDomain, FuzzyColorSpace, FuzzyColorSpaceManager, ColorEvaluationManager
import PyFCS.interface.modules.UtilsTools as UtilsTools

class PyFCSWebApp:
    def __init__(self):
        self.COLOR_SPACE = False
        self.model_3d_options = {
            "Representative": True,
            "Core": False,
            "0.5-cut": False,
            "Support": False,
        }
        self.color_checkboxes = {}  # nombre -> checkbox (para select/deselect)
        self.modified_image = {}          # window_id -> np.uint8(H,W,3) última imagen recoloreada
        self.label_map_cache = {}         # window_id -> np.array(H,W) con strings (labels)
        self.scheme_cache = {}            # window_id -> 'centroid' | 'hsv'
        self.mapping_all_cache = {}        # (window_id, scheme, max_side) -> data_url

        self.proto_map_cache = {}          # (window_id, proto_label, max_side, scheme) -> (data_url, pct_text)
        self.proto_membership_cache = {}   # (window_id, proto_label, max_side) -> membership_uint8



        # estado por imagen 
        self.MEMBERDEGREE = {}          # colores: color_name -> bool
        self.MEMBERDEGREE_IMG = {}      # imágenes: window_id -> bool
        self.ORIGINAL_IMG = {}          # window_id -> bool

        # loading UI references
        self.loading_dialog = None
        self.loading_label = None
        self.loading_progress = None

        self.fuzzy_manager = FuzzyColorSpaceManager()  # root=None
        self.volume_limits = ReferenceDomain(0, 100, -128, 127, -128, 127)

        self.build_layout()

    def build_layout(self):
        ui.page_title('PyFCS Interface (Web)')

        ui.add_head_html('''
        <style>
        body {
            transform: scale(1);
            transform-origin: top left;
            width: 100%;
        }
        </style>
        ''')

        ui.add_head_html('''
        <script>
            window.__pyfcsZ = window.__pyfcsZ || 2000;

            function bringToFront(el) {
            window.__pyfcsZ += 1;
            el.style.zIndex = window.__pyfcsZ;
            }

            function makeDraggable(elId, handleId) {
            const el = document.getElementById(elId);
            const handle = document.getElementById(handleId);
            if (!el || !handle) return;

            // Asegura posicionamiento
            el.style.position = 'absolute';
            el.style.willChange = 'left, top';

            // Evita scroll/touch gestures durante drag
            handle.style.cursor = 'move';
            handle.style.userSelect = 'none';
            handle.style.touchAction = 'none';

            let dragging = false;
            let startX = 0, startY = 0;
            let startLeft = 0, startTop = 0;

            // RAF para suavidad
            let raf = 0;
            let nextLeft = 0, nextTop = 0;

            function applyPos() {
                raf = 0;
                el.style.left = nextLeft + 'px';
                el.style.top  = nextTop  + 'px';
            }

            function onMove(e) {
                if (!dragging) return;

                const dx = e.clientX - startX;
                const dy = e.clientY - startY;

                nextLeft = startLeft + dx;
                nextTop  = startTop  + dy;

                if (!raf) raf = requestAnimationFrame(applyPos);
                e.preventDefault();
            }

            function onUp(e) {
                if (!dragging) return;
                dragging = false;

                window.removeEventListener('pointermove', onMove, true);
                window.removeEventListener('pointerup', onUp, true);

                try { el.releasePointerCapture(e.pointerId); } catch {}
                e.preventDefault();
            }

            handle.addEventListener('pointerdown', (e) => {
                // SOLO botón izquierdo y SOLO si pinchas el handle (no menu)
                if (e.button !== 0) return;

                dragging = true;
                bringToFront(el);

                // offsetLeft/Top => sin saltos raros por rect/scroll
                startLeft = el.offsetLeft;
                startTop  = el.offsetTop;

                startX = e.clientX;
                startY = e.clientY;

                // Captura para no perder el drag
                try { el.setPointerCapture(e.pointerId); } catch {}

                // Escuchamos en window para ir fino incluso si sales del handle
                window.addEventListener('pointermove', onMove, true);
                window.addEventListener('pointerup', onUp, true);

                e.preventDefault();
                e.stopPropagation();
            });

            // Click normal en la ventana -> al frente (sin mover)
            el.addEventListener('mousedown', () => bringToFront(el), {passive: true});
            }
            </script>

        ''')






        # ---- Header / Menus ----
        with ui.header(elevated=True).classes('items-center'):
            ui.label('PyFCS Interface').classes('text-lg font-bold')

            with ui.row().classes('gap-2'):
                with ui.menu():
                    ui.menu_item('Exit', on_click=self.exit_app)
                    ui.button('File', icon='menu').props('flat')

                with ui.menu():
                    ui.menu_item('Open Image', on_click=self.open_image)
                    ui.menu_item('Save Image', on_click=self.save_image)
                    ui.menu_item('Close All', on_click=self.close_all_image)
                    ui.button('Image Manager', icon='image').props('flat')

                with ui.menu():
                    ui.menu_item('New Color Space', on_click=self.show_menu_create_fcs)
                    ui.menu_item('Load Color Space', on_click=self.load_color_space)
                    ui.button('Fuzzy Color Space', icon='palette').props('flat')

                ui.button('About', on_click=self.about_info).props('flat')

        # ---- Toolbar cards (sin Color Evaluation) ----
        with ui.row().classes('w-full q-pa-md items-start gap-4'):
            # Image Manager (más compacto)
            with ui.card().classes('w-[350px] q-pa-md'):
                ui.label('Image Manager').classes('font-bold')
                with ui.row().classes('gap-2'):
                    ui.button('Open Image', icon='folder_open', on_click=self.open_image)
                    ui.button('Save Image', icon='save', on_click=self.save_image)

            # Fuzzy Color Space Manager (más ancho y en una fila)
            with ui.card().classes('w-[450px] q-pa-md'):
                ui.label('Fuzzy Color Space Manager').classes('font-bold')
                with ui.row().classes('gap-2 items-center'):
                    ui.button('New Color Space', icon='add', on_click=self.show_menu_create_fcs)
                    ui.button('Load Color Space', icon='upload_file', on_click=self.load_color_space)

        # ---- Main split (full height minus header+toolbar) ----
        with ui.splitter(value=30).classes('w-full h-[calc(100vh-150px)] q-pa-md') as splitter:
            # LEFT: Image
            with splitter.before:
                with ui.card().classes('w-full h-full'):
                    ui.label('Image Display').classes('font-bold')
                    self.image_workspace = ui.element('div').classes(
                        'relative w-full h-[calc(100%-32px)] bg-white overflow-hidden'
                    )

            # RIGHT: Tabs
            with splitter.after:
                with ui.tabs().classes('w-full') as tabs:
                    model_tab = ui.tab('Model 3D')
                    data_tab = ui.tab('Data')

                with ui.tab_panels(tabs, value=model_tab).classes('w-full h-[calc(100%-48px)]'):
                    # ---- Model 3D ----
                    with ui.tab_panel(model_tab).classes('w-full h-full'):
                        with ui.row().classes('items-center q-gutter-md q-pa-sm'):
                            for name in ["Representative", "Core", "0.5-cut", "Support"]:
                                ui.checkbox(
                                    name,
                                    value=self.model_3d_options[name],
                                    on_change=lambda e, n=name: self.set_model_option(n, e.value),
                                )
                            #ui.space()
                            #ui.button('Interactive Figure', on_click=self.open_interactive_figure)

                        with ui.splitter(value=78).classes('w-full h-[calc(100%-52px)]') as inner:
                            # Center: plot
                            with inner.before:
                                with ui.card().classes('w-full h-full'):
                                    self.plot_container = ui.column().classes('w-full h-full')
                                    with self.plot_container:
                                        ui.label('3D plot').classes('text-gray-500')

                            # Right: colors
                            with inner.after:
                                with ui.card().classes('w-full h-full'):
                                    with ui.row().classes('justify-end gap-2 q-pa-sm'):
                                        ui.button('Select All', on_click=self.select_all_color)
                                        ui.button('Deselect All', on_click=self.deselect_all_color)

                                    self.colors_scroll = ui.scroll_area().classes('w-full h-[calc(100%-56px)] q-pa-sm')
                                    with self.colors_scroll:
                                        self.set_color_list([])

                    # ---- Data ----
                    with ui.tab_panel(data_tab).classes('w-full h-full'):
                        with ui.card().classes('w-full q-pa-md'):
                            ui.label('Name:').classes('font-bold')
                            self.file_name = ui.input(placeholder='').classes('w-80')

                        self.data_table = ui.table(
                            columns=[
                                {'name': 'name', 'label': 'Name', 'field': 'name'},
                                {'name': 'L', 'label': 'L*', 'field': 'L'},
                                {'name': 'a', 'label': 'a*', 'field': 'a'},
                                {'name': 'b', 'label': 'b*', 'field': 'b'},
                            ],
                            rows=[],
                            row_key='name',
                        ).classes('w-full')

                        with ui.row().classes('q-pa-md gap-2'):
                            ui.button('Add New Color', on_click=self.addColor_data_window)
                            ui.button('Apply Changes', on_click=self.apply_changes)






    # ---------- UI helpers ----------
    def set_color_list(self, color_names: list[str]) -> None:
        """(Re)create the checkbox list in the right panel and refresh plot on change."""
        self.color_checkboxes.clear()
        self.colors_scroll.clear()

        # inicializa MEMBERDEGREE si no existe
        if not hasattr(self, 'MEMBERDEGREE') or not isinstance(self.MEMBERDEGREE, dict):
            self.MEMBERDEGREE = {}

        with self.colors_scroll:
            for name in color_names:
                # valor inicial: True si no existe aún, o el que tuviera guardado
                initial = self.MEMBERDEGREE.get(name, True)

                cb = ui.checkbox(
                    name,
                    value=initial,
                    on_change=lambda e, n=name: self.on_color_toggle(n, e.value),
                )
                self.color_checkboxes[name] = cb
                self.MEMBERDEGREE[name] = initial

    
    def list_preset_fcs(self) -> dict:
        """Return a dict: label -> absolute_path for preset .fcs files shipped with the app."""
        presets_dir = os.path.abspath(os.path.join(current_dir, '..', '..', 'fuzzy_color_spaces'))
        if not os.path.isdir(presets_dir):
            return {}

        files = sorted([f for f in os.listdir(presets_dir) if f.lower().endswith('.fcs')])
        return {os.path.splitext(f)[0]: os.path.join(presets_dir, f) for f in files}



    def on_color_toggle(self, name: str, value: bool):
        # guarda estado
        self.MEMBERDEGREE[name] = value

        # recalcula selección (filtra por checkboxes)
        self.update_selected_sets_from_checks()

        # repinta
        self.on_option_select()

    def update_selected_sets_from_checks(self):
        """Update selected_* collections based on MEMBERDEGREE."""
        if not hasattr(self, 'color_data') or not self.color_data:
            return

        enabled = {k for k, v in self.MEMBERDEGREE.items() if v}

        # selected_centroids: dict por nombre como en FCS
        self.selected_centroids = {k: v for k, v in self.color_data.items() if k in enabled}

        # hex_color: tu mapping es hex -> lab; lo dejamos igual (no hace falta filtrarlo),
        # pero si quieres filtrarlo estrictamente, se puede.
        self.selected_hex_color = getattr(self, 'hex_color', {})

        # prototipos/core/support: son listas de Prototype con .label
        def _filter_by_label(protos):
            if not protos:
                return protos
            return [p for p in protos if getattr(p, 'label', None) in enabled]

        self.selected_alpha = _filter_by_label(getattr(self, 'prototypes', None))
        self.selected_core = _filter_by_label(getattr(self, 'cores', None))
        self.selected_support = _filter_by_label(getattr(self, 'supports', None))




    # ---------- Web equivalents of your Tkinter utils ----------
    def custom_warning(self, title="Warning", message="Warning"):
        with ui.dialog() as d, ui.card():
            ui.label(title).classes('text-lg font-bold')
            ui.label(message).classes('text-gray-700')
            with ui.row().classes('justify-end'):
                ui.button('OK', on_click=d.close)
        d.open()

    def show_loading(self, message="Processing..."):
        if self.loading_dialog is None:
            self.loading_dialog = ui.dialog()
            with self.loading_dialog, ui.card().classes('w-80'):
                self.loading_label = ui.label(message).classes('text-base font-bold')
                ui.spinner(size='lg')
                self.loading_progress = ui.linear_progress(0).props(
                    'instant-feedback show-value=false indeterminate'
                )
        else:
            self.loading_label.set_text(message)
            self.loading_progress.set_value(0)
            self.loading_progress.props('indeterminate')

        self.loading_dialog.open()

    def show_loading_color_space(self):
        self.show_loading("Loading Color Space...")

    def set_loading_progress(self, value_0_to_1: float):
        if self.loading_progress is not None:
            self.loading_progress.props(remove='indeterminate')
            self.loading_progress.set_value(round(max(0.0, min(1.0, value_0_to_1)), 2))


    def hide_loading(self):
        if self.loading_dialog is not None:
            self.loading_dialog.close()

    # ---------- callbacks ----------
    def exit_app(self):
        # En web: no cerramos el servidor desde el cliente.
        with ui.dialog() as d, ui.card():
            ui.label('Exit').classes('text-lg font-bold')
            ui.label('Close this tab to exit the application.')
            with ui.row().classes('justify-end'):
                ui.button('OK', on_click=d.close)
        d.open()

    def about_info(self):
        with ui.dialog() as d, ui.card().classes('w-[600px]'):
            ui.label("About PyFCS").classes('text-lg font-bold')
            ui.label(
                "PyFCS: Python Fuzzy Color Software\n"
                "A color modeling Python Software based on Fuzzy Color Spaces.\n"
                "Version 1.0\n\n"
                "Contact: rafaconejo@ugr.es"
            ).style('white-space: pre-line;')
            with ui.row().classes('justify-end'):
                ui.button("Close", on_click=d.close)
        d.open()

    def show_menu_create_fcs(self):
        with ui.dialog() as d, ui.card():
            ui.label('Create New Color Space').classes('text-lg font-bold')
            ui.label('Choose a creation mode:')
            with ui.row().classes('gap-2'):
                ui.button('Palette-Based', on_click=lambda: (d.close(), self.palette_based_creation()))
                ui.button('Image-Based', on_click=lambda: (d.close(), self.image_based_creation()))
            with ui.row().classes('justify-end'):
                ui.button('Cancel', on_click=d.close).props('flat')
        d.open()

    # ----- still-stubs (we'll implement next) -----
    def open_image(self): ui.notify('Open Image (stub)')
    def save_image(self): ui.notify('Save Image (stub)')
    def close_all_image(self): ui.notify('Close All (stub)')
    def load_color_space(self): ui.notify('Load Color Space (stub)')
    def palette_based_creation(self): ui.notify('Palette Based (stub)')
    def image_based_creation(self): ui.notify('Image Based (stub)')
    def addColor_data_window(self): ui.notify('Add New Color (stub)')
    def apply_changes(self): ui.notify('Apply Changes (stub)')
    def open_interactive_figure(self): ui.notify('Interactive Figure (stub)')

    def select_all_color(self):
        for cb in self.color_checkboxes.values():
            cb.set_value(True)

    def deselect_all_color(self):
        for cb in self.color_checkboxes.values():
            cb.set_value(False)

    def set_model_option(self, name, value):
        self.model_3d_options[name] = value
        self.on_option_select()






    def load_color_space(self):
        presets = self.list_preset_fcs()

        with ui.dialog() as d, ui.card().classes('w-[560px]'):
            ui.label('Load Color Space').classes('text-lg font-bold')

            # --- Presets ---
            if presets:
                ui.label('Load a preset (.fcs) from the server:').classes('text-sm text-gray-700')
                preset_select = ui.select(
                    options=list(presets.keys()),
                    value=list(presets.keys())[0],
                    label='Presets',
                ).classes('w-full')

                with ui.row().classes('justify-end gap-2'):
                    ui.button(
                        'Load preset',
                        icon='cloud_download',
                        on_click=lambda: (d.close(), self.load_color_space_from_path(presets[preset_select.value]))
                    )
            else:
                ui.label('No presets found on the server.').classes('text-sm text-gray-500')

            ui.separator()

            # --- Upload ---
            ui.label('Or upload a .cns/.fcs file:').classes('text-sm text-gray-700')
            ui.upload(
                label='Choose file',
                multiple=False,
                auto_upload=True,
                on_upload=lambda e: self._on_color_file_uploaded(e, d),
            ).props('accept=.cns,.fcs')

            with ui.row().classes('justify-end'):
                ui.button('Cancel', on_click=d.close).props('flat')
        d.open()


    
    def load_color_space_from_path(self, filepath: str):
        self.show_loading_color_space()
        try:
            data = self.fuzzy_manager.load_color_file(filepath)

            self.file_path = filepath
            self.file_base_name = os.path.splitext(os.path.basename(filepath))[0]
            self.file_name.set_value(self.file_base_name)

            if data['type'] == 'cns':
                self.color_data = data['color_data']
                self.display_data_window()
                self.update_volumes()

            elif data['type'] == 'fcs':
                self.color_data = data['color_data']
                self.fuzzy_color_space = data['fuzzy_color_space']
                self.cores = self.fuzzy_color_space.cores
                self.supports = self.fuzzy_color_space.supports
                self.prototypes = self.fuzzy_color_space.prototypes
                self.fuzzy_color_space.precompute_pack()

                self.display_data_window()
                self.update_prototypes_info()

            ui.notify('Loaded preset successfully')

        except Exception as ex:
            self.custom_warning('File Error', str(ex))
        finally:
            self.hide_loading()





    async def _on_color_file_uploaded(self, e, dialog):
        dialog.close()
        self.show_loading_color_space()

        tmp_path = None
        try:
            original_name = e.file.name
            content_bytes = await e.file.read()   # ← AQUÍ estaba el problema

            suffix = os.path.splitext(original_name)[1].lower() or '.tmp'
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(content_bytes)
                tmp_path = tmp.name

            # reutilizamos tu parser tal cual
            data = self.fuzzy_manager.load_color_file(tmp_path)

            self.file_path = original_name
            self.file_base_name = os.path.splitext(os.path.basename(original_name))[0]
            self.file_name.set_value(self.file_base_name)

            if data['type'] == 'cns':
                self.color_data = data['color_data']
                self.display_data_window()
                self.update_volumes()

            elif data['type'] == 'fcs':
                self.color_data = data['color_data']
                self.fuzzy_color_space = data['fuzzy_color_space']
                self.cores = self.fuzzy_color_space.cores
                self.supports = self.fuzzy_color_space.supports
                self.prototypes = self.fuzzy_color_space.prototypes
                self.fuzzy_color_space.precompute_pack()

                self.display_data_window()
                self.update_prototypes_info()

            ui.notify(f'Loaded {data["type"].upper()} successfully')

        except Exception as ex:
            self.custom_warning("File Error", str(ex))

        finally:
            self.hide_loading()
            if tmp_path:
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass




    def update_volumes(self):
        self.show_loading("Computing volumes...")

        # mismas llamadas que en desktop
        self.prototypes = UtilsTools.process_prototypes(self.color_data)

        self.fuzzy_color_space = FuzzyColorSpace(space_name=" ", prototypes=self.prototypes)
        self.fuzzy_color_space.precompute_pack()

        # OJO: en tu desktop usas get_cores/get_supports
        self.cores = self.fuzzy_color_space.get_cores()
        self.supports = self.fuzzy_color_space.get_supports()

        self.update_prototypes_info()
        self.hide_loading()


    def update_prototypes_info(self):
        self.COLOR_SPACE = True

        # si no existe aún, la creamos
        if not hasattr(self, 'MEMBERDEGREE') or not isinstance(self.MEMBERDEGREE, dict):
            self.MEMBERDEGREE = {}

        # en desktop: {key: True for key in self.MEMBERDEGREE}
        # aquí lo más útil es: todos los colores a True
        if hasattr(self, 'color_matrix') and self.color_matrix:
            self.MEMBERDEGREE = {name: True for name in self.color_matrix}

        self.selected_centroids = self.color_data
        self.selected_hex_color = getattr(self, 'hex_color', {})
        self.selected_alpha = self.prototypes
        self.selected_core = self.cores
        self.selected_support = self.supports

        self.update_selected_sets_from_checks()
        self.on_option_select()

    
    def on_option_select(self):
        if not self.COLOR_SPACE:
            return

        selected_options = [k for k, v in self.model_3d_options.items() if v]

        # si no hay opciones, limpia el plot
        if not selected_options:
            self.draw_model_3D(None, selected_options)
            return

        try:
            fig = VisualManager.plot_more_combined_3D(
                self.file_base_name,
                self.selected_centroids,
                self.selected_core,
                self.selected_alpha,
                self.selected_support,
                self.volume_limits,
                self.hex_color,
                selected_options,
                filtered_points=getattr(self, 'filtered_points', None),
            )
            self.draw_model_3D(fig, selected_options)
        except Exception as ex:
            self.custom_warning("Plot Error", str(ex))



    def draw_model_3D(self, fig, selected_options):
        self.plot_container.clear()
        with self.plot_container:
            if fig is None:
                ui.label('No 3D option selected').classes('text-gray-500')
            else:
                ui.plotly(fig).classes('w-full h-full')




    def display_data_window(self):
        """
        WEB version of display_data_window:
        - Fill Name input
        - Fill Data table (L,a,b,Label,Color preview)
        - Build hex_color mapping and color_matrix list
        - Refresh right panel color list
        """
        # 1) Name field
        if hasattr(self, 'file_base_name') and self.file_base_name:
            self.file_name.set_value(self.file_base_name)

        # 2) Reset containers like in desktop
        self.hex_color = {}      # hex -> lab
        self.color_matrix = []   # list of color names

        rows = []

        # 3) Iterate your FCS color_data dict
        #    color_data: {color_name: {'positive_prototype': array([L,a,b]), ...}, ...}
        for color_name, color_value in (self.color_data or {}).items():
            # prefer positive_prototype, fallback to Color
            lab = color_value.get('positive_prototype', None)
            if lab is None:
                lab = color_value.get('Color', None)

            lab = np.array(lab, dtype=float).reshape(3,)
            self.color_matrix.append(color_name)

            # LAB -> RGB -> HEX (like desktop)
            rgb01 = skcolor.lab2rgb(lab.reshape(1, 1, 3))[0, 0, :]   # values 0..1
            rgb255 = tuple(int(max(0, min(1, c)) * 255) for c in rgb01)
            hex_color = f'#{rgb255[0]:02x}{rgb255[1]:02x}{rgb255[2]:02x}'
            self.hex_color[hex_color] = lab

            # Color preview cell (HTML div)
            preview = f'''
            <div style="
                width:110px;height:24px;border:1px solid #000;
                background:{hex_color};margin:auto;border-radius:4px;">
            </div>
            '''

            rows.append({
                'L': round(float(lab[0]), 2),
                'a': round(float(lab[1]), 2),
                'b': round(float(lab[2]), 2),
                'name': color_name,
                'color': preview,
            })

        # 4) Push to table + refresh
        # IMPORTANT: we add/ensure the 'color' column exists and uses html format
        self.data_table.columns = [
            {'name': 'L', 'label': 'L*', 'field': 'L'},
            {'name': 'a', 'label': 'a*', 'field': 'a'},
            {'name': 'b', 'label': 'b*', 'field': 'b'},
            {'name': 'name', 'label': 'Name', 'field': 'name'},
            {'name': 'color', 'label': 'Color', 'field': 'color', 'html': True},
        ]
        self.data_table.rows = rows
        self.data_table.update()

        # 5) Right panel list matches desktop behaviour
        self.set_color_list(self.color_matrix)





    def _palette_toggle(self, name: str, value: bool):
        if hasattr(self, 'color_checks') and name in self.color_checks:
            self.color_checks[name]["value"] = value


    def palette_based_creation(self):
        color_space_path = os.path.join(pyfcs_dir, 'fuzzy_color_spaces', 'cns', 'ISCC_NBS_BASIC.cns')
        colors = UtilsTools.load_color_data(color_space_path)

        self.palette_colors = colors
        self.color_checks = {}  # name -> {'value': bool, 'lab': ..., 'rgb': ...}

        # ✅ inicial: todo desmarcado
        for color_name, data in colors.items():
            self.color_checks[color_name] = {
                "value": False,          # <-- aquí
                "lab": data.get("lab"),
                "rgb": data.get("rgb"),
            }

        with ui.dialog() as d, ui.card().classes('w-[560px] h-[680px]'):
            ui.label('Select colors for your Color Space').classes('text-lg font-bold')

            # Guardamos el scroll y un contenedor interno para poder repintar
            self.palette_dialog = d
            self.palette_scroll = ui.scroll_area().classes('w-full h-[560px] border rounded q-pa-sm')
            with self.palette_scroll:
                self.palette_list_container = ui.column().classes('w-full gap-1')

            # pinta lista inicial
            self.render_palette_list()

            with ui.row().classes('justify-end gap-2 w-full'):
                ui.button('Add New Color', icon='add', on_click=self.addColor_create_fcs)
                ui.button('Create Color Space', icon='save', on_click=self.create_color_space)
                ui.button('Close', on_click=d.close).props('flat')

        d.open()



    def render_palette_list(self):
        """Render the palette list from self.color_checks into the dialog scroll area."""
        if not hasattr(self, 'palette_list_container'):
            return

        self.palette_list_container.clear()

        for color_name, data in self.color_checks.items():
            lab = data.get("lab")
            rgb = data.get("rgb")
            checked = bool(data.get("value", False))

            # lab puede ser dict o lista/np.array
            if isinstance(lab, dict):
                L, A, B = float(lab["L"]), float(lab["A"]), float(lab["B"])
            else:
                arr = np.array(lab, dtype=float).reshape(3,)
                L, A, B = float(arr[0]), float(arr[1]), float(arr[2])

            # preview color
            hexcol = None
            if rgb is not None and hasattr(UtilsTools, "rgb_to_hex"):
                hexcol = UtilsTools.rgb_to_hex(rgb)

            lab_text = f"L: {L:.1f}, a: {A:.1f}, b: {B:.1f}"

            with self.palette_list_container:
                with ui.row().classes('w-full items-center justify-between q-pa-xs border-b'):
                    # cuadrito color
                    if hexcol:
                        ui.html(
                            f'<div style="width:18px;height:18px;border:1px solid #000;'
                            f'background:{hexcol};border-radius:3px;"></div>',
                            sanitize=False,
                        )
                    else:
                        ui.label('■').classes('text-gray-500')

                    # nombre + LAB
                    with ui.column().classes('gap-0'):
                        ui.label(color_name).classes('text-sm font-medium')
                        ui.label(lab_text).classes('text-xs text-gray-600')

                    # checkbox (con callback)
                    ui.checkbox(
                        '',
                        value=checked,
                        on_change=lambda e, n=color_name: self._palette_toggle(n, e.value),
                    )




    def addColor_create_fcs(self):
        with ui.dialog() as d, ui.card().classes('w-[420px]'):
            ui.label('Add New Color').classes('text-lg font-bold')
            name_in = ui.input(label='Color name').classes('w-full')
            l_in = ui.number(label='L*', value=50, format='%.2f').classes('w-full')
            a_in = ui.number(label='a*', value=0, format='%.2f').classes('w-full')
            b_in = ui.number(label='b*', value=0, format='%.2f').classes('w-full')

            def _add():
                name = (name_in.value or '').strip()
                if not name:
                    self.custom_warning("Warning", "Color name is required.")
                    return
                if name in self.color_checks:
                    self.custom_warning("Warning", "That color name already exists.")
                    return

                lab = np.array([float(l_in.value), float(a_in.value), float(b_in.value)], dtype=float)

                # opcional: calcula rgb para preview usando tu UtilsTools si existe
                rgb = None
                if hasattr(UtilsTools, "lab_to_rgb"):
                    try:
                        rgb = UtilsTools.lab_to_rgb({'L': lab[0], 'A': lab[1], 'B': lab[2]})
                    except Exception:
                        rgb = None

                self.color_checks[name] = {"value": True, "lab": lab, "rgb": rgb}

                d.close()
                self.render_palette_list()
                ui.notify('Color added')

            with ui.row().classes('justify-end gap-2'):
                ui.button('Cancel', on_click=d.close).props('flat')
                ui.button('Add', on_click=_add)

        d.open()









    def create_color_space(self):
        # Extraer seleccionados en LAB 
        selected_colors_lab = {}
        for name, data in (getattr(self, 'color_checks', {}) or {}).items():
            if data.get("value", False):
                lab = data.get("lab")
                if isinstance(lab, dict):
                    selected_colors_lab[name] = np.array([lab["L"], lab["A"], lab["B"]], dtype=float)
                else:
                    selected_colors_lab[name] = np.array(lab, dtype=float)

        if len(selected_colors_lab) < 2:
            self.custom_warning("Warning", "At least two colors must be selected to create the Color Space.")
            return

        # pedir nombre en dialog web
        with ui.dialog() as d, ui.card().classes('w-[420px]'):
            ui.label('Color Space Name').classes('text-lg font-bold')
            name_input = ui.input(label='Name for the fuzzy color space').classes('w-full')

            def _ok():
                cs_name = (name_input.value or '').strip()
                if not cs_name:
                    self.custom_warning("Warning", "Please enter a name.")
                    return
                d.close()
                self.save_cs(cs_name, selected_colors_lab)

            with ui.row().classes('justify-end gap-2'):
                ui.button('Cancel', on_click=d.close).props('flat')
                ui.button('OK', on_click=_ok)

        d.open()


    def save_cs(self, name: str, selected_colors_lab: dict):
        self.show_loading("Creating .fcs file...")

        tmp_path = None
        try:
            # crear temporal
            tmp_dir = tempfile.mkdtemp(prefix='pyfcs_')
            tmp_path = os.path.join(tmp_dir, f'{name}.fcs')

            # Reutilizar tu InputFCS.write_file (vía Input.instance('.fcs'))
            input_fcs = Input.instance('.fcs')

            # NECESITAS que write_file acepte file_path (cambio mínimo recomendado)
            input_fcs.write_file(name, selected_colors_lab, file_path=tmp_path)

            # descargar al navegador
            ui.download(tmp_path, filename=f'{name}.fcs')

            ui.notify('Download started')

        except Exception as ex:
            self.custom_warning("Save Error", str(ex))

        finally:
            self.hide_loading()






    # IMAGE MANAGER
    def list_preset_images(self) -> dict:
        """Return dict: label -> absolute_path for preset images shipped with the app."""
        presets_dir = os.path.abspath(os.path.join(current_dir, '..', '..', 'image_test'))
        if not os.path.isdir(presets_dir):
            return {}

        exts = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp')
        files = sorted(f for f in os.listdir(presets_dir) if f.lower().endswith(exts))
        return {os.path.splitext(f)[0]: os.path.join(presets_dir, f) for f in files}


    def open_image(self):
        presets = self.list_preset_images()

        with ui.dialog() as d, ui.card().classes('w-[560px]'):
            ui.label('Open Image').classes('text-lg font-bold')

            # --- Presets ---
            if presets:
                ui.label('Load a preset image from the server:').classes('text-sm text-gray-700')
                preset_select = ui.select(
                    options=list(presets.keys()),
                    value=list(presets.keys())[0],
                    label='Presets',
                ).classes('w-full')

                with ui.row().classes('justify-end gap-2'):
                    ui.button(
                        'Load preset',
                        icon='image',
                        on_click=lambda: (d.close(), self.create_floating_window(presets[preset_select.value], preset_select.value))
                    )
            else:
                ui.label('No preset images found on the server.').classes('text-sm text-gray-500')

            ui.separator()

            # --- Upload ---
            ui.label('Or upload an image:').classes('text-sm text-gray-700')
            ui.upload(
                label='Choose image',
                multiple=False,
                auto_upload=True,
                on_upload=lambda e: self._on_image_uploaded(e, d),
            ).props('accept=image/*')

            with ui.row().classes('justify-end'):
                ui.button('Cancel', on_click=d.close).props('flat')

        d.open()



    async def _on_image_uploaded(self, e, dialog):
        dialog.close()
        self.show_loading("Loading image...")

        tmp_path = None
        try:
            original_name = e.file.name
            content_bytes = await e.file.read()

            suffix = os.path.splitext(original_name)[1].lower() or '.png'
            tmp_dir = tempfile.mkdtemp(prefix='pyfcs_img_')
            tmp_path = os.path.join(tmp_dir, f'uploaded{suffix}')

            with open(tmp_path, 'wb') as f:
                f.write(content_bytes)

            self.create_floating_window(tmp_path, display_name=original_name)

        except Exception as ex:
            self.custom_warning("Image Error", str(ex))
        finally:
            self.hide_loading()


    def create_floating_window(self, filename: str, display_name: str | None = None):
        """Web 'floating window': a card with title bar + menu + close + image."""
        if not hasattr(self, "image_windows"):
            self.image_windows = {}

        # --- ensure caches exist (so we can clear them on close) ---
        if not hasattr(self, "label_map_cache"):
            self.label_map_cache = {}          # (window_id, max_side) -> label_map
        if not hasattr(self, "mapping_all_cache"):
            self.mapping_all_cache = {}        # (window_id, scheme, max_side) -> data_url
        if not hasattr(self, "proto_map_cache"):
            self.proto_map_cache = {}          # (window_id, chosen, max_side, scheme) -> (data_url, info_text)
        if not hasattr(self, "proto_membership_cache"):
            self.proto_membership_cache = {}   # (window_id, chosen, max_side) -> gray_uint8
        if not hasattr(self, "scheme_cache"):
            self.scheme_cache = {}             # window_id -> 'centroid' | 'hsv'

        window_id = f"img_{len(self.image_windows) + 1}"
        title = display_name or os.path.basename(filename)

        # estado por ventana
        if not hasattr(self, "ORIGINAL_IMG"):
            self.ORIGINAL_IMG = {}
        self.ORIGINAL_IMG.setdefault(window_id, True)

        if hasattr(self, "MEMBERDEGREE_IMG"):
            self.MEMBERDEGREE_IMG.setdefault(window_id, bool(self.COLOR_SPACE))

        # posición inicial en cascada
        x0 = 20 + 30 * (len(self.image_windows) % 6)
        y0 = 20 + 30 * (len(self.image_windows) % 6)

        # ✅ crea la entrada del dict ANTES de usarla
        self.image_windows[window_id] = {
            "path": filename,
            "title": title,
            "original_source": filename,
            "current_source": filename,
            "card": None,
            "img": None,
            "legend_box": None,
            "legend_title": None,
            "legend_scroll": None,
            "legend_info": None,
            "alt_colors_btn": None,
            "legend_visible": False,
        }

        with self.image_workspace:
            card = ui.card().classes('w-[320px] max-w-[700px] max-h-[700px]').props(f'id={window_id}').style(
                f'position:absolute; left:{x0}px; top:{y0}px; z-index:2000; resize: both; overflow: auto;'
                'min-width:220px; min-height:220px;'
            )
            self.image_windows[window_id]["card"] = card

            with card:
                # ✅ UNA sola title bar (sin duplicar)
                handle_id = f'{window_id}_handle'
                with ui.row().classes('w-full items-center justify-between q-pa-sm bg-gray-200'):
                    ui.label(title).classes('text-sm font-bold select-none').props(f'id={handle_id}')

                    with ui.row().classes('gap-1'):
                        with ui.menu() as m:
                            ui.menu_item('Original Image', on_click=lambda wid=window_id: self.show_original_image(wid))
                            ui.menu_item('Color Mapping', on_click=lambda wid=window_id: self.color_mapping(wid))
                            ui.menu_item('Color Mapping All', on_click=lambda wid=window_id: self.color_mapping_all(wid))
                            # ✅ sin Toggle Legend (como querías)

                        ui.button(icon='more_vert', on_click=m.open).props('flat dense')
                        ui.button(icon='close', on_click=lambda wid=window_id: self.close_image_window(wid)).props('flat dense')

                # Image (crece con el resize del card)
                img = ui.image(filename).classes('w-full h-auto object-contain bg-white q-ma-sm')
                self.image_windows[window_id]["img"] = img

                # Legend/controls container (oculto al inicio) ✅ más compacto
                legend_box = ui.card().classes('w-full q-ma-sm q-pa-sm').style('display:none;')
                with legend_box:
                    legend_title = ui.label('Legend').classes('font-bold text-sm')
                    legend_scroll = ui.scroll_area().classes('w-full h-[110px] q-pa-xs')
                    legend_info = ui.label('').classes('text-xs text-gray-600')

                    alt_btn = ui.button(
                        'Alt. Colors',
                        on_click=lambda wid=window_id: self.toggle_color_scheme(wid),
                    ).props('dense')

                self.image_windows[window_id]["legend_box"] = legend_box
                self.image_windows[window_id]["legend_title"] = legend_title
                self.image_windows[window_id]["legend_scroll"] = legend_scroll
                self.image_windows[window_id]["legend_info"] = legend_info
                self.image_windows[window_id]["alt_colors_btn"] = alt_btn

        # activar drag (espera a que el DOM exista)
        ui.timer(
            0.05,
            lambda: ui.run_javascript(f"makeDraggable('{window_id}', '{window_id}_handle');"),
            once=True,
        )



    def toggle_legend(self, window_id: str):
        win = self.image_windows.get(window_id)
        if not win:
            return
        win["legend_visible"] = not win.get("legend_visible", False)
        box = win.get("legend_box")
        if box:
            box.style('display:block;' if win["legend_visible"] else 'display:none;')


    def close_image_window(self, window_id: str):
        win = getattr(self, "image_windows", {}).get(window_id)
        if not win:
            return

        # elimina del DOM
        if win.get("card") is not None:
            win["card"].delete()

        # --- limpiar caches relacionadas con esa imagen ---
        for k in list(getattr(self, "label_map_cache", {}).keys()):
            if k[0] == window_id:
                del self.label_map_cache[k]

        for k in list(getattr(self, "mapping_all_cache", {}).keys()):
            if k[0] == window_id:
                del self.mapping_all_cache[k]

        for k in list(getattr(self, "proto_map_cache", {}).keys()):
            if k[0] == window_id:
                del self.proto_map_cache[k]

        for k in list(getattr(self, "proto_membership_cache", {}).keys()):
            if k[0] == window_id:
                del self.proto_membership_cache[k]

        if hasattr(self, "scheme_cache") and window_id in self.scheme_cache:
            del self.scheme_cache[window_id]

        if hasattr(self, "ORIGINAL_IMG") and window_id in self.ORIGINAL_IMG:
            del self.ORIGINAL_IMG[window_id]

        if hasattr(self, "MEMBERDEGREE_IMG") and window_id in self.MEMBERDEGREE_IMG:
            del self.MEMBERDEGREE_IMG[window_id]

        del self.image_windows[window_id]


    def color_mapping(self, window_id: str):
        if not getattr(self, 'COLOR_SPACE', False):
            self.custom_warning("No Color Space", "Load a color space first (.cns or .fcs).")
            return

        labels = list(getattr(self, 'color_matrix', []) or [])
        if not labels:
            self.custom_warning("No Data", "No colors loaded to map.")
            return

        # --- ensure caches exist ---
        if not hasattr(self, "proto_map_cache"):
            self.proto_map_cache = {}
        if not hasattr(self, "proto_membership_cache"):
            self.proto_membership_cache = {}
        if not hasattr(self, "scheme_cache"):
            self.scheme_cache = {}

        with ui.dialog() as d, ui.card().classes('w-[520px]'):
            ui.label('Color Mapping (single prototype)').classes('text-lg font-bold')
            ui.label('Membership map for ONE prototype (grayscale).').classes('text-sm text-gray-600')

            sel = ui.select(options=labels, value=labels[0], label='Prototype').classes('w-full')

            async def _apply():
                d.close()
                self.show_loading("Color Mapping...")

                try:
                    max_side = 400

                    # 1) reduced working image (HxWx3 uint8)
                    img_np = self._get_work_image_np(window_id, max_side=max_side)

                    chosen = sel.value
                    scheme = self.scheme_cache.get(window_id, 'centroid')  # no afecta a grayscale, pero lo metemos en cache_key por consistencia
                    cache_key = (window_id, chosen, max_side, scheme)

                    # ✅ cache final render (instant)
                    if cache_key in self.proto_map_cache:
                        data_url, info_text = self.proto_map_cache[cache_key]

                        win = self.image_windows[window_id]
                        win["img"].set_source(data_url)
                        win["current_source"] = data_url

                        self._render_legend(
                            window_id,
                            only_labels=[chosen],
                            info=info_text,
                            mode='single',
                        )
                        self.ORIGINAL_IMG[window_id] = True
                        if hasattr(self, "MEMBERDEGREE"):
                            self.MEMBERDEGREE[window_id] = False
                        return

                    # 2) compute membership map ONLY for this prototype (cached)
                    mkey = (window_id, chosen, max_side)
                    if mkey in self.proto_membership_cache:
                        gray = self.proto_membership_cache[mkey]
                    else:
                        proto_index = self._proto_index_by_label(chosen)
                        gray = await asyncio.to_thread(self._membership_map_for_prototype, img_np, proto_index)
                        self.proto_membership_cache[mkey] = gray

                    # 3) show as grayscale RGB for ui.image
                    out = np.stack([gray, gray, gray], axis=-1)

                    pct = float((gray > 0).sum()) / float(gray.size) * 100.0
                    info_text = f'Selected: {chosen} — {pct:.2f}% (nonzero membership)'

                    self.modified_image[window_id] = out
                    data_url = self._np_to_data_url(out)

                    win = self.image_windows[window_id]
                    win["img"].set_source(data_url)
                    win["current_source"] = data_url

                    # 4) legend only chosen, and NO alt colors
                    self._render_legend(
                        window_id,
                        only_labels=[chosen],
                        info=info_text,
                        mode='single',
                    )

                    # ✅ cache final
                    self.proto_map_cache[cache_key] = (data_url, info_text)

                    self.ORIGINAL_IMG[window_id] = True
                    if hasattr(self, "MEMBERDEGREE"):
                        self.MEMBERDEGREE[window_id] = False

                    # (opcional) auto-mostrar legend al aplicar
                    if not self.image_windows[window_id].get("legend_visible", False):
                        self.toggle_legend(window_id)

                except Exception as e:
                    self.custom_warning("Processing Error", str(e))
                finally:
                    self.hide_loading()

            with ui.row().classes('justify-end gap-2'):
                ui.button('Cancel', on_click=d.close).props('flat')
                ui.button('Apply', icon='palette', on_click=_apply)

        d.open()








    def _np_to_data_url(self, arr_uint8: np.ndarray) -> str:
        """np(H,W,3 uint8) -> data:image/png;base64,..."""
        im = Image.fromarray(arr_uint8, mode='RGB')
        buf = io.BytesIO()
        im.save(buf, format='PNG')
        b64 = base64.b64encode(buf.getvalue()).decode('ascii')
        return f'data:image/png;base64,{b64}'


    def _label_colors_centroid(self) -> dict:
        if not hasattr(self, 'color_data') or not self.color_data:
            return {}

        out = {}
        for label, v in self.color_data.items():
            lab = v.get('positive_prototype', None)
            if lab is None:
                lab = v.get('Color', None)
            if lab is None:
                continue

            lab = np.array(lab, dtype=float).reshape(1, 1, 3)
            rgb01 = skcolor.lab2rgb(lab)[0, 0]
            rgb255 = (np.clip(rgb01, 0, 1) * 255).astype(np.uint8)
            out[label] = rgb255
        return out



    def _label_colors_hsv(self) -> dict:
        """label -> rgb_uint8 using hsv colormap."""
        labels = list(getattr(self, 'color_matrix', []) or [])
        if not labels:
            return {}
        cmap = plt.get_cmap('hsv', len(labels))
        out = {}
        for i, lab in enumerate(labels):
            rgb01 = np.array(cmap(i)[:3], dtype=float)
            rgb255 = (np.clip(rgb01, 0, 1) * 255).astype(np.uint8)
            # mantiene 'black' negro
            if lab.lower() == 'black':
                rgb255 = np.array([0, 0, 0], dtype=np.uint8)
            out[lab] = rgb255
        return out


    def _compute_label_map(self, img_uint8: np.ndarray, progress_callback=None) -> np.ndarray:
        self.fuzzy_color_space.precompute_pack()

        img01 = img_uint8.astype(np.float32) / 255.0
        lab_img = skcolor.rgb2lab(img01)
        lab_q = np.round(lab_img, 2)

        h, w = lab_q.shape[:2]
        total = h * w
        label_map = np.empty((h, w), dtype=object)

        membership_cache = {}
        processed = 0

        for y in range(h):
            for x in range(w):
                lab_color = lab_q[y, x]
                key = (int(lab_color[0] * 100), int(lab_color[1] * 100), int(lab_color[2] * 100))

                best_label = membership_cache.get(key)
                if best_label is None:
                    m = self.fuzzy_color_space.calculate_membership(lab_color)
                    best_label = max(m, key=m.get) if m else None
                    membership_cache[key] = best_label

                label_map[y, x] = best_label
                processed += 1

                if progress_callback and (processed % 5000 == 0 or processed == total):
                    progress_callback(processed, total)

        return label_map




    def show_original_image(self, window_id: str):
        try:
            win = self.image_windows[window_id]
            win["img"].set_source(win["original_source"])
            win["current_source"] = win["original_source"]

            # ocultar legend/controles
            if "legend_box" in win:
                win["legend_box"].style('display:none;')

            self.ORIGINAL_IMG[window_id] = False
            ui.notify('Original image restored')
        except Exception as e:
            self.custom_warning("Display Error", str(e))


    def color_mapping_all(self, window_id: str):
        if not getattr(self, 'COLOR_SPACE', False):
            self.custom_warning("No Color Space", "Load a color space first (.cns or .fcs).")
            return

        client = context.client  # ✅ contexto UI del usuario

        async def _run():
            # todo lo que sea UI debe ir "with client:"
            with client:
                self.show_loading("Color Mapping All...")

            try:
                img_np = self._get_work_image_np(window_id, max_side=400)

                def progress(cur, total):
                    # ✅ esto también toca UI, así que dentro de client
                    with client:
                        self.set_loading_progress(cur / total)

                if window_id not in self.label_map_cache:
                    label_map = await asyncio.to_thread(self._compute_label_map, img_np, progress)
                    self.label_map_cache[window_id] = label_map
                else:
                    label_map = self.label_map_cache[window_id]

                scheme = self.scheme_cache.get(window_id, 'centroid')
                colors = self._label_colors_centroid() if scheme == 'centroid' else self._label_colors_hsv()

                h, w = label_map.shape
                out = np.zeros((h, w, 3), dtype=np.uint8)

                for label, rgb in colors.items():
                    out[label_map == label] = rgb
                out[label_map == None] = np.array([0, 0, 0], dtype=np.uint8)

                self.modified_image[window_id] = out
                data_url = self._np_to_data_url(out)

                with client:
                    win = self.image_windows[window_id]
                    win["img"].set_source(data_url)
                    win["current_source"] = data_url
                    self._render_legend(window_id)
                    ui.notify('Color mapping applied (preview)')

            except Exception as e:
                with client:
                    self.custom_warning("Processing Error", str(e))

            finally:
                with client:
                    self.hide_loading()

        asyncio.create_task(_run())




    def _get_work_image_np(self, window_id: str, max_side: int = 400) -> np.ndarray:
        path = self.image_windows[window_id]["path"]
        img = Image.open(path).convert('RGB')

        w, h = img.size
        if max(w, h) > max_side:
            scale = max_side / float(max(w, h))
            img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)

        return np.array(img, dtype=np.uint8)



    def _render_legend(self, window_id: str, only_labels=None, info: str | None = None, mode: str = 'all'):
        win = self.image_windows[window_id]

        legend_box = win.get("legend_box")
        legend_scroll = win.get("legend_scroll")
        legend_info = win.get("legend_info")
        alt_btn = win.get("alt_colors_btn")

        if legend_box is None or legend_scroll is None or legend_info is None:
            return

        # mostrar panel legend siempre que llamemos
        legend_box.style('display:block;')

        # show/hide alt button depending on mode
        if alt_btn is not None:
            alt_btn.set_visibility(mode == 'all')

        scheme = self.scheme_cache.get(window_id, 'centroid')
        colors = self._label_colors_centroid() if scheme == 'centroid' else self._label_colors_hsv()

        labels = only_labels if only_labels is not None else list(getattr(self, 'color_matrix', []) or [])

        legend_scroll.clear()
        with legend_scroll:
            for lab in labels:
                rgb = colors.get(lab, np.array([0, 0, 0], dtype=np.uint8))
                hexcol = f'#{int(rgb[0]):02x}{int(rgb[1]):02x}{int(rgb[2]):02x}'
                with ui.row().classes('items-center gap-2'):
                    ui.html(
                        f'<div style="width:18px;height:18px;border:1px solid #000;background:{hexcol};border-radius:3px;"></div>',
                        sanitize=False,
                    )
                    ui.label(lab).classes('text-sm')

        legend_info.set_text(info or '')




    def toggle_color_scheme(self, window_id: str):
        current = self.scheme_cache.get(window_id, 'centroid')
        self.scheme_cache[window_id] = 'hsv' if current == 'centroid' else 'centroid'

        # si ya hay label_map, recolorea sin recalcular
        if window_id in self.label_map_cache:
            # reaplica mapping con el nuevo esquema
            self.color_mapping_all(window_id)
        else:
            self._render_legend(window_id)




    def _proto_index_by_label(self, label: str) -> int:
        for i, p in enumerate(self.prototypes):
            if getattr(p, 'label', None) == label:
                return i
        raise ValueError(f'Prototype not found: {label}')

    def _membership_map_for_prototype(self, img_np_rgb255: np.ndarray, proto_index: int, progress_cb=None) -> np.ndarray:
        """
        Returns grayscale membership map uint8 for ONE prototype.
        img_np_rgb255: HxWx3 uint8 (0..255)
        """
        img = img_np_rgb255.astype(np.float32) / 255.0
        lab = skcolor.rgb2lab(img)
        lab_q = np.round(lab, 2)  # 0.01 precision

        h, w, _ = lab_q.shape
        flat = lab_q.reshape(-1, 3)

        total = flat.shape[0]
        out = np.empty(total, dtype=np.float32)

        cache = {}
        fcs = self.fuzzy_color_space  # alias

        for i in range(total):
            L, A, B = flat[i]
            key = (float(L), float(A), float(B))
            val = cache.get(key)
            if val is None:
                val = fcs.calculate_membership_for_prototype(np.array([L, A, B], dtype=float), proto_index)
                cache[key] = val
            out[i] = val

            if progress_cb and (i % 5000 == 0 or i == total - 1):
                progress_cb(i + 1, total)

        gray = (out.reshape(h, w) * 255.0)
        gray = np.clip(gray, 0, 255).astype(np.uint8)
        return gray










    def _compose_with_legend(self, img_uint8: np.ndarray, window_id: str) -> Image.Image:
        """Return a PIL Image with a legend appended at the bottom."""
        img = Image.fromarray(img_uint8, mode='RGB')
        labels = list(getattr(self, 'color_matrix', []) or [])

        scheme = self.scheme_cache.get(window_id, 'centroid')
        colors = self._label_colors_centroid() if scheme == 'centroid' else self._label_colors_hsv()

        # layout legend
        pad = 12
        swatch = 18
        line_h = 24
        max_lines = max(1, len(labels))

        legend_h = pad * 2 + line_h * max_lines
        out = Image.new('RGB', (img.width, img.height + legend_h), (255, 255, 255))
        out.paste(img, (0, 0))

        draw = ImageDraw.Draw(out)

        # font (safe fallback)
        try:
            font = ImageFont.truetype("arial.ttf", 14)
        except Exception:
            font = ImageFont.load_default()

        y0 = img.height + pad
        x0 = pad

        for i, lab in enumerate(labels):
            rgb = colors.get(lab, np.array([0, 0, 0], dtype=np.uint8))
            hexcol = (int(rgb[0]), int(rgb[1]), int(rgb[2]))
            y = y0 + i * line_h

            # swatch
            draw.rectangle([x0, y + 3, x0 + swatch, y + 3 + swatch], fill=hexcol, outline=(0, 0, 0))
            # text
            draw.text((x0 + swatch + 10, y + 2), lab, fill=(0, 0, 0), font=font)

        return out

    def save_image(self):
        # Debe haber al menos 1 ventana con imagen
        wins = getattr(self, "image_windows", {})
        if not wins:
            self.custom_warning("Save Image", "There are no image windows to save.")
            return

        options = {wid: wins[wid].get("title", wid) for wid in wins.keys()}

        with ui.dialog() as d, ui.card().classes('w-[520px]'):
            ui.label('Save Image').classes('text-lg font-bold')
            ui.label('Choose which image you want to download.').classes('text-sm text-gray-600')

            sel = ui.select(
                options=options,
                value=list(options.keys())[0],
                label='Image window',
            ).classes('w-full')

            include_legend = ui.checkbox('Include legend (if available)', value=True)
            fmt = ui.select(options=['png', 'jpg'], value='png', label='Format').classes('w-full')

            async def _download():
                wid = sel.value
                d.close()

                try:
                    # Escoge qué datos guardar:
                    # - si existe modified_image[wid] -> guardamos eso
                    # - si no, guardamos la original
                    if hasattr(self, 'modified_image') and wid in self.modified_image:
                        arr = self.modified_image[wid]
                        pil = Image.fromarray(arr.astype(np.uint8), mode='RGB')
                    else:
                        # original
                        path = self.image_windows[wid]["path"]
                        pil = Image.open(path).convert('RGB')

                    # ¿leyenda?
                    if include_legend.value and hasattr(self, 'label_map_cache') and wid in self.label_map_cache:
                        # si hemos hecho mapping, entonces sí tiene sentido legend
                        pil = self._compose_with_legend(np.array(pil, dtype=np.uint8), wid)

                    # encode
                    buf = io.BytesIO()
                    if fmt.value == 'jpg':
                        pil.save(buf, format='JPEG', quality=95)
                        mime = 'image/jpeg'
                        ext = 'jpg'
                    else:
                        pil.save(buf, format='PNG')
                        mime = 'image/png'
                        ext = 'png'

                    data = buf.getvalue()

                    # filename
                    base = self.image_windows[wid].get("title", wid).replace(' ', '_')
                    filename = f'{base}.{ext}'

                    # ✅ fuerza descarga en el navegador
                    ui.download(data, filename=filename, media_type=mime)
                    ui.notify(f'Downloading: {filename}')

                except Exception as e:
                    self.custom_warning("Save Error", str(e))

            with ui.row().classes('justify-end gap-2'):
                ui.button('Cancel', on_click=d.close).props('flat')
                ui.button('Download', icon='download', on_click=_download)

        d.open()




app = PyFCSWebApp()

def main():
    ui.run(
        host='0.0.0.0',
        port=int(os.environ.get('PORT', 8080)),
        title='PyFCS Web',
    )

main()
