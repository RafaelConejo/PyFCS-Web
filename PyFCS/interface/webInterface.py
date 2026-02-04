import os
from nicegui import ui
import tempfile
import sys
import numpy as np
from skimage import color as skcolor


### current path ###
current_dir = os.path.dirname(__file__)
pyfcs_dir = os.path.abspath(os.path.join(current_dir, '..', '..'))

# Add the PyFCS path to sys.path
sys.path.append(pyfcs_dir)

from PyFCS import Input, VisualManager, ReferenceDomain, FuzzyColorSpace, ImageManager, FuzzyColorSpaceManager, ColorEvaluationManager
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
                    self.image_view = ui.image('').classes('w-full h-[calc(100%-32px)] object-contain bg-white')

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
                                        ui.label('3D plot placeholder').classes('text-gray-500')

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
                self.loading_progress = ui.linear_progress(value=0).props('instant-feedback')
        else:
            self.loading_label.set_text(message)
            self.loading_progress.set_value(0)

        self.loading_dialog.open()

    def show_loading_color_space(self):
        self.show_loading("Loading Color Space...")

    def set_loading_progress(self, value_0_to_1: float):
        if self.loading_progress is not None:
            self.loading_progress.set_value(max(0.0, min(1.0, value_0_to_1)))

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
        """Web: upload .cns/.fcs -> temp file -> reuse FuzzyColorSpaceManager.load_color_file(path)."""
        with ui.dialog() as d, ui.card().classes('w-[520px]'):
            ui.label('Load Color Space').classes('text-lg font-bold')
            ui.label('Upload a .cns or .fcs file.')

            ui.upload(
                label='Choose file',
                multiple=False,
                auto_upload=True,
                on_upload=lambda e: self._on_color_file_uploaded(e, d),
            ).props('accept=.cns,.fcs')

            with ui.row().classes('justify-end'):
                ui.button('Cancel', on_click=d.close).props('flat')
        d.open()



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





app = PyFCSWebApp()

def main():
    ui.run(
        host='0.0.0.0',
        port=int(os.environ.get('PORT', 8080)),
        title='PyFCS Web',
    )

main()
