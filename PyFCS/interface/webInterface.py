import os
import io
import sys
import base64
import asyncio
import tempfile
import mimetypes
import numpy as np
from nicegui import ui
from nicegui import context
import matplotlib.pyplot as plt
from skimage import color as skcolor
from PIL import Image, ImageDraw, ImageFont

### current path ###
current_dir = os.path.dirname(__file__)
pyfcs_dir = os.path.abspath(os.path.join(current_dir, '..', '..'))

# Add the PyFCS path to sys.path
sys.path.append(pyfcs_dir)

from PyFCS import Input, VisualManager, ReferenceDomain, FuzzyColorSpace, FuzzyColorSpaceManager
import PyFCS.interface.modules.UtilsTools as UtilsTools


"""
PyFCSWebApp (Lite Web Demo)

This is the **lite web version of PyFCS**, created as a browser-based proof-of-concept / demo.
It reproduces the core PyFCS interface workflows in a web UI (NiceGUI), focusing on:
- Image workspace handling (open/save/close placeholders here)
- Fuzzy Color Space management (create/load placeholders here)
- 3D model view options and color selection (checkbox-driven filtering)
- A data tab to display color LAB values and a visual swatch

This file mainly defines the web layout, UI components, and the state management
needed to connect GUI actions (checkboxes, menu items, options) with the backend logic.
"""

class PyFCSWebApp:
    def __init__(self):
        # Flag indicating whether a color space is currently loaded/active
        self.COLOR_SPACE = False

        # 3D model display options (what to plot / show)
        self.model_3d_options = {
            "Representative": True,
            "Core": False,
            "0.5-cut": False,
            "Support": False,
        }

        # GUI state: color name -> checkbox reference (used for select/deselect all)
        self.color_checkboxes = {}

        # Per-image cached data / state
        self.modified_image = {}           # window_id -> np.uint8(H,W,3) last recolored image
        self.label_map_cache = {}          # window_id -> np.array(H,W) of strings (labels)
        self.scheme_cache = {}             # window_id -> 'centroid' | 'hsv'
        self.mapping_all_cache = {}        # (window_id, scheme, max_side) -> data_url

        # Prototype-specific caches (used for prototype overlays / membership maps)
        self.proto_map_cache = {}          # (window_id, proto_label, max_side, scheme) -> (data_url, pct_text)
        self.proto_membership_cache = {}   # (window_id, proto_label, max_side) -> membership_uint8

        # --- State per image / per color selection ---
        self.MEMBERDEGREE = {}             # colors: color_name -> bool (enabled/disabled)
        self.MEMBERDEGREE_IMG = {}         # images: window_id -> bool
        self.ORIGINAL_IMG = {}             # images: window_id -> bool (show original vs modified)

        # Loading UI references (dialog + widgets)
        self.loading_dialog = None
        self.loading_label = None
        self.loading_progress = None

        # Managers / domain limits used by other parts of the application
        self.fuzzy_manager = FuzzyColorSpaceManager()  # root=None in web context
        self.volume_limits = ReferenceDomain(0, 100, -128, 127, -128, 127)

        # Build the UI layout immediately
        self.build_layout()

    def build_layout(self):
        """
        Create the NiceGUI layout:
        - Header with menus (File/Image Manager/Fuzzy Color Space/About)
        - Toolbar cards (Image Manager + Fuzzy Color Space Manager)
        - Main splitter:
            LEFT  -> Image workspace
            RIGHT -> Tabs (Model 3D / Data)
        Also injects custom HTML/CSS/JS for draggable windows and pixel picking.
        """
        ui.page_title('PyFCS Interface (Web)')

        # Set app theme colors
        ui.colors(primary='#8b5cf6')

        # Slight CSS tweak (kept for consistent scaling / layout control)
        ui.add_head_html('''
        <style>
        body {
            transform: scale(1);
            transform-origin: top left;
            width: 100%;
        }
        </style>
        ''')

        # JavaScript: draggable elements with z-index stacking (window-like behavior)
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

            // Ensure absolute positioning for drag
            el.style.position = 'absolute';
            el.style.willChange = 'left, top';

            // Prevent scroll/touch gestures while dragging
            handle.style.cursor = 'move';
            handle.style.userSelect = 'none';
            handle.style.touchAction = 'none';

            let dragging = false;
            let startX = 0, startY = 0;
            let startLeft = 0, startTop = 0;

            // RAF for smooth updates
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
                // Only left mouse button and only from the handle
                if (e.button !== 0) return;

                dragging = true;
                bringToFront(el);

                // offsetLeft/Top avoids weird jumps from bounding rect/scroll
                startLeft = el.offsetLeft;
                startTop  = el.offsetTop;

                startX = e.clientX;
                startY = e.clientY;

                // Capture pointer to keep dragging even if leaving handle area
                try { el.setPointerCapture(e.pointerId); } catch {}

                // Listen globally for smoother behavior
                window.addEventListener('pointermove', onMove, true);
                window.addEventListener('pointerup', onUp, true);

                e.preventDefault();
                e.stopPropagation();
            });

            // Clicking the window brings it to front (without moving it)
            el.addEventListener('mousedown', () => bringToFront(el), {passive: true});
            }
            </script>
        ''')

        # JavaScript: register a pixel picker on an <img> element and emit a "pick" event to backend
        ui.add_head_html('''
        <script>
            window.registerPixelPicker = function(imgId){
            const img = document.getElementById(imgId);
            if (!img) return;

            // Avoid duplicating listeners if the image is reloaded
            img._pickerRegistered = true;

            const canvas = document.createElement('canvas');
            const ctx = canvas.getContext('2d', { willReadFrequently: true });

            function syncCanvas() {
                if (!img.naturalWidth || !img.naturalHeight) return;
                canvas.width = img.naturalWidth;
                canvas.height = img.naturalHeight;
                ctx.drawImage(img, 0, 0);
            }

            if (img.complete) syncCanvas();
            img.addEventListener('load', syncCanvas);

            img.style.cursor = 'crosshair';

            img.onclick = (ev) => {
                // Ensure the canvas matches the current image
                syncCanvas();

                const rect = img.getBoundingClientRect();
                const x = Math.floor((ev.clientX - rect.left) * img.naturalWidth / rect.width);
                const y = Math.floor((ev.clientY - rect.top) * img.naturalHeight / rect.height);

                const data = ctx.getImageData(x, y, 1, 1).data;
                const payload = { r: data[0], g: data[1], b: data[2], x, y };

                // NiceGUI helper used to emit events to the backend
                emitEvent(img, 'pick', payload);
            };
            }
        </script>
        ''')

        # ---- Header / Menus ----
        with ui.header(elevated=True).classes('items-center'):
            ui.label('PyFCS Interface').classes('text-lg font-bold')

            with ui.row().classes('gap-2'):
                # File menu (web "exit" just shows a dialog)
                with ui.menu():
                    ui.menu_item('Exit', on_click=self.exit_app)
                    ui.button('File', icon='menu').props('flat')

                # Image manager menu
                with ui.menu():
                    ui.menu_item('Open Image', on_click=self.open_image)
                    ui.menu_item('Save Image', on_click=self.save_image)
                    ui.menu_item('Close All', on_click=self.close_all_image)
                    ui.button('Image Manager', icon='image').props('flat')

                # Fuzzy color space menu
                with ui.menu():
                    ui.menu_item('New Color Space', on_click=self.show_menu_create_fcs)
                    ui.menu_item('Load Color Space', on_click=self.load_color_space)
                    ui.button('Fuzzy Color Space', icon='palette').props('flat')

                # About action
                ui.button('About', on_click=self.about_info).props('flat')

        # ---- Toolbar cards (without "Color Evaluation" section) ----
        with ui.row().classes('w-full q-pa-md items-start gap-4'):
            # Image Manager (compact)
            with ui.card().classes('w-[350px] q-pa-md'):
                ui.label('Image Manager').classes('font-bold')
                with ui.row().classes('gap-2'):
                    ui.button('Open Image', icon='folder_open', on_click=self.open_image)
                    ui.button('Save Image', icon='save', on_click=self.save_image)

            # Fuzzy Color Space Manager (wider)
            with ui.card().classes('w-[450px] q-pa-md'):
                ui.label('Fuzzy Color Space Manager').classes('font-bold')
                with ui.row().classes('gap-2 items-center'):
                    ui.button('New Color Space', icon='add', on_click=self.show_menu_create_fcs)
                    ui.button('Load Color Space', icon='upload_file', on_click=self.load_color_space)

        # ---- Main split (full height minus header+toolbar) ----
        with ui.splitter(value=30).classes('w-full h-[calc(100vh-150px)] q-pa-md') as splitter:
            # LEFT: Image area (workspace for draggable windows / image elements)
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
                        # Top row: model options checkboxes
                        with ui.row().classes('items-center q-gutter-md q-pa-sm'):
                            for name in ["Representative", "Core", "0.5-cut", "Support"]:
                                ui.checkbox(
                                    name,
                                    value=self.model_3d_options[name],
                                    on_change=lambda e, n=name: self.set_model_option(n, e.value),
                                )

                        # Split: plot (left) and color selection list (right)
                        with ui.splitter(value=78).classes('w-full h-[calc(100%-52px)]') as inner:
                            # Center: plot container
                            with inner.before:
                                with ui.card().classes('w-full h-full'):
                                    self.plot_container = ui.column().classes('w-full h-full')
                                    with self.plot_container:
                                        ui.label('3D plot').classes('text-gray-500')

                            # Right: color list + select/deselect actions
                            with inner.after:
                                with ui.card().classes('w-full h-full'):
                                    with ui.row().classes('justify-end gap-2 q-pa-sm'):
                                        ui.button('Select All', on_click=self.select_all_color)
                                        ui.button('Deselect All', on_click=self.deselect_all_color)

                                    self.colors_scroll = ui.scroll_area().classes('w-full h-[calc(100%-56px)] q-pa-sm')
                                    with self.colors_scroll:
                                        # Initialize with an empty list (populated after loading a color space)
                                        self.set_color_list([])

                    # ---- Data ----
                    with ui.tab_panel(data_tab).classes('w-full h-full'):
                        # File name field
                        with ui.card().classes('w-full q-pa-md'):
                            ui.label('Name:').classes('font-bold')
                            self.file_name = ui.input(placeholder='').classes('w-80').props('readonly')

                        # Table showing LAB values and a color swatch column
                        self.data_table = ui.table(
                            columns=[
                                {'name': 'name', 'label': 'Name', 'field': 'name'},
                                {'name': 'L', 'label': 'L*', 'field': 'L'},
                                {'name': 'a', 'label': 'a*', 'field': 'a'},
                                {'name': 'b', 'label': 'b*', 'field': 'b'},
                                {'name': 'color', 'label': 'Color', 'field': 'color', 'sortable': False, 'align': 'center'},
                            ],
                            rows=[],
                            row_key='name',
                        ).classes('w-full')

                        # Custom cell renderer to display the swatch as a colored rectangle
                        self.data_table.add_slot('body-cell-color', r'''
                            <q-td :props="props">
                            <div style="width:110px;height:24px;border:1px solid #000;margin:auto;border-radius:4px;"
                                :style="{ background: props.value }">
                            </div>
                            </q-td>
                        ''')

                        # Buttons for adding/applying changes are currently disabled (kept as reference)
                        # with ui.row().classes('q-pa-md gap-2'):
                        #     ui.button('Add New Color', on_click=self.addColor_data_window)
                        #     ui.button('Apply Changes', on_click=self.apply_changes)

    # ---------- UI helpers ----------
    def set_color_list(self, color_names: list[str]) -> None:
        """
        (Re)create the checkbox list shown in the right panel and refresh plot state on change.

        This method:
        - Clears any previous checkbox references
        - Ensures MEMBERDEGREE exists as a dict
        - Creates one checkbox per color
        - Stores both the UI checkbox and the persisted enabled/disabled value
        """
        self.color_checkboxes.clear()
        self.colors_scroll.clear()

        # Ensure MEMBERDEGREE exists and is a dictionary
        if not hasattr(self, 'MEMBERDEGREE') or not isinstance(self.MEMBERDEGREE, dict):
            self.MEMBERDEGREE = {}

        with self.colors_scroll:
            for name in color_names:
                # Initial value: previously saved value, otherwise True by default
                initial = self.MEMBERDEGREE.get(name, True)

                cb = ui.checkbox(
                    name,
                    value=initial,
                    on_change=lambda e, n=name: self.on_color_toggle(n, e.value),
                )
                self.color_checkboxes[name] = cb
                self.MEMBERDEGREE[name] = initial

    def list_preset_fcs(self) -> dict:
        """
        Return a dict mapping preset label -> absolute .fcs path.

        Presets are .fcs files shipped with the app, stored under:
        ../.. / fuzzy_color_spaces (relative to current_dir)
        """
        presets_dir = os.path.abspath(os.path.join(current_dir, '..', '..', 'fuzzy_color_spaces'))
        if not os.path.isdir(presets_dir):
            return {}

        files = sorted([f for f in os.listdir(presets_dir) if f.lower().endswith('.fcs')])
        return {os.path.splitext(f)[0]: os.path.join(presets_dir, f) for f in files}

    def on_color_toggle(self, name: str, value: bool):
        """
        Checkbox callback for enabling/disabling a color.

        Steps:
        - Persist checkbox state into MEMBERDEGREE
        - Update selected sets used by plotting/processing
        - Trigger a refresh (on_option_select)
        """
        # Persist state
        self.MEMBERDEGREE[name] = value

        # Recompute "selected_*" collections (based on current enabled colors)
        self.update_selected_sets_from_checks()

        # Refresh plot / UI state
        self.on_option_select()

    def update_selected_sets_from_checks(self):
        """
        Update the internal selected_* collections based on MEMBERDEGREE.

        - selected_centroids: dict filtered from self.color_data by enabled colors
        - selected_hex_color: kept as-is (not filtered by default)
        - selected_alpha/core/support: lists of Prototype objects filtered by .label
        """
        if not hasattr(self, 'color_data') or not self.color_data:
            return

        # Enabled color labels
        enabled = {k for k, v in self.MEMBERDEGREE.items() if v}

        # Centroids are stored in dict form (like the loaded FCS data structure)
        self.selected_centroids = {k: v for k, v in self.color_data.items() if k in enabled}

        # Hex mapping is kept unchanged (if strict filtering is needed, it can be added later)
        self.selected_hex_color = getattr(self, 'hex_color', {})

        # Helper to filter Prototype lists by their label
        def _filter_by_label(protos):
            if not protos:
                return protos
            return [p for p in protos if getattr(p, 'label', None) in enabled]

        self.selected_alpha = _filter_by_label(getattr(self, 'prototypes', None))
        self.selected_core = _filter_by_label(getattr(self, 'cores', None))
        self.selected_support = _filter_by_label(getattr(self, 'supports', None))

    # ---------- Web equivalents of your Tkinter utils ----------
    def custom_warning(self, title="Warning", message="Warning"):
        """Show a simple modal warning dialog."""
        with ui.dialog() as d, ui.card():
            ui.label(title).classes('text-lg font-bold')
            ui.label(message).classes('text-gray-700')
            with ui.row().classes('justify-end'):
                ui.button('OK', on_click=d.close)
        d.open()

    def show_loading(self, message="Processing..."):
        """
        Show (or reuse) a loading dialog with a spinner and progress bar.
        """
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
        """Convenience wrapper for a standardized loading message."""
        self.show_loading("Loading Color Space...")

    def set_loading_progress(self, value_0_to_1: float):
        """
        Update the loading progress bar value (0..1).
        Disables indeterminate mode once explicit values are provided.
        """
        if self.loading_progress is not None:
            self.loading_progress.props(remove='indeterminate')
            self.loading_progress.set_value(round(max(0.0, min(1.0, value_0_to_1)), 2))

    def hide_loading(self):
        """Close the loading dialog if it exists."""
        if self.loading_dialog is not None:
            self.loading_dialog.close()

    # ---------- callbacks ----------
    def exit_app(self):
        """
        "Exit" action in web context:
        you cannot stop the server from the browser, so we show an instruction dialog.
        """
        with ui.dialog() as d, ui.card():
            ui.label('Exit').classes('text-lg font-bold')
            ui.label('Close this tab to exit the application.')
            with ui.row().classes('justify-end'):
                ui.button('OK', on_click=d.close)
        d.open()

    def about_info(self):
        """Show an 'About' dialog with basic application information."""
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
        """
        Entry point for creating a new fuzzy color space in the web UI.
        Current behavior: directly opens palette-based creation (stub).
        """
        self.palette_based_creation()

        # Alternative dialog-based selector (kept commented as reference)
        # with ui.dialog() as d, ui.card():
        #     ui.label('Create New Color Space').classes('text-lg font-bold')
        #     ui.label('Choose a creation mode:')
        #     with ui.row().classes('gap-2'):
        #         ui.button('Palette-Based', on_click=lambda: (d.close(), self.palette_based_creation()))
        #         ui.button('Image-Based', on_click=lambda: (d.close(), self.image_based_creation()))
        #     with ui.row().classes('justify-end'):
        #         ui.button('Cancel', on_click=d.close).props('flat')
        # d.open()

    # ----- still-stubs (to be implemented later) -----
    def open_image(self):
        """Open Image action (placeholder)."""
        ui.notify('Open Image (stub)')

    def save_image(self):
        """Save Image action (placeholder)."""
        ui.notify('Save Image (stub)')

    def close_all_image(self):
        """Close all images action (placeholder)."""
        ui.notify('Close All (stub)')

    def load_color_space(self):
        """Load fuzzy color space action (placeholder)."""
        ui.notify('Load Color Space (stub)')

    def palette_based_creation(self):
        """Palette-based color space creation (placeholder)."""
        ui.notify('Palette Based (stub)')

    def image_based_creation(self):
        """Image-based color space creation (placeholder)."""
        ui.notify('Image Based (stub)')

    def addColor_data_window(self):
        """Add new color dialog (placeholder)."""
        ui.notify('Add New Color (stub)')

    def apply_changes(self):
        """Apply changes action (placeholder)."""
        ui.notify('Apply Changes (stub)')

    def open_interactive_figure(self):
        """Open interactive figure action (placeholder)."""
        ui.notify('Interactive Figure (stub)')

    def select_all_color(self):
        """Set all color checkboxes to True (enabled)."""
        for cb in self.color_checkboxes.values():
            cb.set_value(True)

    def deselect_all_color(self):
        """Set all color checkboxes to False (disabled)."""
        for cb in self.color_checkboxes.values():
            cb.set_value(False)

    def set_model_option(self, name, value):
        """
        Update one of the 3D model option toggles and refresh the plot/UI.

        Parameters
        ----------
        name : str
            Option key (Representative/Core/0.5-cut/Support).
        value : bool
            New checkbox value.
        """
        self.model_3d_options[name] = value
        self.on_option_select()














    # ---------------------------------------------------------------------
    # LOAD FCS
    # ---------------------------------------------------------------------
    def load_color_space(self):
        """
        Open a modal dialog that lets the user load a color space in two ways:
        1) Load a preset .fcs file already available on the server.
        2) Upload a local .cns or .fcs file from the user's computer.

        This is one of the main features of the PyFCS Web Lite demo:
        it connects the web UI to the existing PyFCS parsers and then triggers
        the 3D visualization / data table refresh.
        """
        # Get available preset FCS files shipped with the web app
        presets = self.list_preset_fcs()

        # Build the loading dialog
        with ui.dialog() as d, ui.card().classes('w-[560px]'):
            ui.label('Load Color Space').classes('text-lg font-bold')

            # --- Presets section ---
            if presets:
                ui.label('Load a preset (.fcs) from the server:').classes('text-sm text-gray-700')

                # Dropdown to pick which preset to load
                preset_select = ui.select(
                    options=list(presets.keys()),
                    value=list(presets.keys())[0],
                    label='Presets',
                ).classes('w-full')

                # Action buttons
                with ui.row().classes('justify-end gap-2'):
                    ui.button(
                        'Load preset',
                        icon='cloud_download',
                        # Close dialog and load the selected preset path
                        on_click=lambda: (d.close(), self.load_color_space_from_path(presets[preset_select.value]))
                    )
            else:
                ui.label('No presets found on the server.').classes('text-sm text-gray-500')

            ui.separator()

            # --- Upload section ---
            ui.label('Or upload a .cns/.fcs file:').classes('text-sm text-gray-700')

            # NiceGUI upload will immediately upload and call the handler
            ui.upload(
                label='Choose file',
                multiple=False,
                auto_upload=True,
                on_upload=lambda e: self._on_color_file_uploaded(e, d),
            ).props('accept=.cns,.fcs')

            # Cancel closes the dialog without doing anything
            with ui.row().classes('justify-end'):
                ui.button('Cancel', on_click=d.close).props('flat')

        d.open()

    def load_color_space_from_path(self, filepath: str):
        """
        Load a color space file from a server-side path (preset).

        This method:
        - Shows a loading dialog
        - Uses FuzzyColorSpaceManager.load_color_file() to parse the file
        - Updates UI state: file name, data table, and 3D model plot
        - For .cns: constructs the fuzzy color space (volumes) on the fly
        - For .fcs: loads the full fuzzy color space including prototypes/cores/supports
        """
        self.show_loading_color_space()
        try:
            # Parse file using the existing PyFCS input system
            data = self.fuzzy_manager.load_color_file(filepath)

            # Store reference info for UI / plots
            self.file_path = filepath
            self.file_base_name = os.path.splitext(os.path.basename(filepath))[0]
            self.file_name.set_value(self.file_base_name)

            # CNS: only raw color data -> compute volumes and then update UI
            if data['type'] == 'cns':
                self.color_data = data['color_data']
                self.display_data_window()
                self.update_volumes()

            # FCS: full fuzzy color space -> use precomputed structures
            elif data['type'] == 'fcs':
                self.color_data = data['color_data']
                self.fuzzy_color_space = data['fuzzy_color_space']

                # Extract structures needed by the 3D visualizations
                self.cores = self.fuzzy_color_space.cores
                self.supports = self.fuzzy_color_space.supports
                self.prototypes = self.fuzzy_color_space.prototypes

                # Precompute cached geometry / membership structures (as in desktop)
                self.fuzzy_color_space.precompute_pack()

                # Refresh UI
                self.display_data_window()
                self.update_prototypes_info()

            ui.notify('Loaded preset successfully')

        except Exception as ex:
            # Any parsing/IO errors are displayed to the user
            self.custom_warning('File Error', str(ex))
        finally:
            # Always close the loading dialog
            self.hide_loading()

    async def _on_color_file_uploaded(self, e, dialog):
        """
        NiceGUI upload callback (client -> server).

        Workflow:
        - Close the selection dialog
        - Read uploaded file bytes from the browser (await e.file.read())
        - Save to a temporary file (so existing PyFCS parsers can be reused unchanged)
        - Parse using FuzzyColorSpaceManager.load_color_file()
        - Update app state and refresh the 3D plot + data table
        - Remove the temporary file at the end
        """
        # Close the upload dialog immediately
        dialog.close()
        self.show_loading_color_space()

        tmp_path = None
        try:
            # Original name is shown in UI, but we parse from a temporary file
            original_name = e.file.name

            # IMPORTANT: read file content as bytes from the browser upload
            content_bytes = await e.file.read()

            # Keep the original extension so the parser can detect type
            suffix = os.path.splitext(original_name)[1].lower() or '.tmp'
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(content_bytes)
                tmp_path = tmp.name

            # Reuse the same parser logic as desktop
            data = self.fuzzy_manager.load_color_file(tmp_path)

            # Update state used by plots/tables
            self.file_path = original_name
            self.file_base_name = os.path.splitext(os.path.basename(original_name))[0]
            self.file_name.set_value(self.file_base_name)

            # CNS: compute volumes from prototypes
            if data['type'] == 'cns':
                self.color_data = data['color_data']
                self.display_data_window()
                self.update_volumes()

            # FCS: load full fuzzy color space and precompute pack
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
            # Ensure loading dialog is closed and temp file is removed
            self.hide_loading()
            if tmp_path:
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

    def update_volumes(self):
        """
        Compute fuzzy color space volumes from raw color data (.cns).

        This mirrors the desktop workflow:
        - Convert parsed color_data into Prototype objects
        - Build a FuzzyColorSpace instance
        - Run precompute_pack() to prepare internal structures
        - Compute cores and supports
        - Update selection state and refresh the plot
        """
        self.show_loading("Computing volumes...")

        # Same calls used in the desktop application
        self.prototypes = UtilsTools.process_prototypes(self.color_data)

        # Build fuzzy color space and precompute
        self.fuzzy_color_space = FuzzyColorSpace(space_name=" ", prototypes=self.prototypes)
        self.fuzzy_color_space.precompute_pack()

        # In desktop, cores/supports are computed using getter methods
        self.cores = self.fuzzy_color_space.get_cores()
        self.supports = self.fuzzy_color_space.get_supports()

        # Refresh selection state + plots
        self.update_prototypes_info()
        self.hide_loading()

    def update_prototypes_info(self):
        """
        Update internal state after a color space is loaded/updated.

        This method:
        - Marks that a color space exists (COLOR_SPACE = True)
        - Ensures MEMBERDEGREE exists
        - Initializes default selection sets (centroids, prototypes, cores, supports)
        - Applies checkbox filtering (update_selected_sets_from_checks)
        - Triggers a plot refresh (on_option_select)
        """
        self.COLOR_SPACE = True

        # Ensure MEMBERDEGREE exists
        if not hasattr(self, 'MEMBERDEGREE') or not isinstance(self.MEMBERDEGREE, dict):
            self.MEMBERDEGREE = {}

        # In the desktop version MEMBERDEGREE is initialized from a color matrix.
        # Here we default everything to enabled when available.
        if hasattr(self, 'color_matrix') and self.color_matrix:
            self.MEMBERDEGREE = {name: True for name in self.color_matrix}

        # Default selected sets = everything
        self.selected_centroids = self.color_data
        self.selected_hex_color = getattr(self, 'hex_color', {})
        self.selected_alpha = self.prototypes
        self.selected_core = self.cores
        self.selected_support = self.supports

        # Apply current checkbox states
        self.update_selected_sets_from_checks()

        # Redraw plot with current options
        self.on_option_select()















    # ---------------------------------------------------------------------
    # Tab: "Model 3D"
    # ---------------------------------------------------------------------
    def on_option_select(self):
        """
        Rebuild the 3D figure whenever:
        - The user toggles a 3D option (Representative/Core/0.5-cut/Support)
        - The user enables/disables colors via checkboxes

        If no options are selected, the plot area is cleared.
        Otherwise, it calls VisualManager.plot_more_combined_3D() and renders it via ui.plotly().
        """
        if not self.COLOR_SPACE:
            return

        # Which layers the user wants to display
        selected_options = [k for k, v in self.model_3d_options.items() if v]

        # If nothing selected -> clear plot
        if not selected_options:
            self.draw_model_3D(None, selected_options)
            return

        try:
            # Generate the Plotly figure (same VisualManager logic as desktop)
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

            # Render figure into the UI
            self.draw_model_3D(fig, selected_options)

        except Exception as ex:
            # Plotly/geometry errors are shown as a dialog
            self.custom_warning("Plot Error", str(ex))

    def draw_model_3D(self, fig, selected_options):
        """
        Render the given Plotly figure in the right-side plot container.

        Parameters
        ----------
        fig : plotly.graph_objects.Figure or None
            Figure to display. If None, a placeholder label is shown.
        selected_options : list
            Currently selected model layers (not used directly here, but kept for clarity).
        """
        self.plot_container.clear()
        with self.plot_container:
            if fig is None:
                ui.label('No 3D option selected').classes('text-gray-500')
            else:
                ui.plotly(fig).classes('w-full h-full')













    # ---------------------------------------------------------------------
    # Tab: "DATA"
    # ---------------------------------------------------------------------
    def display_data_window(self):
        """
        Build/refresh the **Data** tab in the PyFCS Web Lite UI.

        WEB equivalent of the desktop `display_data_window()`:
        - Updates the file name input (top of the Data tab).
        - Populates the Data table with one row per color: Name + (L*, a*, b*) + color swatch.
        - Rebuilds helper structures used across the app:
            * self.hex_color   : dict mapping HEX string -> LAB numpy array
            * self.color_matrix: list of color names (used to create the checkbox list)
        - Refreshes the right-side color checkbox panel to match the loaded colors.

        Notes:
        - LAB values are taken from `positive_prototype` when available, otherwise from `Color`.
        - LAB -> RGB conversion may produce NaNs for invalid LAB; in that case a black fallback is used.
        """
        # 1) Fill the Name field (if the base name exists)
        if hasattr(self, 'file_base_name') and self.file_base_name:
            self.file_name.set_value(self.file_base_name)

        # 2) Reset containers (mirrors desktop behavior)
        self.hex_color = {}      # HEX -> LAB
        self.color_matrix = []   # ordered list of color names (for checkboxes)

        rows = []

        # 3) Iterate over the loaded color_data dictionary:
        #    self.color_data = { color_name: {'positive_prototype': [L,a,b], ...}, ... }
        for color_name, color_value in (self.color_data or {}).items():

            # Prefer 'positive_prototype', but keep compatibility with alternative key 'Color'
            lab = color_value.get('positive_prototype', None)
            if lab is None:
                lab = color_value.get('Color', None)

            # Normalize to a 1D (3,) float array
            lab = np.array(lab, dtype=float).reshape(3,)
            self.color_matrix.append(color_name)

            # Convert LAB -> RGB in [0, 1] (skimage expects shape (1,1,3))
            rgb01 = skcolor.lab2rgb(lab.reshape(1, 1, 3))[0, 0, :]

            # If conversion produced invalid values (e.g., NaNs), fall back to black
            if not np.all(np.isfinite(rgb01)):
                rgb01 = np.array([0.0, 0.0, 0.0])

            # Clip to valid range and convert to 0..255 (use rounding, not truncation)
            rgb01 = np.clip(rgb01, 0.0, 1.0)
            rgb255 = tuple(int(round(c * 255.0)) for c in rgb01)

            # Build HEX representation for UI swatches
            hex_color = f'#{rgb255[0]:02x}{rgb255[1]:02x}{rgb255[2]:02x}'

            # Store mapping used by plotting / other modules (HEX -> LAB)
            self.hex_color[hex_color] = lab

            # Optional HTML preview (kept here as reference; the table uses the HEX value slot renderer)
            preview = f'''
            <div style="
                width:110px;height:24px;border:1px solid #000;
                background:{hex_color};margin:auto;border-radius:4px;">
            </div>
            '''

            # Add a row to the Data table
            rows.append({
                'L': round(float(lab[0]), 2),
                'a': round(float(lab[1]), 2),
                'b': round(float(lab[2]), 2),
                'name': color_name,
                'color': hex_color,   # IMPORTANT: keep only the HEX string (slot renders the swatch)
            })

        # 4) Push rows into the NiceGUI table and refresh
        self.data_table.rows = rows
        self.data_table.update()

        # 5) Update the right-side checkbox list to match the loaded colors
        self.set_color_list(self.color_matrix)













    # ---------------------------------------------------------------------
    # CREATE FCS
    # ---------------------------------------------------------------------
    def _palette_toggle(self, name: str, value: bool):
        """
        Update the selection state for a palette color.

        Parameters
        ----------
        name : str
            Color name (key in self.color_checks).
        value : bool
            New checkbox state (True = selected, False = not selected).
        """
        # Keep selection state in self.color_checks (so the UI can be re-rendered)
        if hasattr(self, 'color_checks') and name in self.color_checks:
            self.color_checks[name]["value"] = value

    def palette_based_creation(self):
        """
        Open the **Palette-Based Creation** dialog (web version).

        This is the main workflow used to create a new fuzzy color space from a set of colors:
        - Loads a predefined palette (.cns) shipped with the web demo (ISCC_NBS_BASIC.cns).
        - Builds an internal dictionary (self.color_checks) that stores:
            * value: checkbox state (selected or not)
            * lab  : LAB representation
            * rgb  : RGB preview color
        - Renders a scrollable list where each row shows:
            * color swatch
            * color name
            * LAB values
            * a checkbox to select the color
        - Provides buttons to:
            * Pick colors from a loaded image (pixel picker)
            * Add a new color manually
            * Create/export the resulting .fcs file
        """
        # Palette source (.cns) shipped with the project
        color_space_path = os.path.join(pyfcs_dir, 'fuzzy_color_spaces', 'cns', 'ISCC_NBS_BASIC.cns')

        # Load palette colors as {name: {"lab": ..., "rgb": ...}}
        colors = UtilsTools.load_color_data(color_space_path)

        # Persist palette colors (used when adding new colors or avoiding collisions)
        self.palette_colors = colors

        # Dictionary used by the UI list: name -> {"value": bool, "lab": ..., "rgb": ...}
        self.color_checks = {}

        # Initial state: all colors start unchecked (nothing selected by default)
        for color_name, data in colors.items():
            self.color_checks[color_name] = {
                "value": False,
                "lab": data.get("lab"),
                "rgb": data.get("rgb"),
            }

        # Build the dialog UI
        with ui.dialog() as d, ui.card().classes('w-[560px] h-[680px]'):
            ui.label('Select colors for your Color Space').classes('text-lg font-bold')

            # Keep references so we can repaint the list dynamically
            self.palette_dialog = d
            self.palette_scroll = ui.scroll_area().classes('w-full h-[560px] border rounded q-pa-sm')
            with self.palette_scroll:
                self.palette_list_container = ui.column().classes('w-full gap-1')

            # Paint initial list
            self.render_palette_list()

            ui.separator().classes('my-2')

            # Action buttons
            with ui.row().classes('w-full justify-center items-center gap-3 pt-2'):
                ui.button('Pick from Image', icon='colorize', on_click=self.open_palette_image_picker).props('unelevated')
                ui.button('Add New Color', icon='add', on_click=self.addColor_create_fcs).props('unelevated')
                ui.button('Create Color Space', icon='save', on_click=self.create_color_space).props('unelevated')
                ui.button('Close', on_click=d.close).props('flat')

        d.open()

    def render_palette_list(self):
        """
        Render the scrollable palette list UI from `self.color_checks`.

        Each row shows:
        - A small color swatch (if RGB is known)
        - The color name
        - The LAB values formatted with 1 decimal
        - A checkbox that toggles `self.color_checks[name]["value"]`

        This method is called:
        - Initially after opening the palette dialog
        - After adding a color (manual or picked from image)
        - Whenever you want to repaint the list from state
        """
        if not hasattr(self, 'palette_list_container'):
            return

        # Clear previous UI rows before re-rendering
        self.palette_list_container.clear()

        for color_name, data in self.color_checks.items():
            lab = data.get("lab")
            rgb = data.get("rgb")
            checked = bool(data.get("value", False))

            # LAB can be stored either as dict {"L","A","B"} or as list/np.array
            if isinstance(lab, dict):
                L, A, B = float(lab["L"]), float(lab["A"]), float(lab["B"])
            else:
                arr = np.array(lab, dtype=float).reshape(3,)
                L, A, B = float(arr[0]), float(arr[1]), float(arr[2])

            # Color preview (HEX) if possible
            hexcol = None
            if rgb is not None and hasattr(UtilsTools, "rgb_to_hex"):
                hexcol = UtilsTools.rgb_to_hex(rgb)

            lab_text = f"L: {L:.1f}, a: {A:.1f}, b: {B:.1f}"

            with self.palette_list_container:
                with ui.row().classes('w-full items-center justify-between q-pa-xs border-b'):
                    # Small swatch
                    if hexcol:
                        ui.html(
                            f'<div style="width:18px;height:18px;border:1px solid #000;'
                            f'background:{hexcol};border-radius:3px;"></div>',
                            sanitize=False,
                        )
                    else:
                        ui.label('■').classes('text-gray-500')

                    # Name + LAB block
                    with ui.column().classes('gap-0'):
                        ui.label(color_name).classes('text-sm font-medium')
                        ui.label(lab_text).classes('text-xs text-gray-600')

                    # Checkbox with callback to update internal selection state
                    ui.checkbox(
                        '',
                        value=checked,
                        on_change=lambda e, n=color_name: self._palette_toggle(n, e.value),
                    )

    def open_palette_image_picker(self):
        """
        Open a dialog that lets the user pick a color from any currently loaded image.

        Requirements:
        - At least one image must be loaded into self.image_windows.

        UI layout:
        - LEFT: image selector, name input, RGB/LAB readouts, preview swatch, "Add Selected Color" button
        - RIGHT: interactive image viewer where clicking selects a pixel

        Internally this sets up:
        - self._picker_pil_full : full-resolution PIL image for pixel reads
        - self._picked_rgb/_picked_lab : last picked values
        """
        # Need loaded images to pick from
        if not hasattr(self, "image_windows") or not self.image_windows:
            self.custom_warning("No images", "Open an image first (Image Manager → Open Image).")
            return

        # Map visible title -> window_id
        options = {win.get("title", wid): wid for wid, win in self.image_windows.items()}
        first_label = next(iter(options.keys()))
        first_wid = options[first_label]

        # Picker state
        self._picker_pil_full = None
        self._picker_selected_wid = first_wid
        self._picked_rgb = None
        self._picked_lab = None
        self._picker_current_source = None  # optional tracking

        with ui.dialog() as d, ui.card().classes('w-[980px] max-w-[98vw]'):
            ui.label('Pick Color from Image').classes('text-lg font-bold')

            # Layout: left controls, right viewer
            with ui.row().classes('w-full gap-6 items-start'):
                # LEFT: controls
                with ui.column().classes('w-[320px]'):
                    ui.select(
                        options=list(options.keys()),
                        value=first_label,
                        label='Loaded Images',
                        on_change=lambda e: self._picker_load_image(options[e.value]),
                    ).classes('w-full')

                    ui.separator()
                    ui.label('Color name:').classes('text-sm text-gray-700')
                    self._picker_name_input = ui.input(placeholder='e.g. MyColor').classes('w-full')

                    ui.separator()
                    self._picker_rgb_label = ui.label('RGB: -').classes('text-sm')
                    self._picker_lab_label = ui.label('LAB: -').classes('text-sm')

                    # Small preview swatch (updated after each click)
                    self._picker_preview = ui.element('div').style(
                        'width:80px; height:26px; border:1px solid #000; '
                        'border-radius:4px; background:#000'
                    )

                    ui.button(
                        'Add Selected Color',
                        icon='add',
                        on_click=lambda: self._picker_add_selected_color(),
                    ).classes('w-full')

                    with ui.row().classes('justify-end w-full'):
                        ui.button('Close', on_click=d.close).props('flat')

                # RIGHT: viewer
                with ui.column().classes('flex-1'):
                    ui.label('Click on the image to pick a pixel').classes('text-sm text-gray-600')

                    # Container used to re-render the viewer on image changes
                    self._picker_view_container = ui.column().classes('w-full')
                    with self._picker_view_container:
                        ui.label('Select an image on the left').classes('text-gray-500')

        d.open()

        # Load first image immediately
        self._picker_load_image(first_wid)

    def _source_to_pil(self, src: str) -> Image.Image:
        """
        Convert an image source into a PIL Image.

        Supports:
        - Local filesystem paths (server-side)
        - Data URLs ("data:image/...;base64,...")
        """
        if isinstance(src, str) and src.startswith('data:image'):
            header, b64data = src.split(',', 1)
            raw = base64.b64decode(b64data)
            return Image.open(io.BytesIO(raw))
        return Image.open(src)

    def _file_to_data_url(self, path: str) -> str:
        """
        Read a local file (server-side) and convert it into a base64 data URL
        so it can be displayed in the browser.
        """
        mime, _ = mimetypes.guess_type(path)
        if not mime:
            mime = "image/png"
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        return f"data:{mime};base64,{b64}"

    def _picker_load_image(self, window_id: str):
        """
        Load the selected image into the picker viewer and prepare PIL for pixel reading.

        Steps:
        - Resolve source from image_windows[window_id] (current_source or path)
        - Convert to PIL (RGB)
        - Ensure viewer source is a data URL (if it's a server file path)
        - Render ui.interactive_image() with on_mouse callback
        """
        if window_id not in self.image_windows:
            return

        win = self.image_windows[window_id]
        src = win.get("current_source") or win.get("path")
        if not src:
            self.custom_warning("Image Error", "Selected image has no source.")
            return

        self._picker_selected_wid = window_id

        # Full-resolution PIL image for pixel sampling
        pil = self._source_to_pil(src).convert("RGB")
        self._picker_pil_full = pil

        # Convert filesystem path -> data URL for browser display
        if isinstance(src, str) and (not src.startswith("data:image")):
            if not os.path.exists(src):
                self.custom_warning("Image Error", "Selected image path not found on server.")
                return
            view_src = self._file_to_data_url(src)
        else:
            view_src = src

        # Render the interactive image
        self._picker_view_container.clear()
        with self._picker_view_container:
            self._picker_img = ui.interactive_image(
                view_src,
                on_mouse=self._picker_on_mouse,
            ).classes('w-full')

    def _picker_on_mouse(self, e):
        """
        Mouse/click handler for the interactive image.

        On click:
        - Reads the pixel at (image_x, image_y) from the PIL image
        - Converts RGB -> LAB (using UtilsTools.srgb_to_lab)
        - Updates labels and preview swatch
        """
        # We only care about click-like events
        if e.type not in ('click', 'mousedown'):
            return

        if self._picker_pil_full is None:
            return

        # NiceGUI interactive_image exposes coordinates in image space
        ix = int(getattr(e, 'image_x', -1))
        iy = int(getattr(e, 'image_y', -1))

        if ix < 0 or iy < 0:
            # Debug fallback if coordinate fields differ in your NiceGUI version
            print("EVENT:", e)
            return

        # Clamp to valid image bounds
        w, h = self._picker_pil_full.size
        ix = max(0, min(w - 1, ix))
        iy = max(0, min(h - 1, iy))

        # Read pixel (RGB)
        r, g, b = self._picker_pil_full.getpixel((ix, iy))

        # Convert to LAB for PyFCS usage
        lab = UtilsTools.srgb_to_lab(r, g, b)

        # Store last-picked values
        self._picked_rgb = (r, g, b)
        self._picked_lab = lab

        # Update UI
        self._picker_rgb_label.set_text(f"RGB: ({r}, {g}, {b})")
        self._picker_lab_label.set_text(f"LAB: ({lab[0]:.2f}, {lab[1]:.2f}, {lab[2]:.2f})")

        hex_color = f"#{r:02x}{g:02x}{b:02x}"
        self._picker_preview.style(f'background:{hex_color}')

    def _picker_add_selected_color(self):
        """
        Add the currently picked color into the palette list and repaint.

        Rules:
        - A pixel must have been picked first (RGB/LAB must exist).
        - A non-empty name is required.
        - If the name already exists, a numeric suffix is appended to avoid collisions.
        - New colors are added unchecked by default.
        """
        if self._picked_lab is None or self._picked_rgb is None:
            self.custom_warning("Pick a color", "Click on the image first to pick a pixel.")
            return

        name = (self._picker_name_input.value or "").strip()
        if not name:
            self.custom_warning("Name required", "Please enter a name for the selected color.")
            return

        # Avoid collisions by appending _2, _3, ...
        base = name
        i = 2
        while name in self.palette_colors:
            name = f"{base}_{i}"
            i += 1

        lab_dict = {
            "L": float(self._picked_lab[0]),
            "A": float(self._picked_lab[1]),
            "B": float(self._picked_lab[2]),
        }
        rgb = self._picked_rgb

        # Persist into the base palette dict (like any other palette color)
        self.palette_colors[name] = {
            "lab": lab_dict,
            "rgb": rgb,
            "source_image": self._picker_selected_wid,
        }

        # Also add into the checkbox state dict (unchecked by default)
        self.color_checks[name] = {"value": False, "lab": lab_dict, "rgb": rgb}

        # Re-render the palette list
        self.render_palette_list()

        # Clear name input for next addition
        self._picker_name_input.set_value("")
        ui.notify(f'Added color: {name}')

    def addColor_create_fcs(self):
        """
        Open a dialog to manually add a new LAB color to the palette.

        The new color:
        - Must have a unique name
        - Stores LAB as a numpy array
        - Tries to compute an RGB preview using UtilsTools.lab_to_rgb (if available)
        - Is added as selected (value=True) by default
        """
        with ui.dialog() as d, ui.card().classes('w-[420px]'):
            ui.label('Add New Color').classes('text-lg font-bold')
            name_in = ui.input(label='Color name').classes('w-full')
            l_in = ui.number(label='L*', value=50, format='%.2f').classes('w-full')
            a_in = ui.number(label='a*', value=0, format='%.2f').classes('w-full')
            b_in = ui.number(label='b*', value=0, format='%.2f').classes('w-full')

            def _add():
                # Validate name
                name = (name_in.value or '').strip()
                if not name:
                    self.custom_warning("Warning", "Color name is required.")
                    return
                if name in self.color_checks:
                    self.custom_warning("Warning", "That color name already exists.")
                    return

                # Build LAB vector
                lab = np.array([float(l_in.value), float(a_in.value), float(b_in.value)], dtype=float)

                # Optional: compute RGB for preview
                rgb = None
                if hasattr(UtilsTools, "lab_to_rgb"):
                    try:
                        rgb = UtilsTools.lab_to_rgb({'L': lab[0], 'A': lab[1], 'B': lab[2]})
                    except Exception:
                        rgb = None

                # Add to selection dict (manual colors start selected)
                self.color_checks[name] = {"value": True, "lab": lab, "rgb": rgb}

                d.close()
                self.render_palette_list()
                ui.notify('Color added')

            with ui.row().classes('justify-end gap-2'):
                ui.button('Cancel', on_click=d.close).props('flat')
                ui.button('Add', on_click=_add)

        d.open()

    def create_color_space(self):
        """
        Create a new fuzzy color space from the currently selected palette colors.

        Steps:
        - Extract all colors where self.color_checks[name]["value"] is True.
        - Normalize LAB into numpy arrays (supports dict and array forms).
        - Require at least 2 colors (otherwise creation is not meaningful).
        - Ask the user for the new color space name.
        - Call save_cs() which generates and downloads the .fcs file.
        """
        # Extract selected colors in LAB
        selected_colors_lab = {}
        for name, data in (getattr(self, 'color_checks', {}) or {}).items():
            if data.get("value", False):
                lab = data.get("lab")
                if isinstance(lab, dict):
                    selected_colors_lab[name] = np.array([lab["L"], lab["A"], lab["B"]], dtype=float)
                else:
                    selected_colors_lab[name] = np.array(lab, dtype=float)

        # Require at least two colors
        if len(selected_colors_lab) < 2:
            self.custom_warning("Warning", "At least two colors must be selected to create the Color Space.")
            return

        # Ask for the fuzzy color space name (web dialog)
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
        """
        Generate a .fcs file from the selected LAB colors and trigger a browser download.

        Implementation details:
        - Creates a temporary folder and writes the .fcs there.
        - Reuses the existing PyFCS writer through Input.instance('.fcs').
        - Uses ui.download() to send the generated file to the client.

        Parameters
        ----------
        name : str
            Output fuzzy color space name (also used as filename).
        selected_colors_lab : dict
            Mapping color name -> np.array([L, a, b]) used to build the .fcs content.
        """
        self.show_loading("Creating .fcs file...")

        tmp_path = None
        try:
            # Create a temporary output path
            tmp_dir = tempfile.mkdtemp(prefix='pyfcs_')
            tmp_path = os.path.join(tmp_dir, f'{name}.fcs')

            # Reuse your InputFCS.write_file via the Input factory
            input_fcs = Input.instance('.fcs')

            # NOTE: write_file must accept `file_path` to write to a specific output location
            input_fcs.write_file(name, selected_colors_lab, file_path=tmp_path)

            # Trigger browser download
            ui.download(tmp_path, filename=f'{name}.fcs')
            ui.notify('Download started')

        except Exception as ex:
            self.custom_warning("Save Error", str(ex))

        finally:
            self.hide_loading()



















    # -------------------------------------------------------------------------
    # IMAGE MANAGER (PyFCS Web Lite)
    #
    # This section provides the web equivalent of the desktop "Image Manager".
    # It allows users to:
    # - Load preset images shipped with the web demo (server-side files).
    # - Upload images from the browser (saved temporarily on the server).
    # - Open each image inside a draggable, resizable "floating window" (a NiceGUI card).
    # - Provide per-window actions (Original / Color Mapping / Color Mapping All) via a menu.
    # - Cleanly close image windows and purge all cached data related to that window.
    # -------------------------------------------------------------------------
    def list_preset_images(self) -> dict:
        """
        Return a dict mapping preset label -> absolute image path for images shipped with the app.

        Presets are expected under:
            ../../image_test   (relative to current_dir)

        Returns
        -------
        dict
            { "label_without_extension": "/abs/path/to/image.ext", ... }
            Empty dict if the preset directory does not exist.
        """
        presets_dir = os.path.abspath(os.path.join(current_dir, '..', '..', 'image_test'))
        if not os.path.isdir(presets_dir):
            return {}

        # Supported image extensions
        exts = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp')

        # List images and return a label->path mapping
        files = sorted(f for f in os.listdir(presets_dir) if f.lower().endswith(exts))
        return {os.path.splitext(f)[0]: os.path.join(presets_dir, f) for f in files}

    def open_image(self):
        """
        Open the "Open Image" dialog.

        The user can:
        - Load a preset image from the server.
        - Upload a local image from the browser.

        On success, an image floating window is created via create_floating_window().
        """
        presets = self.list_preset_images()

        with ui.dialog() as d, ui.card().classes('w-[560px]'):
            ui.label('Open Image').classes('text-lg font-bold')

            # --- Presets section ---
            if presets:
                ui.label('Load a preset image from the server:').classes('text-sm text-gray-700')

                # Dropdown to select a preset file
                preset_select = ui.select(
                    options=list(presets.keys()),
                    value=list(presets.keys())[0],
                    label='Presets',
                ).classes('w-full')

                with ui.row().classes('justify-end gap-2'):
                    ui.button(
                        'Load preset',
                        icon='image',
                        # Close dialog and open the selected preset in a floating window
                        on_click=lambda: (
                            d.close(),
                            self.create_floating_window(
                                presets[preset_select.value],
                                preset_select.value
                            )
                        )
                    )
            else:
                ui.label('No preset images found on the server.').classes('text-sm text-gray-500')

            ui.separator()

            # --- Upload section ---
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
        """
        NiceGUI upload callback for images.

        Workflow:
        - Close the dialog
        - Read uploaded file bytes from the browser (await e.file.read())
        - Save into a temporary directory on the server
        - Create a floating window pointing to that temporary file

        Notes
        -----
        This method does not delete the temporary file immediately because the
        image window needs continued access to it while open.
        """
        dialog.close()
        self.show_loading("Loading image...")

        tmp_path = None
        try:
            original_name = e.file.name
            content_bytes = await e.file.read()

            # Keep original extension when possible
            suffix = os.path.splitext(original_name)[1].lower() or '.png'

            # Store uploaded image server-side
            tmp_dir = tempfile.mkdtemp(prefix='pyfcs_img_')
            tmp_path = os.path.join(tmp_dir, f'uploaded{suffix}')

            with open(tmp_path, 'wb') as f:
                f.write(content_bytes)

            # Open the uploaded image in a floating window
            self.create_floating_window(tmp_path, display_name=original_name)

        except Exception as ex:
            self.custom_warning("Image Error", str(ex))
        finally:
            self.hide_loading()

    def create_floating_window(self, filename: str, display_name: str | None = None):
        """
        Create a web "floating window" for an image.

        The floating window is implemented as a NiceGUI card that is:
        - Draggable (via the JS makeDraggable helper)
        - Resizable (CSS resize: both)
        - Self-contained (stores references to its UI widgets in self.image_windows)

        The window also prepares per-image caches that are used by mapping/legend features.

        Parameters
        ----------
        filename : str
            Path to the image file on the server.
        display_name : str | None
            Optional window title. If None, the basename of `filename` is used.
        """
        # Main registry: window_id -> metadata + widget references
        if not hasattr(self, "image_windows"):
            self.image_windows = {}

        # --- Ensure caches exist (so we can remove entries on close) ---
        if not hasattr(self, "label_map_cache"):
            self.label_map_cache = {}          # e.g. (window_id, max_side) -> label_map
        if not hasattr(self, "mapping_all_cache"):
            self.mapping_all_cache = {}        # e.g. (window_id, scheme, max_side) -> data_url
        if not hasattr(self, "proto_map_cache"):
            self.proto_map_cache = {}          # e.g. (window_id, chosen, max_side, scheme) -> (data_url, info_text)
        if not hasattr(self, "proto_membership_cache"):
            self.proto_membership_cache = {}   # e.g. (window_id, chosen, max_side) -> gray_uint8
        if not hasattr(self, "scheme_cache"):
            self.scheme_cache = {}             # window_id -> 'centroid' | 'hsv'

        # Unique window id
        window_id = f"img_{len(self.image_windows) + 1}"

        # Window title shown in the title bar
        title = display_name or os.path.basename(filename)

        # Per-window state: whether to show original vs mapped image
        if not hasattr(self, "ORIGINAL_IMG"):
            self.ORIGINAL_IMG = {}
        self.ORIGINAL_IMG.setdefault(window_id, True)

        # Per-window state: whether membership overlays are enabled (only if a color space is loaded)
        if hasattr(self, "MEMBERDEGREE_IMG"):
            self.MEMBERDEGREE_IMG.setdefault(window_id, bool(self.COLOR_SPACE))

        # Initial cascade position so multiple windows do not overlap perfectly
        x0 = 20 + 30 * (len(self.image_windows) % 6)
        y0 = 20 + 30 * (len(self.image_windows) % 6)

        # IMPORTANT: create the registry entry BEFORE creating widgets that reference it
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

        # Render the floating card inside the image workspace container
        with self.image_workspace:
            card = ui.card().classes('w-[320px] max-w-[700px] max-h-[700px]').props(f'id={window_id}').style(
                # Absolute positioning + resizable card
                f'position:absolute; left:{x0}px; top:{y0}px; z-index:2000; resize: both; overflow: auto;'
                'min-width:220px; min-height:220px;'
            )
            self.image_windows[window_id]["card"] = card

            with card:
                # Title bar (acts as drag handle)
                handle_id = f'{window_id}_handle'
                with ui.row().classes('w-full items-center justify-between q-pa-sm bg-gray-200'):
                    ui.label(title).classes('text-sm font-bold select-none').props(f'id={handle_id}')

                    # Window controls: menu + close button
                    with ui.row().classes('gap-1'):
                        with ui.menu() as m:
                            ui.menu_item('Original Image', on_click=lambda wid=window_id: self.show_original_image(wid))
                            ui.menu_item('Color Mapping', on_click=lambda wid=window_id: self.color_mapping(wid))
                            ui.menu_item('Color Mapping All', on_click=lambda wid=window_id: self.color_mapping_all(wid))
                            # No "Toggle Legend" entry (as requested)

                        ui.button(icon='more_vert', on_click=m.open).props('flat dense')
                        ui.button(icon='close', on_click=lambda wid=window_id: self.close_image_window(wid)).props('flat dense')

                # Image widget (grows with the resized card)
                img = ui.image(filename).classes('w-full h-auto object-contain bg-white q-ma-sm')
                self.image_windows[window_id]["img"] = img

                # Legend container (hidden by default)
                legend_box = ui.card().classes('w-full q-ma-sm q-pa-sm').style('display:none;')
                with legend_box:
                    legend_title = ui.label('Legend').classes('font-bold text-sm')
                    legend_scroll = ui.scroll_area().classes('w-full h-[110px] q-pa-xs')
                    legend_info = ui.label('').classes('text-xs text-gray-600')

                    # Alternate color scheme button (centroid/hsv, etc.)
                    alt_btn = ui.button(
                        'Alt. Colors',
                        on_click=lambda wid=window_id: self.toggle_color_scheme(wid),
                    ).props('dense')

                # Store legend widget references
                self.image_windows[window_id]["legend_box"] = legend_box
                self.image_windows[window_id]["legend_title"] = legend_title
                self.image_windows[window_id]["legend_scroll"] = legend_scroll
                self.image_windows[window_id]["legend_info"] = legend_info
                self.image_windows[window_id]["alt_colors_btn"] = alt_btn

        # Enable drag behavior once the DOM is ready
        ui.timer(
            0.05,
            lambda: ui.run_javascript(f"makeDraggable('{window_id}', '{window_id}_handle');"),
            once=True,
        )

    def toggle_legend(self, window_id: str):
        """
        Show/hide the legend panel inside a given image window.

        Note:
        In your current menu you removed the "Toggle Legend" action, but this
        helper is still available and may be called from elsewhere if needed.
        """
        win = self.image_windows.get(window_id)
        if not win:
            return

        # Flip visibility state
        win["legend_visible"] = not win.get("legend_visible", False)

        # Apply CSS display toggle
        box = win.get("legend_box")
        if box:
            box.style('display:block;' if win["legend_visible"] else 'display:none;')

    def close_image_window(self, window_id: str):
        """
        Close an image floating window and purge all related cached data.

        This method:
        - Removes the card from the DOM
        - Deletes any cache entries tied to the window_id
        - Removes per-window state flags (scheme_cache, ORIGINAL_IMG, MEMBERDEGREE_IMG)
        - Finally deletes the entry from self.image_windows
        """
        win = getattr(self, "image_windows", {}).get(window_id)
        if not win:
            return

        # Remove window card from the page
        if win.get("card") is not None:
            win["card"].delete()

        # --- Clear caches related to this image window ---
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

        # Remove per-window scheme state
        if hasattr(self, "scheme_cache") and window_id in self.scheme_cache:
            del self.scheme_cache[window_id]

        # Remove per-window flags
        if hasattr(self, "ORIGINAL_IMG") and window_id in self.ORIGINAL_IMG:
            del self.ORIGINAL_IMG[window_id]

        if hasattr(self, "MEMBERDEGREE_IMG") and window_id in self.MEMBERDEGREE_IMG:
            del self.MEMBERDEGREE_IMG[window_id]

        # Remove from registry
        del self.image_windows[window_id]















    # -------------------------------------------------------------------------
    # IMAGE PROCESSING / MAPPING FUNCTIONS (PyFCS Web Lite)
    #
    # This block contains the core image-related functionality in the web demo:
    # - Single-prototype membership visualization (grayscale membership map)
    # - Full image color mapping (each pixel -> best fuzzy label)
    # - Helpers for caching, legend rendering, and color scheme switching
    #
    # IMPORTANT IDEA:
    # The web version heavily uses caching because recomputing fuzzy memberships
    # per pixel is expensive. For repeated actions, cached results are reused.
    # -------------------------------------------------------------------------
    def color_mapping(self, window_id: str):
        """
        Apply a **single-prototype** membership visualization on the selected image window.

        This operation:
        - Requires a loaded color space (COLOR_SPACE == True).
        - Lets the user pick ONE prototype label.
        - Computes (or reuses) a membership map for that prototype:
            * Result is a grayscale image where brightness ~ membership degree.
        - Updates the floating window image source with the rendered output.
        - Renders a legend that includes only the selected prototype label.

        Caching strategy:
        - proto_membership_cache[(window_id, chosen, max_side)] stores the computed grayscale map.
        - proto_map_cache[(window_id, chosen, max_side, scheme)] stores the final PNG data URL + info text.
        """
        # Require a loaded color space
        if not getattr(self, 'COLOR_SPACE', False):
            self.custom_warning("No Color Space", "Load a color space first (.cns or .fcs).")
            return

        # Available labels come from color_matrix
        labels = list(getattr(self, 'color_matrix', []) or [])
        if not labels:
            self.custom_warning("No Data", "No colors loaded to map.")
            return

        # Ensure caches exist
        if not hasattr(self, "proto_map_cache"):
            self.proto_map_cache = {}
        if not hasattr(self, "proto_membership_cache"):
            self.proto_membership_cache = {}
        if not hasattr(self, "scheme_cache"):
            self.scheme_cache = {}

        # Dialog: select which prototype to visualize
        with ui.dialog() as d, ui.card().classes('w-[520px]'):
            ui.label('Color Mapping (single prototype)').classes('text-lg font-bold')
            ui.label('Membership map for ONE prototype (grayscale).').classes('text-sm text-gray-600')

            sel = ui.select(options=labels, value=labels[0], label='Prototype').classes('w-full')

            async def _apply():
                # Close dialog and start processing
                d.close()
                self.show_loading("Color Mapping...")

                try:
                    max_side = 400

                    # 1) Load a reduced working image (HxWx3 uint8)
                    img_np = self._get_work_image_np(window_id, max_side=max_side)

                    chosen = sel.value

                    # Scheme does not affect grayscale, but kept in cache key for consistency
                    scheme = self.scheme_cache.get(window_id, 'centroid')
                    cache_key = (window_id, chosen, max_side, scheme)

                    # 2) If we already have the final rendered result, reuse it instantly
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

                        # Update window state flags
                        self.ORIGINAL_IMG[window_id] = True
                        if hasattr(self, "MEMBERDEGREE"):
                            self.MEMBERDEGREE[window_id] = False
                        return

                    # 3) Compute membership map ONLY for this prototype (cached separately)
                    mkey = (window_id, chosen, max_side)
                    if mkey in self.proto_membership_cache:
                        gray = self.proto_membership_cache[mkey]
                    else:
                        proto_index = self._proto_index_by_label(chosen)
                        gray = await asyncio.to_thread(self._membership_map_for_prototype, img_np, proto_index)
                        self.proto_membership_cache[mkey] = gray

                    # 4) Convert grayscale to 3-channel RGB for ui.image
                    out = np.stack([gray, gray, gray], axis=-1)

                    # Percentage of pixels with non-zero membership
                    pct = float((gray > 0).sum()) / float(gray.size) * 100.0
                    info_text = f'Selected: {chosen} — {pct:.2f}% (nonzero membership)'

                    # Store last output and update window image
                    self.modified_image[window_id] = out
                    data_url = self._np_to_data_url(out)

                    win = self.image_windows[window_id]
                    win["img"].set_source(data_url)
                    win["current_source"] = data_url

                    # 5) Render legend for only the chosen label (and hide alt-colors button)
                    self._render_legend(
                        window_id,
                        only_labels=[chosen],
                        info=info_text,
                        mode='single',
                    )

                    # Cache final render for instant reuse
                    self.proto_map_cache[cache_key] = (data_url, info_text)

                    # Update window flags
                    self.ORIGINAL_IMG[window_id] = True
                    if hasattr(self, "MEMBERDEGREE"):
                        self.MEMBERDEGREE[window_id] = False

                    # Optionally auto-show the legend panel
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
        """
        Convert an RGB uint8 numpy array into a PNG data URL.

        Parameters
        ----------
        arr_uint8 : np.ndarray
            Array of shape (H, W, 3), dtype=uint8.

        Returns
        -------
        str
            A string like: 'data:image/png;base64,...'
        """
        im = Image.fromarray(arr_uint8, mode='RGB')
        buf = io.BytesIO()
        im.save(buf, format='PNG')
        b64 = base64.b64encode(buf.getvalue()).decode('ascii')
        return f'data:image/png;base64,{b64}'

    def _label_colors_centroid(self) -> dict:
        """
        Build a mapping: label -> RGB uint8 using each label's centroid color.

        The centroid is read from:
        - color_data[label]['positive_prototype'] (preferred)
        - or color_data[label]['Color'] (fallback)

        Returns
        -------
        dict
            { label: np.array([R,G,B], dtype=np.uint8), ... }
        """
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
        """
        Build a mapping: label -> RGB uint8 using an HSV colormap.

        This provides clearly separated colors for visualization when the
        centroid-based colors may be too similar.

        Special case:
        - If the label name is "black" (case-insensitive), force RGB to (0,0,0).

        Returns
        -------
        dict
            { label: np.array([R,G,B], dtype=np.uint8), ... }
        """
        labels = list(getattr(self, 'color_matrix', []) or [])
        if not labels:
            return {}

        cmap = plt.get_cmap('hsv', len(labels))
        out = {}

        for i, lab in enumerate(labels):
            rgb01 = np.array(cmap(i)[:3], dtype=float)
            rgb255 = (np.clip(rgb01, 0, 1) * 255).astype(np.uint8)

            # Keep "black" truly black
            if lab.lower() == 'black':
                rgb255 = np.array([0, 0, 0], dtype=np.uint8)

            out[lab] = rgb255

        return out

    def _compute_label_map(self, img_uint8: np.ndarray, progress_callback=None) -> np.ndarray:
        """
        Compute the best fuzzy label for each pixel in an image.

        Steps:
        - Convert RGB uint8 -> LAB (skimage)
        - Quantize LAB values to 0.01 precision (round(..., 2))
        - For each pixel:
            * compute membership dict = fuzzy_color_space.calculate_membership(lab_color)
            * choose the label with maximum membership
        - Returns a (H, W) array of Python objects (strings or None)

        Performance notes:
        - A membership_cache dict is used to avoid recalculating memberships for
          repeated LAB values (very common after quantization).
        - progress_callback(processed, total) can be provided to update UI progress.

        Parameters
        ----------
        img_uint8 : np.ndarray
            Image as HxWx3 uint8.
        progress_callback : callable | None
            Function called with (processed, total) occasionally.

        Returns
        -------
        np.ndarray
            Object array (H, W) with the best label (string) or None.
        """
        # Ensure internal precomputations are ready (safe to call repeatedly)
        self.fuzzy_color_space.precompute_pack()

        # Convert RGB -> LAB
        img01 = img_uint8.astype(np.float32) / 255.0
        lab_img = skcolor.rgb2lab(img01)

        # Quantize LAB to reduce unique values and speed up caching
        lab_q = np.round(lab_img, 2)

        h, w = lab_q.shape[:2]
        total = h * w
        label_map = np.empty((h, w), dtype=object)

        # Cache: quantized LAB key -> best label
        membership_cache = {}
        processed = 0

        for y in range(h):
            for x in range(w):
                lab_color = lab_q[y, x]

                # Integer key to reduce float hashing issues
                key = (int(lab_color[0] * 100), int(lab_color[1] * 100), int(lab_color[2] * 100))

                best_label = membership_cache.get(key)
                if best_label is None:
                    m = self.fuzzy_color_space.calculate_membership(lab_color)
                    best_label = max(m, key=m.get) if m else None
                    membership_cache[key] = best_label

                label_map[y, x] = best_label
                processed += 1

                # Update progress periodically
                if progress_callback and (processed % 5000 == 0 or processed == total):
                    progress_callback(processed, total)

        return label_map

    def show_original_image(self, window_id: str):
        """
        Restore and display the original image in the floating window.

        This:
        - Sets the image source back to win["original_source"]
        - Hides the legend panel
        - Updates ORIGINAL_IMG state flag
        """
        try:
            win = self.image_windows[window_id]
            win["img"].set_source(win["original_source"])
            win["current_source"] = win["original_source"]

            # Hide legend/controls
            if "legend_box" in win:
                win["legend_box"].style('display:none;')

            self.ORIGINAL_IMG[window_id] = False
            ui.notify('Original image restored')

        except Exception as e:
            self.custom_warning("Display Error", str(e))

    def color_mapping_all(self, window_id: str):
        """
        Apply **full image** color mapping: each pixel is assigned the best fuzzy label,
        and then recolored using a chosen color scheme.

        Requirements:
        - A color space must be loaded.

        Workflow:
        - Load a reduced working image (max_side=400).
        - Compute (or reuse) the label_map for this image window:
            label_map[y,x] = best label (string) or None
        - Convert label_map -> RGB output using a color scheme:
            * 'centroid': use each label centroid color
            * 'hsv'     : use an HSV colormap for distinct label colors
        - Render output as a PNG data URL and update the window image.
        - Render/update the legend.

        UI threading:
        - Uses `context.client` and wraps UI updates with `with client:` because
          heavy computations are done in a background task.
        """
        if not getattr(self, 'COLOR_SPACE', False):
            self.custom_warning("No Color Space", "Load a color space first (.cns or .fcs).")
            return

        client = context.client  # UI context for the current user session

        async def _run():
            # UI calls must be executed within the client context
            with client:
                self.show_loading("Color Mapping All...")

            try:
                # Reduced image for faster preview mapping
                img_np = self._get_work_image_np(window_id, max_side=400)

                # Progress callback updates the loading bar
                def progress(cur, total):
                    with client:
                        self.set_loading_progress(cur / total)

                # Compute label map only once per window (cached)
                if window_id not in self.label_map_cache:
                    label_map = await asyncio.to_thread(self._compute_label_map, img_np, progress)
                    self.label_map_cache[window_id] = label_map
                else:
                    label_map = self.label_map_cache[window_id]

                # Choose recoloring scheme for labels
                scheme = self.scheme_cache.get(window_id, 'centroid')
                colors = self._label_colors_centroid() if scheme == 'centroid' else self._label_colors_hsv()

                # Build recolored output image
                h, w = label_map.shape
                out = np.zeros((h, w, 3), dtype=np.uint8)

                for label, rgb in colors.items():
                    out[label_map == label] = rgb

                # Pixels with no label -> black
                out[label_map == None] = np.array([0, 0, 0], dtype=np.uint8)

                # Store and display
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

        # Run asynchronously so the UI remains responsive
        asyncio.create_task(_run())

    def _get_work_image_np(self, window_id: str, max_side: int = 400) -> np.ndarray:
        """
        Load the image window file into a resized numpy array (uint8 RGB).

        Resizing is performed so that max(width, height) <= max_side,
        keeping aspect ratio, using LANCZOS for good downsampling.

        Parameters
        ----------
        window_id : str
            ID of the image window.
        max_side : int
            Maximum allowed size for the longer image side.

        Returns
        -------
        np.ndarray
            HxWx3 uint8 RGB image.
        """
        path = self.image_windows[window_id]["path"]
        img = Image.open(path).convert('RGB')

        w, h = img.size
        if max(w, h) > max_side:
            scale = max_side / float(max(w, h))
            img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)

        return np.array(img, dtype=np.uint8)

    def _render_legend(self, window_id: str, only_labels=None, info: str | None = None, mode: str = 'all'):
        """
        Render/update the legend panel for an image window.

        The legend displays:
        - A swatch per label (from the current color scheme)
        - The label name
        - Optional info text (e.g., selected prototype stats)

        Parameters
        ----------
        window_id : str
            Target image window.
        only_labels : list[str] | None
            If provided, render only these labels. Otherwise render all labels.
        info : str | None
            Optional text shown under the legend list.
        mode : str
            'all'    -> show Alt. Colors button
            'single' -> hide Alt. Colors button (used in single-prototype mode)
        """
        win = self.image_windows[window_id]

        legend_box = win.get("legend_box")
        legend_scroll = win.get("legend_scroll")
        legend_info = win.get("legend_info")
        alt_btn = win.get("alt_colors_btn")

        if legend_box is None or legend_scroll is None or legend_info is None:
            return

        # Always show legend when rendering it
        legend_box.style('display:block;')

        # Show/hide Alt. Colors depending on mode
        if alt_btn is not None:
            alt_btn.set_visibility(mode == 'all')

        # Determine current scheme and build label->RGB mapping
        scheme = self.scheme_cache.get(window_id, 'centroid')
        colors = self._label_colors_centroid() if scheme == 'centroid' else self._label_colors_hsv()

        # Decide which labels to show
        labels = only_labels if only_labels is not None else list(getattr(self, 'color_matrix', []) or [])

        # Build legend entries
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

        # Info text area
        legend_info.set_text(info or '')

    def toggle_color_scheme(self, window_id: str):
        """
        Toggle the recoloring scheme used in Color Mapping All:
        - 'centroid' (default): uses centroid LAB->RGB per label
        - 'hsv'              : uses an HSV colormap for distinct label colors

        If a label_map is already cached for this window, recoloring is reapplied
        instantly without recalculating memberships.
        """
        current = self.scheme_cache.get(window_id, 'centroid')
        self.scheme_cache[window_id] = 'hsv' if current == 'centroid' else 'centroid'

        # If label map exists, recolor immediately (no recomputation)
        if window_id in self.label_map_cache:
            self.color_mapping_all(window_id)
        else:
            self._render_legend(window_id)

    def _proto_index_by_label(self, label: str) -> int:
        """
        Return the index of a Prototype in self.prototypes matching the given label.

        Raises
        ------
        ValueError
            If no prototype with that label exists.
        """
        for i, p in enumerate(self.prototypes):
            if getattr(p, 'label', None) == label:
                return i
        raise ValueError(f'Prototype not found: {label}')

    def _membership_map_for_prototype(self, img_np_rgb255: np.ndarray, proto_index: int, progress_cb=None) -> np.ndarray:
        """
        Compute a grayscale membership map (uint8) for ONE prototype over an image.

        Parameters
        ----------
        img_np_rgb255 : np.ndarray
            Input image as HxWx3 uint8 (0..255).
        proto_index : int
            Index of the prototype in self.prototypes for which membership is computed.
        progress_cb : callable | None
            Optional callback progress_cb(current, total).

        Returns
        -------
        np.ndarray
            HxW uint8 array where 0..255 corresponds to membership degree 0..1.
        """
        # Convert RGB to LAB
        img = img_np_rgb255.astype(np.float32) / 255.0
        lab = skcolor.rgb2lab(img)

        # Quantize for caching and stability
        lab_q = np.round(lab, 2)

        h, w, _ = lab_q.shape
        flat = lab_q.reshape(-1, 3)

        total = flat.shape[0]
        out = np.empty(total, dtype=np.float32)

        # Cache repeated LAB values -> membership (saves time due to quantization)
        cache = {}
        fcs = self.fuzzy_color_space  # alias

        for i in range(total):
            L, A, B = flat[i]
            key = (float(L), float(A), float(B))

            val = cache.get(key)
            if val is None:
                val = fcs.calculate_membership_for_prototype(
                    np.array([L, A, B], dtype=float),
                    proto_index
                )
                cache[key] = val

            out[i] = val

            if progress_cb and (i % 5000 == 0 or i == total - 1):
                progress_cb(i + 1, total)

        # Scale membership (0..1) -> grayscale (0..255)
        gray = (out.reshape(h, w) * 255.0)
        gray = np.clip(gray, 0, 255).astype(np.uint8)
        return gray











    
    # ---------------------------------------------------------------------
    # SAVE IMAGE
    # ---------------------------------------------------------------------
    def _compose_with_legend(self, img_uint8: np.ndarray, window_id: str) -> Image.Image:
        """
        Create a new PIL image with a **legend appended at the bottom**.

        The legend is built from:
        - self.color_matrix (label order)
        - The current color scheme for this window (centroid vs hsv)
        - Swatches + label text drawn onto a white background

        Parameters
        ----------
        img_uint8 : np.ndarray
            Base image as an RGB uint8 array (H, W, 3).
        window_id : str
            Image window id used to read the current scheme (scheme_cache) and label colors.

        Returns
        -------
        PIL.Image.Image
            A new image containing the original content on top and a legend area below.
        """
        # Convert numpy array to PIL Image
        img = Image.fromarray(img_uint8, mode='RGB')

        # Labels to display in the legend (same order as the loaded color space)
        labels = list(getattr(self, 'color_matrix', []) or [])

        # Pick the current recoloring scheme for this window
        scheme = self.scheme_cache.get(window_id, 'centroid')
        colors = self._label_colors_centroid() if scheme == 'centroid' else self._label_colors_hsv()

        # --- Legend layout parameters ---
        pad = 12          # padding around the legend
        swatch = 18       # swatch square size
        line_h = 24       # height per legend row
        max_lines = max(1, len(labels))

        # Output image = original height + legend height
        legend_h = pad * 2 + line_h * max_lines
        out = Image.new('RGB', (img.width, img.height + legend_h), (255, 255, 255))
        out.paste(img, (0, 0))

        draw = ImageDraw.Draw(out)

        # Font setup (safe fallback if arial.ttf is unavailable)
        try:
            font = ImageFont.truetype("arial.ttf", 14)
        except Exception:
            font = ImageFont.load_default()

        # Starting origin for legend content
        y0 = img.height + pad
        x0 = pad

        # Draw each label row: swatch + label name
        for i, lab in enumerate(labels):
            rgb = colors.get(lab, np.array([0, 0, 0], dtype=np.uint8))
            fill_rgb = (int(rgb[0]), int(rgb[1]), int(rgb[2]))
            y = y0 + i * line_h

            # Color swatch (outlined in black)
            draw.rectangle(
                [x0, y + 3, x0 + swatch, y + 3 + swatch],
                fill=fill_rgb,
                outline=(0, 0, 0)
            )

            # Label text
            draw.text((x0 + swatch + 10, y + 2), lab, fill=(0, 0, 0), font=font)

        return out

    def save_image(self):
        """
        Download an image from one of the currently open floating windows.

        The user can choose:
        - Which image window to save
        - Output format: PNG or JPG
        - Whether to include the legend (only meaningful if a mapping/label_map exists)

        Save logic:
        - If a mapped/processed image exists in self.modified_image[wid], that version is saved.
        - Otherwise, the original image file is saved.
        - If "Include legend" is enabled AND a label_map is cached for that window,
          a legend block is appended at the bottom before encoding.
        """
        # Must have at least one open image window
        wins = getattr(self, "image_windows", {})
        if not wins:
            self.custom_warning("Save Image", "There are no image windows to save.")
            return

        # Build select options: window_id -> title
        options = {wid: wins[wid].get("title", wid) for wid in wins.keys()}

        with ui.dialog() as d, ui.card().classes('w-[520px]'):
            ui.label('Save Image').classes('text-lg font-bold')
            ui.label('Choose which image you want to download.').classes('text-sm text-gray-600')

            # Select which floating window to export
            sel = ui.select(
                options=options,
                value=list(options.keys())[0],
                label='Image window',
            ).classes('w-full')

            # Export options
            include_legend = ui.checkbox('Include legend (if available)', value=True)
            fmt = ui.select(options=['png', 'jpg'], value='png', label='Format').classes('w-full')

            async def _download():
                """
                Encode and send the selected image to the browser as a download.
                """
                wid = sel.value
                d.close()

                try:
                    # Decide which image data to export:
                    # - mapped/processed image if available
                    # - otherwise the original image
                    if hasattr(self, 'modified_image') and wid in self.modified_image:
                        arr = self.modified_image[wid]
                        pil = Image.fromarray(arr.astype(np.uint8), mode='RGB')
                    else:
                        path = self.image_windows[wid]["path"]
                        pil = Image.open(path).convert('RGB')

                    # Append legend if requested AND if we have a label_map for this window
                    if include_legend.value and hasattr(self, 'label_map_cache') and wid in self.label_map_cache:
                        pil = self._compose_with_legend(np.array(pil, dtype=np.uint8), wid)

                    # Encode to bytes
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

                    # Build a safe download filename
                    base = self.image_windows[wid].get("title", wid).replace(' ', '_')
                    filename = f'{base}.{ext}'

                    # Force browser download
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
