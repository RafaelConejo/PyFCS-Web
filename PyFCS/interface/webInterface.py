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

import hashlib
import time
import shutil

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

        self.is_loading_color_space = False     # Flag indicating a FCS is loading
        self.bulk_updating_colors = False       # Flag to avoid triggering callbacks during bulk checkbox updates
        self.color_checkboxes = {}              # Dict: color_name -> checkbox reference (UI state control)
        self.loading_dialog = None              # Reference to the loading dialog (created once and reused)
        self.loading_label = None               # Label inside loading dialog to show current operation message
        self.is_processing_mapping = False      # Flag indicating Color Mapping / Mapping All is running
        self.cancel_mapping_requested = False   # Flag requested by user to cancel current mapping task
        self.current_mapping_window_id = None       # Window id currently being processed
        self.loading_cancel_btn = None          # Reference to the cancel button inside loading dialog

        # Lower value = faster mapping.
        self.PREVIEW_MAX_SIDE = 256

        # LAB quantization step for web preview.
        self.LAB_QUANT_STEP = 0.05

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
                window.__pyfcsZ += 2;
                el.style.zIndex = window.__pyfcsZ;

                // If this image window has a linked legend window, keep it above the image
                const legend = document.getElementById(el.id + '_legend');
                if (legend) {
                    legend.style.zIndex = window.__pyfcsZ + 1;
                }
            }

            function makeDraggable(elId, handleId) {
                const el = document.getElementById(elId);
                const handle = document.getElementById(handleId);
                if (!el || !handle) return;

                if (el._pyfcsDragInstalled) return;
                el._pyfcsDragInstalled = true;

                el.style.position = 'absolute';
                el.style.willChange = 'left, top';
                el.style.transform = 'none';

                handle.style.cursor = 'move';
                handle.style.userSelect = 'none';
                handle.style.touchAction = 'none';

                // Source of truth for position
                if (!el.dataset.x) el.dataset.x = String(el.offsetLeft || 0);
                if (!el.dataset.y) el.dataset.y = String(el.offsetTop || 0);

                function applyStoredPosition() {
                    const x = parseFloat(el.dataset.x || '0');
                    const y = parseFloat(el.dataset.y || '0');
                    el.style.left = x + 'px';
                    el.style.top = y + 'px';
                    el.style.transform = 'none';
                }

                let dragging = false;
                let startX = 0, startY = 0;
                let startLeft = 0, startTop = 0;
                let raf = 0;
                let nextLeft = 0, nextTop = 0;

                function applyPos() {
                    raf = 0;
                    el.dataset.x = String(nextLeft);
                    el.dataset.y = String(nextTop);
                    applyStoredPosition();
                }

                function onMove(e) {
                    if (!dragging) return;

                    const dx = e.clientX - startX;
                    const dy = e.clientY - startY;

                    nextLeft = startLeft + dx;
                    nextTop = startTop + dy;

                    if (!raf) raf = requestAnimationFrame(applyPos);
                    e.preventDefault();
                }

                function onUp(e) {
                    if (!dragging) return;
                    dragging = false;

                    window.removeEventListener('pointermove', onMove, true);
                    window.removeEventListener('pointerup', onUp, true);

                    try { el.releasePointerCapture(e.pointerId); } catch {}

                    if (raf) {
                        cancelAnimationFrame(raf);
                        raf = 0;
                    }

                    el.dataset.x = String(nextLeft);
                    el.dataset.y = String(nextTop);
                    applyStoredPosition();

                    e.preventDefault();
                }

                handle.addEventListener('pointerdown', (e) => {
                    if (e.button !== 0) return;

                    dragging = true;
                    bringToFront(el);

                    startLeft = parseFloat(el.dataset.x || el.offsetLeft || 0);
                    startTop  = parseFloat(el.dataset.y || el.offsetTop || 0);

                    startX = e.clientX;
                    startY = e.clientY;

                    nextLeft = startLeft;
                    nextTop = startTop;

                    try { el.setPointerCapture(e.pointerId); } catch {}

                    window.addEventListener('pointermove', onMove, true);
                    window.addEventListener('pointerup', onUp, true);

                    e.preventDefault();
                    e.stopPropagation();
                });

                el.addEventListener('mousedown', () => bringToFront(el), {passive: true});

                // IMPORTANT:
                // If NiceGUI rewrites the style attribute, restore the stored x/y immediately.
                const styleObserver = new MutationObserver(() => {
                    if (!dragging) applyStoredPosition();
                });
                styleObserver.observe(el, {
                    attributes: true,
                    attributeFilter: ['style']
                });
                el._pyfcsStyleObserver = styleObserver;

                // Initial position application
                applyStoredPosition();
            }

            window.restoreFloatingPosition = function(elId) {
                const el = document.getElementById(elId);
                if (!el) return;

                const x = parseFloat(el.dataset.x || '0');
                const y = parseFloat(el.dataset.y || '0');

                el.style.left = x + 'px';
                el.style.top = y + 'px';
                el.style.transform = 'none';
            };
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
                    self.btn_file_menu = ui.button('File', icon='menu').props('flat')

                # Image manager menu
                with ui.menu():
                    ui.menu_item('Open Image', on_click=self.open_image)
                    ui.menu_item('Save Image', on_click=self.save_image)
                    ui.menu_item('Close All', on_click=self.close_all_image)
                    self.btn_image_manager_menu = ui.button('Image Manager', icon='image').props('flat')

                # Fuzzy color space menu
                with ui.menu():
                    ui.menu_item('New Color Space', on_click=self.show_menu_create_fcs)
                    ui.menu_item('Load Color Space', on_click=self.load_color_space)
                    self.btn_fuzzy_color_space_menu = ui.button('Fuzzy Color Space', icon='palette').props('flat')

                # About action
                self.btn_about = ui.button('About', on_click=self.about_info).props('flat')

        # ---- Toolbar cards (without "Color Evaluation" section) ----
        with ui.row().classes('w-full q-pa-md items-start gap-4'):
            # Image Manager (compact)
            with ui.card().classes('w-[350px] q-pa-md'):
                ui.label('Image Manager').classes('font-bold')
                with ui.row().classes('gap-2'):
                    self.btn_open_image = ui.button('Open Image', icon='folder_open', on_click=self.open_image)
                    self.btn_save_image = ui.button('Save Image', icon='save', on_click=self.save_image)

            # Fuzzy Color Space Manager (wider)
            with ui.card().classes('w-[450px] q-pa-md'):
                ui.label('Fuzzy Color Space Manager').classes('font-bold')
                with ui.row().classes('gap-2 items-center'):
                    self.btn_new_color_space = ui.button('New Color Space', icon='add', on_click=self.show_menu_create_fcs)
                    self.btn_load_color_space = ui.button('Load Color Space', icon='upload_file', on_click=self.load_color_space)

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
                                        self.btn_select_all = ui.button('Select All', on_click=self.select_all_color)
                                        self.btn_deselect_all = ui.button('Deselect All', on_click=self.deselect_all_color)

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
    def set_ui_busy(self, busy: bool):
        widgets = [
            getattr(self, 'btn_file_menu', None),
            getattr(self, 'btn_image_manager_menu', None),
            getattr(self, 'btn_fuzzy_color_space_menu', None),
            getattr(self, 'btn_about', None),
            getattr(self, 'btn_open_image', None),
            getattr(self, 'btn_save_image', None),
            getattr(self, 'btn_new_color_space', None),
            getattr(self, 'btn_load_color_space', None),
            getattr(self, 'btn_select_all', None),
            getattr(self, 'btn_deselect_all', None),
        ]

        for widget in widgets:
            if widget is None:
                continue
            if busy:
                widget.disable()
            else:
                widget.enable()

    def set_color_list(self, color_names: list[str]) -> None:
        """
        (Re)create the checkbox list shown in the right panel and refresh plot state on change.

        This method:
        - Clears any previous checkbox references
        - Ensures MEMBERDEGREE exists as a dict
        - Creates one checkbox per color
        - Stores both the UI checkbox and the persisted enabled/disabled value
        """
        if not hasattr(self, 'color_checkboxes') or not isinstance(self.color_checkboxes, dict):
            self.color_checkboxes = {}
        else:
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

        # Avoid triggering many refreshes during bulk updates
        if getattr(self, 'bulk_updating_colors', False):
            return

        # Ignore updates while color space is loading
        if getattr(self, 'is_loading_color_space', False):
            return

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
        with ui.dialog() as d, ui.card().classes('w-[600px] max-w-full'):
            ui.label(title).classes('text-lg font-bold')
            ui.label(message).classes('text-gray-700').style('white-space: pre-wrap; word-break: break-word;')
            with ui.row().classes('justify-end'):
                ui.button('OK', on_click=d.close)
        d.open()


    def show_loading(self, message="Processing...", cancellable=False):
        """
        Show (or reuse) a loading dialog with a spinner and progress bar.
        """
        self.set_ui_busy(True)

        if self.loading_dialog is None:
            self.loading_dialog = ui.dialog()
            with self.loading_dialog, ui.card().classes('w-80'):
                self.loading_label = ui.label(message).classes('text-base font-bold')
                ui.spinner(size='lg')
                self.loading_progress = ui.linear_progress(0).props(
                    'instant-feedback show-value=false indeterminate'
                )
                with ui.row().classes('justify-end w-full'):
                    self.loading_cancel_btn = ui.button(
                        'Cancel',
                        on_click=self.cancel_current_mapping
                    ).props('flat')
        else:
            self.loading_label.set_text(message)
            self.loading_progress.set_value(0)
            self.loading_progress.props('indeterminate')

        if self.loading_cancel_btn is not None:
            if cancellable:
                self.loading_cancel_btn.enable()
                self.loading_cancel_btn.style('display:block;')
            else:
                self.loading_cancel_btn.disable()
                self.loading_cancel_btn.style('display:none;')

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

        if self.loading_cancel_btn is not None:
            self.loading_cancel_btn.disable()
            self.loading_cancel_btn.style('display:none;')

        self.set_ui_busy(False)


    def reopen_loading_if_busy(self, message="Processing..."):
        """Reopen the loading dialog if a long-running task is already in progress."""
        if getattr(self, 'is_processing_mapping', False):
            if self.loading_label is not None:
                self.loading_label.set_text(message)
            if self.loading_dialog is not None:
                self.loading_dialog.open()
            return True
        return False


    def _is_window_mapping_locked(self, window_id: str) -> bool:
        """Return True if the given window is currently running a mapping task."""
        return (
            getattr(self, 'is_processing_mapping', False)
            and getattr(self, 'current_mapping_window_id', None) == window_id
        )


    def cancel_current_mapping(self):
        """Request cancellation of the current mapping task."""
        if self.is_processing_mapping:
            self.cancel_mapping_requested = True
            ui.notify('Cancelling current mapping...', color='warning')

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
        if getattr(self, 'is_loading_color_space', False):
            ui.notify('Please wait until the current operation finishes.', color='warning')
            return

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
        if getattr(self, 'is_loading_color_space', False):
            ui.notify('Please wait until the current operation finishes.', color='warning')
            return
        ui.notify('Open Image (stub)')

    def save_image(self):
        """Save Image action (placeholder)."""
        if getattr(self, 'is_loading_color_space', False):
            ui.notify('Please wait until the current operation finishes.', color='warning')
            return
        ui.notify('Save Image (stub)')

    def close_all_image(self):
        """Close all images action (placeholder)."""
        if getattr(self, 'is_loading_color_space', False):
            ui.notify('Please wait until the current operation finishes.', color='warning')
            return
        ui.notify('Close All (stub)')

    def load_color_space(self):
        """Load fuzzy color space action (placeholder)."""
        if getattr(self, 'is_loading_color_space', False):
            ui.notify('A file is already being loaded, please wait.', color='warning')
            return
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
        if getattr(self, 'is_loading_color_space', False):
            ui.notify('Please wait until the current operation finishes.', color='warning')
            return

        self.bulk_updating_colors = True
        try:
            for name, cb in self.color_checkboxes.items():
                cb.set_value(True)
                self.MEMBERDEGREE[name] = True
        finally:
            self.bulk_updating_colors = False

        self.update_selected_sets_from_checks()
        self.on_option_select()

    def deselect_all_color(self):
        """Set all color checkboxes to False (disabled)."""
        if getattr(self, 'is_loading_color_space', False):
            ui.notify('Please wait until the current operation finishes.', color='warning')
            return

        self.bulk_updating_colors = True
        try:
            for name, cb in self.color_checkboxes.items():
                cb.set_value(False)
                self.MEMBERDEGREE[name] = False
        finally:
            self.bulk_updating_colors = False

        self.update_selected_sets_from_checks()
        self.on_option_select()

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

        if getattr(self, 'is_loading_color_space', False):
            return

        if not getattr(self, 'COLOR_SPACE', False):
            return

        self.on_option_select()














    # ---------------------------------------------------------------------
    # LOAD FCS
    # ---------------------------------------------------------------------
    def load_color_space(self):
        """
        Open a modal dialog that lets the user load a color space in two ways:
        1) Load a preset .fcs file already available on the server.
        2) Upload a local .cns or .fcs file from the user's computer.
        """
        presets = self.list_preset_fcs()

        with ui.dialog() as d, ui.card().classes('w-[560px]'):
            ui.label('Load Color Space').classes('text-lg font-bold')

            # estado local de widgets para poder bloquearlos
            load_preset_btn = None
            upload_widget = None
            cancel_btn = None

            if presets:
                ui.label('Load a preset (.fcs) from the server:').classes('text-sm text-gray-700')

                preset_select = ui.select(
                    options=list(presets.keys()),
                    value=list(presets.keys())[0],
                    label='Presets',
                ).classes('w-full')

                with ui.row().classes('justify-end gap-2'):
                    load_preset_btn = ui.button(
                        'Load preset',
                        icon='cloud_download',
                        on_click=lambda: self._handle_preset_load(
                            d,
                            presets[preset_select.value],
                            load_preset_btn,
                            upload_widget,
                            cancel_btn
                        )
                    )
            else:
                ui.label('No presets found on the server.').classes('text-sm text-gray-500')

            ui.separator()

            ui.label('Or upload a .cns/.fcs file:').classes('text-sm text-gray-700')

            upload_widget = ui.upload(
                label='Choose file',
                multiple=False,
                auto_upload=True,
                on_upload=lambda e: self._on_color_file_uploaded(
                    e, d, load_preset_btn, upload_widget, cancel_btn
                ),
            ).props('accept=.cns,.fcs')

            with ui.row().classes('justify-end'):
                cancel_btn = ui.button('Cancel', on_click=d.close).props('flat')

        d.open()


    def _set_load_dialog_enabled(self, enabled, *widgets):
        for widget in widgets:
            if widget is None:
                continue
            if enabled:
                widget.enable()
            else:
                widget.disable()


    def _handle_preset_load(self, dialog, filepath, *widgets):
        if self.is_loading_color_space:
            ui.notify('A file is already being loaded, please wait.', color='warning')
            return

        self._set_load_dialog_enabled(False, *widgets)
        dialog.close()
        self.load_color_space_from_path(filepath)



    def load_color_space_from_path(self, filepath: str):
        """
        Load a color space file from a server-side path (preset).
        """
        if self.is_loading_color_space:
            ui.notify('A file is already being loaded, please wait.', color='warning')
            return

        self.is_loading_color_space = True
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
            self.is_loading_color_space = False
            if getattr(self, 'COLOR_SPACE', False):
                self.on_option_select()



    async def _on_color_file_uploaded(self, e, dialog, *widgets):
        """
        NiceGUI upload callback (client -> server).
        """
        if self.is_loading_color_space:
            ui.notify('A file is already being loaded, please wait.', color='warning')
            return

        self.is_loading_color_space = True
        self._set_load_dialog_enabled(False, *widgets)

        dialog.close()
        self.show_loading_color_space()

        tmp_path = None
        try:
            original_name = e.file.name
            content_bytes = await e.file.read()

            suffix = os.path.splitext(original_name)[1].lower() or '.tmp'
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(content_bytes)
                tmp_path = tmp.name

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
            self.is_loading_color_space = False

            if getattr(self, 'COLOR_SPACE', False):
                self.on_option_select()

            if tmp_path:
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass



    def update_volumes(self):
        """
        Compute fuzzy color space volumes from raw color data (.cns).
        """
        self.show_loading("Computing volumes...")

        try:
            self.prototypes = UtilsTools.process_prototypes(self.color_data)

            self.fuzzy_color_space = FuzzyColorSpace(
                space_name=" ",
                prototypes=self.prototypes
            )
            self.fuzzy_color_space.precompute_pack()

            self.cores = self.fuzzy_color_space.get_cores()
            self.supports = self.fuzzy_color_space.get_supports()

            self.update_prototypes_info()

        finally:
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
        if not getattr(self, 'COLOR_SPACE', False):
            return

        if getattr(self, 'is_loading_color_space', False):
            return

        if getattr(self, 'bulk_updating_colors', False):
            return

        # Which layers the user wants to display
        selected_options = [k for k, v in self.model_3d_options.items() if v]

        # If nothing selected -> clear plot
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
                self.plot_widget = ui.plotly(fig).classes('w-full h-full')













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
        if hasattr(self, 'file_base_name') and self.file_base_name:
            self.file_name.set_value(self.file_base_name)

        self.hex_color = {}
        self.color_matrix = []

        rows = []

        for color_name, color_value in (self.color_data or {}).items():
            if not isinstance(color_value, dict):
                continue

            lab = color_value.get('positive_prototype', None)
            if lab is None:
                lab = color_value.get('Color', None)

            if lab is None:
                continue

            try:
                lab = np.array(lab, dtype=float).reshape(3,)
            except Exception:
                continue

            self.color_matrix.append(color_name)

            try:
                rgb01 = skcolor.lab2rgb(lab.reshape(1, 1, 3))[0, 0, :]
            except Exception:
                rgb01 = np.array([0.0, 0.0, 0.0])

            if not np.all(np.isfinite(rgb01)):
                rgb01 = np.array([0.0, 0.0, 0.0])

            rgb01 = np.clip(rgb01, 0.0, 1.0)
            rgb255 = tuple(int(round(c * 255.0)) for c in rgb01)

            hex_color = f'#{rgb255[0]:02x}{rgb255[1]:02x}{rgb255[2]:02x}'

            self.hex_color[hex_color] = lab

            rows.append({
                'L': round(float(lab[0]), 2),
                'a': round(float(lab[1]), 2),
                'b': round(float(lab[2]), 2),
                'name': color_name,
                'color': hex_color,
            })

        self.data_table.rows = rows
        self.data_table.update()

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
        tmp_dir = None
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
            self.create_floating_window(
                tmp_path,
                display_name=original_name,
                temp_dir=tmp_dir,
            )

        except Exception as ex:
            self.custom_warning("Image Error", str(ex))
        finally:
            self.hide_loading()

    def create_floating_window(self, filename: str, display_name: str | None = None, temp_dir: str | None = None):
        """
        Create a web "floating window" for an image.
        """
        if not hasattr(self, "image_windows"):
            self.image_windows = {}

        if not hasattr(self, "_image_window_counter"):
            self._image_window_counter = 0

        if not hasattr(self, "label_map_cache"):
            self.label_map_cache = {}
        if not hasattr(self, "mapping_all_cache"):
            self.mapping_all_cache = {}
        if not hasattr(self, "proto_map_cache"):
            self.proto_map_cache = {}
        if not hasattr(self, "proto_membership_cache"):
            self.proto_membership_cache = {}
        if not hasattr(self, "scheme_cache"):
            self.scheme_cache = {}
        if not hasattr(self, "cm_cache"):
            self.cm_cache = {}
        if not hasattr(self, "best_idx_cache"):
            self.best_idx_cache = {}

        # CHANGED: monotonic unique window ids to avoid reuse
        self._image_window_counter += 1
        window_id = f"img_{self._image_window_counter}"

        title = display_name or os.path.basename(filename)

        if not hasattr(self, "ORIGINAL_IMG"):
            self.ORIGINAL_IMG = {}
        self.ORIGINAL_IMG.setdefault(window_id, True)

        if hasattr(self, "MEMBERDEGREE_IMG"):
            self.MEMBERDEGREE_IMG.setdefault(window_id, bool(self.COLOR_SPACE))

        x0 = 20 + 30 * (len(self.image_windows) % 6)
        y0 = 20 + 30 * (len(self.image_windows) % 6)

        preview_source = self._make_preview_data_url_from_path(filename, max_side=self.PREVIEW_MAX_SIDE)

        self.image_windows[window_id] = {
            "path": filename,
            "title": title,
            "original_source": preview_source,
            "current_source": preview_source,
            "card": None,
            "img": None,
            "img_container": None,
            "legend_box": None,
            "legend_title": None,
            "legend_scroll": None,
            "legend_info": None,
            "alt_colors_btn": None,
            "legend_visible": False,

            # CHANGED: track temp upload folder for cleanup
            "temp_dir": temp_dir,
            "is_temp": temp_dir is not None,

            # CHANGED: persist floating position
            "x": x0,
            "y": y0,

            # preview cache placeholders
            "preview_cache_ready": False,
            "preview_np": None,
            "preview_hash": None,
            "lab_q": None,
            "lab_int": None,
            "uniq": None,
            "inv": None,
            "preview_h": None,
            "preview_w": None,
        }

        # This avoids repeated resize/RGB->LAB/unique work later.
        self._prepare_window_preview_cache(window_id)

        with self.image_workspace:
            card = ui.card() \
            .classes('max-w-[700px] max-h-[700px] flex flex-col') \
            .props(f'id={window_id}') \
            .style(
                f'position:absolute; left:{x0}px; top:{y0}px; z-index:2000; '
                'width:320px; height:320px; resize:both; overflow:hidden; '
                'min-width:220px; min-height:220px;'
            ) \
            .on('window_moved', lambda e, wid=window_id: self._on_window_moved(wid, e))
            self.image_windows[window_id]["card"] = card

            with card:
                handle_id = f'{window_id}_handle'
                with ui.row().classes('w-full items-center justify-between q-pa-sm bg-gray-200 shrink-0'):
                    ui.label(title).classes('text-sm font-bold select-none').props(f'id={handle_id}')

                    with ui.row().classes('gap-1'):
                        with ui.menu() as m:
                            ui.menu_item('Original Image', on_click=lambda wid=window_id: self.show_original_image(wid))
                            ui.menu_item('Color Mapping', on_click=lambda wid=window_id: self.color_mapping(wid))
                            ui.menu_item('Color Mapping All', on_click=lambda wid=window_id: self.color_mapping_all(wid))

                        ui.button(icon='more_vert', on_click=m.open).props('flat dense')
                        ui.button(icon='close', on_click=lambda wid=window_id: self.close_image_window(wid)).props('flat dense')

                with ui.element('div').classes('w-full flex-1 min-h-0 overflow-hidden bg-white q-ma-sm') as img_container:
                    img = ui.image(preview_source).classes('w-full h-full object-contain')
                    self.image_windows[window_id]["img"] = img
                    self.image_windows[window_id]["img_container"] = img_container

                legend_box = ui.card().classes('w-full q-ma-sm q-pa-sm shrink-0').style('display:none;')
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

        ui.run_javascript(f"""
            setTimeout(() => {{
                const el = document.getElementById("{window_id}");
                const handle = document.getElementById("{window_id}_handle");
                if (!el || !handle) return;

                makeDraggable("{window_id}", "{window_id}_handle");

                const savePos = () => {{
                    el.dataset.currentLeft = String(parseFloat(el.style.left || '0'));
                    el.dataset.currentTop = String(parseFloat(el.style.top || '0'));
                }};

                if (!el._pyfcsPosObserverInstalled) {{
                    el._pyfcsPosObserverInstalled = true;
                    savePos();

                    const observer = new MutationObserver(savePos);
                    observer.observe(el, {{
                        attributes: true,
                        attributeFilter: ['style']
                    }});

                    el._pyfcsPosObserver = observer;
                }}
            }}, 50);
        """)

    def _remember_window_position(self, window_id: str):
        """
        Read the current DOM position of the floating image window and store it in Python.
        """
        ui.run_javascript(f"""
            (() => {{
                const el = document.getElementById("{window_id}");
                if (!el) return;

                const x = parseFloat(el.style.left || '0');
                const y = parseFloat(el.style.top || '0');

                window.pyfcsWindowPos = window.pyfcsWindowPos || {{}};
                window.pyfcsWindowPos["{window_id}"] = {{ x, y }};
            }})()
        """)

    def _install_position_observer(self, window_id: str):
        """
        Track window position directly on DOM dataset so it can be restored later.
        """
        ui.run_javascript(f"""
            (() => {{
                const el = document.getElementById("{window_id}");
                if (!el) return;

                if (el._pyfcsPosObserverInstalled) return;
                el._pyfcsPosObserverInstalled = true;

                const savePos = () => {{
                    el.dataset.currentLeft = String(parseFloat(el.style.left || '0'));
                    el.dataset.currentTop = String(parseFloat(el.style.top || '0'));
                }};

                savePos();

                const observer = new MutationObserver(savePos);
                observer.observe(el, {{
                    attributes: true,
                    attributeFilter: ['style']
                }});

                el._pyfcsPosObserver = observer;
            }})();
        """)

    def _restore_window_position(self, window_id: str):
        """
        Restore floating position from Python state.
        """
        win = self.image_windows.get(window_id)
        if not win:
            return

        left = int(win.get("x", 20))
        top = int(win.get("y", 20))

        ui.run_javascript(f"""
            (() => {{
                const el = document.getElementById("{window_id}");
                if (!el) return;
                el.style.left = "{left}px";
                el.style.top = "{top}px";
                el.dataset.currentLeft = "{left}";
                el.dataset.currentTop = "{top}";
            }})()
        """)

    def _sync_window_position_from_browser(self, window_id: str):
        """
        Pull the latest browser-stored position into Python state.
        """
        ui.run_javascript(f"""
            (() => {{
                const pos = window.pyfcsWindowPos && window.pyfcsWindowPos["{window_id}"];
                if (!pos) return;
                window.__pyfcs_last_pos = pos;
            }})()
        """)



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
        """
        if self._is_window_mapping_locked(window_id):
            self.cancel_mapping_requested = True

        win = getattr(self, "image_windows", {}).get(window_id)
        if not win:
            return

        ui.run_javascript(f"""
            (() => {{
                const el = document.getElementById("{window_id}");
                if (!el) return;

                if (el._pyfcsPosObserver) {{
                    el._pyfcsPosObserver.disconnect();
                    el._pyfcsPosObserver = null;
                }}
                el._pyfcsPosObserverInstalled = false;
            }})();
        """)

        self._uninstall_legend_follow_behavior(window_id)

        if hasattr(self, "legend_windows") and window_id in self.legend_windows:
            lw = self.legend_windows.pop(window_id)
            if lw:
                lw.delete()

        for k in list(getattr(self, "label_map_cache", {}).keys()):
            if isinstance(k, tuple) and len(k) > 0 and k[0] == window_id:
                del self.label_map_cache[k]

        for k in list(getattr(self, "mapping_all_cache", {}).keys()):
            if isinstance(k, tuple) and len(k) > 0 and k[0] == window_id:
                del self.mapping_all_cache[k]

        for k in list(getattr(self, "proto_map_cache", {}).keys()):
            if isinstance(k, tuple) and len(k) > 0 and k[0] == window_id:
                del self.proto_map_cache[k]

        for k in list(getattr(self, "proto_membership_cache", {}).keys()):
            if isinstance(k, tuple) and len(k) > 0 and k[0] == window_id:
                del self.proto_membership_cache[k]

        if hasattr(self, "cm_cache") and window_id in self.cm_cache:
            del self.cm_cache[window_id]

        if hasattr(self, "scheme_cache") and window_id in self.scheme_cache:
            del self.scheme_cache[window_id]

        if hasattr(self, "ORIGINAL_IMG") and window_id in self.ORIGINAL_IMG:
            del self.ORIGINAL_IMG[window_id]

        if hasattr(self, "MEMBERDEGREE_IMG") and window_id in self.MEMBERDEGREE_IMG:
            del self.MEMBERDEGREE_IMG[window_id]

        temp_dir = win.get("temp_dir")
        if temp_dir and os.path.isdir(temp_dir):
            try:
                shutil.rmtree(temp_dir, ignore_errors=True)
            except Exception:
                pass

        if win.get("card") is not None:
            win["card"].delete()

        del self.image_windows[window_id]



    def _make_preview_data_url_from_path(self, path: str, max_side: int) -> str:
        """
        Build a display preview (PNG data URL) from an image file.

        IMPORTANT:
        - This preview is used as the "original image" shown in the floating window.
        - We intentionally display a reduced version so that:
            * Original Image
            * Color Mapping
            * Color Mapping All
        all share the same base dimensions and do not visually reflow the window.
        """
        img = Image.open(path).convert('RGB')

        w, h = img.size
        if max(w, h) > max_side:
            scale = max_side / float(max(w, h))
            img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)

        return self._np_to_data_url(np.array(img, dtype=np.uint8))
    
    def _on_window_moved(self, window_id: str, e):
        """
        Persist the dragged floating window position in Python state and server-side style.
        """
        try:
            args = e.args or {}
            left = int(round(float(args.get('left', 0))))
            top = int(round(float(args.get('top', 0))))

            win = self.image_windows.get(window_id)
            if not win:
                return

            win["x"] = left
            win["y"] = top

            # IMPORTANT:
            # Update the NiceGUI element style so future UI refreshes keep the new position
            card = win.get("card")
            if card is not None:
                card.style(
                    f'position:absolute; left:{left}px; top:{top}px; z-index:2000; '
                    'width:320px; height:320px; resize:both; overflow:hidden; '
                    'min-width:220px; min-height:220px;'
                )

            # If legend exists, keep it attached after persisting the new position
            if hasattr(self, "legend_windows") and window_id in self.legend_windows:
                self._move_legend_next_to_image(window_id)

        except Exception as ex:
            self.custom_warning("Move Error", str(ex))

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
    async def show_original_image(self, window_id: str):
        """
        Restore and display the original preview image in the floating window.
        """
        if self._is_window_mapping_locked(window_id):
            self.reopen_loading_if_busy("Color Mapping All...")
            ui.notify('This image is currently being processed.', color='warning')
            return

        try:
            win = self.image_windows[window_id]
            win["img"].set_source(win["original_source"])
            win["current_source"] = win["original_source"]

            if "legend_box" in win:
                win["legend_box"].style('display:none;')

            win["legend_visible"] = False
            self.ORIGINAL_IMG[window_id] = False
            self._restore_window_position(window_id)
            ui.notify('Original image restored')

            self._uninstall_legend_follow_behavior(window_id)

            if hasattr(self, "legend_windows") and window_id in self.legend_windows:
                lw = self.legend_windows.pop(window_id)
                if lw:
                    lw.delete()

        except Exception as e:
            self.custom_warning("Display Error", str(e))




    def color_mapping(self, window_id: str):
        """
        Apply a single-prototype membership visualization on the selected image window.
        """
        if getattr(self, 'is_processing_mapping', False):
            self.reopen_loading_if_busy("Color Mapping...")
            ui.notify('A mapping task is already running.', color='warning')
            return

        if not getattr(self, 'COLOR_SPACE', False):
            self.custom_warning("No Color Space", "Load a color space first (.cns or .fcs).")
            return

        labels = list(getattr(self, 'color_matrix', []) or [])
        if not labels:
            self.custom_warning("No Data", "No colors loaded to map.")
            return

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

                self.cancel_mapping_requested = False
                self.is_processing_mapping = True
                self.current_mapping_window_id = window_id

                self.show_loading("Color Mapping...", cancellable=True)

                try:
                    if self.cancel_mapping_requested:
                        ui.notify("Mapping cancelled", color='warning')
                        return

                    if window_id not in self.image_windows:
                        ui.notify("Mapping cancelled because the image window was closed", color='warning')
                        return

                    max_side = self.PREVIEW_MAX_SIDE
                    img_np = self._get_work_image_np(window_id, max_side=max_side)

                    if self.cancel_mapping_requested:
                        ui.notify("Mapping cancelled", color='warning')
                        return

                    if window_id not in self.image_windows:
                        ui.notify("Mapping cancelled because the image window was closed", color='warning')
                        return

                    chosen = sel.value
                    scheme = self.scheme_cache.get(window_id, 'centroid')
                    cache_key = (window_id, chosen, max_side, scheme)

                    if cache_key in self.proto_map_cache:
                        data_url, info_text = self.proto_map_cache[cache_key]

                        win = self.image_windows.get(window_id)
                        if not win:
                            ui.notify("Mapping cancelled because the image window was closed", color='warning')
                            return

                        win["img"].set_source(data_url)
                        win["current_source"] = data_url
                        self._restore_window_position(window_id)

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

                    mkey = (window_id, chosen, max_side)
                    if mkey in self.proto_membership_cache:
                        gray = self.proto_membership_cache[mkey]
                    else:
                        proto_index = self._proto_index_by_label(chosen)
                        gray = await asyncio.to_thread(
                            self._membership_map_for_prototype,
                            img_np,
                            proto_index
                        )

                        if gray is None:
                            ui.notify("Mapping cancelled", color='warning')
                            return

                        self.proto_membership_cache[mkey] = gray

                    if self.cancel_mapping_requested:
                        ui.notify("Mapping cancelled", color='warning')
                        return

                    if window_id not in self.image_windows:
                        ui.notify("Mapping cancelled because the image window was closed", color='warning')
                        return

                    out = np.stack([gray, gray, gray], axis=-1)

                    pct = float((gray > 0).sum()) / float(gray.size) * 100.0
                    info_text = f'Selected: {chosen} — {pct:.2f}% (nonzero membership)'

                    self.modified_image[window_id] = out
                    data_url = self._np_to_data_url(out)

                    win = self.image_windows.get(window_id)
                    if not win:
                        ui.notify("Mapping cancelled because the image window was closed", color='warning')
                        return

                    win["img"].set_source(data_url)
                    win["current_source"] = data_url

                    self._render_legend(
                        window_id,
                        only_labels=[chosen],
                        info=info_text,
                        mode='single',
                    )

                    self.proto_map_cache[cache_key] = (data_url, info_text)

                    self.ORIGINAL_IMG[window_id] = True
                    if hasattr(self, "MEMBERDEGREE"):
                        self.MEMBERDEGREE[window_id] = False

                except Exception as e:
                    self.custom_warning("Processing Error", str(e))

                finally:
                    self.is_processing_mapping = False
                    self.cancel_mapping_requested = False
                    self.current_mapping_window_id = None
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
        Compute the best prototype index for each pixel in an image.

        OPTIMIZATION:
        - Convert RGB -> LAB
        - Quantize LAB using self.LAB_QUANT_STEP
        - Find unique quantized LAB values
        - Compute best prototype index only once per unique color
        - Rebuild the full index map through inverse indices

        Returns
        -------
        np.ndarray
            int32 array (H, W), values:
            - 0..N-1 = prototype index
            - -1     = no prototype found
        """
        self.fuzzy_color_space.precompute_pack()

        img01 = img_uint8.astype(np.float32) / 255.0
        lab_img = skcolor.rgb2lab(img01)
        lab_q = self._quantize_lab(lab_img)

        h, w = lab_q.shape[:2]
        step = float(getattr(self, 'LAB_QUANT_STEP', 0.05))
        lab_int = np.round(lab_q.reshape(-1, 3) / step).astype(np.int32)

        uniq, inv = np.unique(lab_int, axis=0, return_inverse=True)
        best_for_uniq = self._best_idx_for_unique_lab(uniq, progress_callback)

        label_map = best_for_uniq[inv].reshape(h, w).astype(np.int32)
        return label_map




    def color_mapping_all(self, window_id: str):
        """
        Apply full image color mapping using fast prototype-index mapping.
        """
        if not getattr(self, 'COLOR_SPACE', False):
            self.custom_warning("No Color Space", "Load a color space first (.cns or .fcs).")
            return

        if getattr(self, 'is_processing_mapping', False):
            self.reopen_loading_if_busy("Color Mapping All...")
            ui.notify('A mapping task is already running.', color='warning')
            return

        client = context.client

        async def _run():
            self.cancel_mapping_requested = False
            self.is_processing_mapping = True
            self.current_mapping_window_id = window_id

            with client:
                self.show_loading("Color Mapping All...", cancellable=True)

            try:
                self.fuzzy_color_space.precompute_pack()

                if self.cancel_mapping_requested:
                    with client:
                        ui.notify("Mapping cancelled", color='warning')
                    return

                self._prepare_window_preview_cache(window_id)
                win = self.image_windows.get(window_id)
                if not win:
                    with client:
                        ui.notify("Mapping cancelled because the image window was closed", color='warning')
                    return

                preview_hash = win["preview_hash"]
                proto_sig = self._prototype_signature()
                cache_key = (preview_hash, proto_sig)

                if not hasattr(self, "cm_cache"):
                    self.cm_cache = {}
                if window_id not in self.cm_cache:
                    self.cm_cache[window_id] = {}

                def progress(cur, total):
                    if self.cancel_mapping_requested:
                        return
                    with client:
                        self.set_loading_progress(cur / total)

                if cache_key not in self.cm_cache[window_id]:
                    uniq = win["uniq"]
                    inv = win["inv"]
                    h = win["preview_h"]
                    w = win["preview_w"]

                    best_for_uniq = await asyncio.to_thread(self._best_idx_for_unique_lab, uniq, progress)

                    if best_for_uniq is None:
                        with client:
                            ui.notify("Mapping cancelled", color='warning')
                        return

                    if self.cancel_mapping_requested:
                        with client:
                            ui.notify("Mapping cancelled", color='warning')
                        return

                    if window_id not in self.image_windows:
                        with client:
                            ui.notify("Mapping cancelled because the image window was closed", color='warning')
                        return

                    label_map = best_for_uniq[inv].reshape(h, w).astype(np.int32)

                    self.cm_cache[window_id][cache_key] = {
                        "label_map": label_map,
                        "preview_hash": preview_hash,
                        "proto_sig": proto_sig,
                    }
                else:
                    label_map = self.cm_cache[window_id][cache_key]["label_map"]

                if self.cancel_mapping_requested:
                    with client:
                        ui.notify("Mapping cancelled", color='warning')
                    return

                if window_id not in self.image_windows:
                    with client:
                        ui.notify("Mapping cancelled because the image window was closed", color='warning')
                    return

                scheme = self.scheme_cache.get(window_id, 'centroid')
                palette = self._palette_centroid_uint8() if scheme == 'centroid' else self._palette_hsv_uint8()

                out = self._render_recolored_from_index_map(label_map, palette)

                if self.cancel_mapping_requested:
                    with client:
                        ui.notify("Mapping cancelled", color='warning')
                    return

                if window_id not in self.image_windows:
                    with client:
                        ui.notify("Mapping cancelled because the image window was closed", color='warning')
                    return

                self.modified_image[window_id] = out
                data_url = self._np_to_data_url(out)

                with client:
                    win = self.image_windows.get(window_id)
                    if not win:
                        with client:
                            ui.notify("Mapping cancelled because the image window was closed", color='warning')
                        return

                    win["img"].set_source(data_url)
                    win["current_source"] = data_url
                    self._render_legend(window_id)
                    self._restore_window_position(window_id)
                    ui.notify('Color mapping applied (preview)')

            except RuntimeError as e:
                msg = str(e)
                with client:
                    if "cancelled" in msg.lower():
                        ui.notify(msg, color='warning')
                    else:
                        self.custom_warning("Processing Error", msg)

            except Exception as e:
                with client:
                    self.custom_warning("Processing Error", str(e))

            finally:
                self.is_processing_mapping = False
                self.cancel_mapping_requested = False
                self.current_mapping_window_id = None
                with client:
                    self.hide_loading()

        asyncio.create_task(_run())



    def _get_work_image_np(self, window_id: str, max_side: int) -> np.ndarray:
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

    def _render_legend(self, window_id: str, only_labels=None, info=None, mode='all'):
        """
        Render legend in a separate floating window.
        """
        scheme = self.scheme_cache.get(window_id, 'centroid')
        colors = self._label_colors_centroid() if scheme == 'centroid' else self._label_colors_hsv()

        labels = only_labels if only_labels is not None else [p.label for p in self.prototypes]

        # CHANGED:
        # Pass mode through so the separate legend window knows whether to show
        # the Alt. Colors button (only for full Color Mapping All mode).
        self._show_legend_window(window_id, labels, colors, info, mode=mode)

    def toggle_color_scheme(self, window_id: str):
        """
        Toggle the recoloring scheme used in Color Mapping All:
        - 'centroid' (default)
        - 'hsv'

        OPTIMIZATION:
        If a label_map is already cached, only recolor the image.
        No membership recomputation is performed.
        """
        current = self.scheme_cache.get(window_id, 'centroid')
        self.scheme_cache[window_id] = 'hsv' if current == 'centroid' else 'centroid'

        if not hasattr(self, "cm_cache") or window_id not in self.cm_cache:
            self._render_legend(window_id)
            return

        win = self.image_windows.get(window_id)
        if not win or not win.get("preview_cache_ready", False):
            self._render_legend(window_id)
            return

        preview_hash = win["preview_hash"]
        proto_sig = self._prototype_signature()
        cache_key = (preview_hash, proto_sig)

        if cache_key not in self.cm_cache[window_id]:
            # No cached label_map yet -> normal path
            self.color_mapping_all(window_id)
            return

        label_map = self.cm_cache[window_id][cache_key]["label_map"]
        scheme = self.scheme_cache.get(window_id, 'centroid')
        palette = self._palette_centroid_uint8() if scheme == 'centroid' else self._palette_hsv_uint8()

        out = self._render_recolored_from_index_map(label_map, palette)
        self.modified_image[window_id] = out
        data_url = self._np_to_data_url(out)

        win["img"].set_source(data_url)
        win["current_source"] = data_url
        self._restore_window_position(window_id)
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

        OPTIMIZATION:
        - quantize LAB
        - compute membership once per unique color
        - reconstruct final image using inverse mapping
        """
        img = img_np_rgb255.astype(np.float32) / 255.0
        lab = skcolor.rgb2lab(img)
        lab_q = self._quantize_lab(lab)

        h, w, _ = lab_q.shape
        step = float(getattr(self, 'LAB_QUANT_STEP', 0.05))
        lab_int = np.round(lab_q.reshape(-1, 3) / step).astype(np.int32)

        uniq, inv = np.unique(lab_int, axis=0, return_inverse=True)

        total_unique = len(uniq)
        unique_memberships = np.empty(total_unique, dtype=np.float32)

        fcs = self.fuzzy_color_space

        for i, key_int in enumerate(uniq):
            if getattr(self, 'cancel_mapping_requested', False):
                return None
        
            lab_color = key_int.astype(np.float32) * step
            unique_memberships[i] = fcs.calculate_membership_for_prototype(
                np.array([lab_color[0], lab_color[1], lab_color[2]], dtype=float),
                proto_index
            )

            if progress_cb and (i % 200 == 0 or i == total_unique - 1):
                progress_cb(i + 1, total_unique)

        gray = (unique_memberships[inv].reshape(h, w) * 255.0)
        gray = np.clip(gray, 0, 255).astype(np.uint8)
        return gray
    


    def _image_hash(self, arr: np.ndarray) -> str:
        """
        Stable hash for a preview image numpy array.
        Used to cache mapping results by actual preview content.
        """
        return hashlib.blake2b(arr.tobytes(), digest_size=16).hexdigest()



    def _quantize_lab(self, lab_img: np.ndarray) -> np.ndarray:
        """
        Quantize LAB values using a configurable step.

        Example:
        - step=0.05 -> fewer unique colors -> faster mapping
        """
        step = float(getattr(self, 'LAB_QUANT_STEP', 0.05))
        return np.round(lab_img / step) * step



    def _prepare_window_preview_cache(self, window_id: str):
        """
        Precompute and cache all preview data needed by mapping.

        Cached data:
        - preview_np        : HxWx3 uint8 reduced image
        - preview_hash      : stable content hash
        - lab_q             : quantized LAB image
        - lab_int           : flattened integer LAB keys
        - uniq              : unique LAB keys
        - inv               : inverse mapping to rebuild image-sized results
        - h, w              : preview height/width

        This avoids repeating:
        - file load
        - resize
        - RGB -> LAB
        - quantization
        - np.unique
        """
        win = self.image_windows[window_id]

        if win.get("preview_cache_ready", False):
            return

        preview_np = self._get_work_image_np(window_id, max_side=self.PREVIEW_MAX_SIDE)

        img01 = preview_np.astype(np.float32) / 255.0
        lab_img = skcolor.rgb2lab(img01)
        lab_q = self._quantize_lab(lab_img)

        h, w = lab_q.shape[:2]

        # Integer keys derived from quantized LAB.
        # We divide by quant step so equal quantized colors share identical integer keys.
        step = float(getattr(self, 'LAB_QUANT_STEP', 0.05))
        lab_int = np.round(lab_q.reshape(-1, 3) / step).astype(np.int32)

        uniq, inv = np.unique(lab_int, axis=0, return_inverse=True)

        win["preview_np"] = preview_np
        win["preview_hash"] = self._image_hash(preview_np)
        win["lab_q"] = lab_q
        win["lab_int"] = lab_int
        win["uniq"] = uniq
        win["inv"] = inv
        win["preview_h"] = h
        win["preview_w"] = w
        win["preview_cache_ready"] = True



    def _palette_centroid_uint8(self) -> np.ndarray:
        """
        Build a palette array (N,3) uint8 in self.prototypes order using centroid colors.
        """
        palette = []

        for p in self.prototypes:
            label = getattr(p, 'label', None)
            if label is None or not hasattr(self, 'color_data') or label not in self.color_data:
                palette.append(np.array([0, 0, 0], dtype=np.uint8))
                continue

            v = self.color_data[label]
            lab = v.get('positive_prototype', None)
            if lab is None:
                lab = v.get('Color', None)

            if lab is None:
                palette.append(np.array([0, 0, 0], dtype=np.uint8))
                continue

            lab = np.array(lab, dtype=float).reshape(1, 1, 3)
            rgb01 = skcolor.lab2rgb(lab)[0, 0]
            rgb255 = (np.clip(rgb01, 0, 1) * 255).astype(np.uint8)
            palette.append(rgb255)

        return np.stack(palette, axis=0).astype(np.uint8)



    def _palette_hsv_uint8(self) -> np.ndarray:
        """
        Build a palette array (N,3) uint8 in self.prototypes order using HSV colors.
        """
        cmap = plt.get_cmap('hsv', len(self.prototypes))
        palette = []

        for i, p in enumerate(self.prototypes):
            rgb01 = np.array(cmap(i)[:3], dtype=float)
            rgb255 = (np.clip(rgb01, 0, 1) * 255).astype(np.uint8)

            if getattr(p, 'label', '').lower() == 'black':
                rgb255 = np.array([0, 0, 0], dtype=np.uint8)

            palette.append(rgb255)

        return np.stack(palette, axis=0).astype(np.uint8)



    def _prototype_signature(self) -> tuple:
        """
        Signature of current prototype set / order.
        Used to invalidate caches safely when prototypes change.
        """
        return tuple(getattr(p, 'label', '') for p in self.prototypes)



    def _best_idx_for_unique_lab(self, uniq: np.ndarray, progress_callback=None) -> np.ndarray:
        """
        Compute best prototype index for each unique quantized LAB key.
        """
        if not hasattr(self, "best_idx_cache"):
            self.best_idx_cache = {}

        proto_sig = self._prototype_signature()
        step = float(getattr(self, 'LAB_QUANT_STEP', 0.05))

        out = np.empty((uniq.shape[0],), dtype=np.int32)
        total = int(uniq.shape[0])

        for i in range(total):
            if getattr(self, 'cancel_mapping_requested', False):
                return None

            key_int = uniq[i]
            cache_key = (proto_sig, int(key_int[0]), int(key_int[1]), int(key_int[2]))

            best_idx = self.best_idx_cache.get(cache_key)
            if best_idx is None:
                lab_color = key_int.astype(np.float32) * step
                best_idx = int(self.fuzzy_color_space.best_prototype_index_from_lab(
                    (float(lab_color[0]), float(lab_color[1]), float(lab_color[2]))
                ))
                self.best_idx_cache[cache_key] = best_idx

            out[i] = best_idx

            if progress_callback and (i % 200 == 0 or i == total - 1):
                progress_callback(i + 1, total)

        return out
            


    def _render_recolored_from_index_map(self, label_map_idx: np.ndarray, palette_uint8: np.ndarray) -> np.ndarray:
        """
        Fast recolor from prototype index map.

        label_map_idx:
        - int32 array (H,W)
        - values 0..N-1 or -1 for no match

        palette_uint8:
        - (N,3) uint8
        """
        out = np.zeros((label_map_idx.shape[0], label_map_idx.shape[1], 3), dtype=np.uint8)
        valid_mask = (label_map_idx >= 0)
        if np.any(valid_mask):
            out[valid_mask] = palette_uint8[label_map_idx[valid_mask].astype(np.int32)]
        return out

    def _move_legend_next_to_image(self, window_id: str):
        """
        Reposition an existing legend window so it stays attached to the image window.
        """
        if not hasattr(self, "legend_windows") or window_id not in self.legend_windows:
            return

        legend_card = self.legend_windows.get(window_id)
        if not legend_card:
            return

        legend_id = f"{window_id}_legend"

        ui.run_javascript(f"""
            (() => {{
                const img = document.getElementById("{window_id}");
                const legend = document.getElementById("{legend_id}");
                if (!img || !legend) return;

                const imgLeft = parseFloat(img.style.left || '0');
                const imgTop = parseFloat(img.style.top || '0');
                const imgWidth = img.offsetWidth || 0;

                legend.style.left = (imgLeft + imgWidth + 10) + 'px';
                legend.style.top = imgTop + 'px';
            }})()
        """)

    def _install_legend_follow_behavior(self, window_id: str):
        """
        Install a lightweight observer so the legend follows the image window
        when the image is dragged or resized.
        """
        legend_id = f"{window_id}_legend"

        ui.run_javascript(f"""
            (() => {{
                const img = document.getElementById("{window_id}");
                const legend = document.getElementById("{legend_id}");
                if (!img || !legend) return;

                // Clean old observers if they still exist
                if (img._pyfcsLegendMutationObserver) {{
                    img._pyfcsLegendMutationObserver.disconnect();
                    img._pyfcsLegendMutationObserver = null;
                }}
                if (img._pyfcsLegendResizeObserver) {{
                    img._pyfcsLegendResizeObserver.disconnect();
                    img._pyfcsLegendResizeObserver = null;
                }}

                const syncLegend = () => {{
                    const imgLeft = parseFloat(img.style.left || '0');
                    const imgTop = parseFloat(img.style.top || '0');
                    const imgWidth = img.offsetWidth || 0;

                    legend.style.left = (imgLeft + imgWidth + 10) + 'px';
                    legend.style.top = imgTop + 'px';
                }};

                syncLegend();

                const observer = new MutationObserver(syncLegend);
                observer.observe(img, {{
                    attributes: true,
                    attributeFilter: ['style', 'class']
                }});

                const resizeObserver = new ResizeObserver(syncLegend);
                resizeObserver.observe(img);

                img._pyfcsLegendMutationObserver = observer;
                img._pyfcsLegendResizeObserver = resizeObserver;
                img._pyfcsLegendFollowInstalled = true;
            }})();
        """)

    def _uninstall_legend_follow_behavior(self, window_id: str):
        """
        Remove the JS observers that keep the legend attached to the image window.
        This must be called before deleting/recreating the legend window.
        """
        ui.run_javascript(f"""
            (() => {{
                const img = document.getElementById("{window_id}");
                if (!img) return;

                if (img._pyfcsLegendMutationObserver) {{
                    img._pyfcsLegendMutationObserver.disconnect();
                    img._pyfcsLegendMutationObserver = null;
                }}

                if (img._pyfcsLegendResizeObserver) {{
                    img._pyfcsLegendResizeObserver.disconnect();
                    img._pyfcsLegendResizeObserver = null;
                }}

                img._pyfcsLegendFollowInstalled = false;
            }})();
        """)

    def _show_legend_window(self, window_id: str, labels, colors, info=None, mode='all'):
        """
        Show the legend in a separate floating window attached to the image window.
        """
        if not hasattr(self, "legend_windows"):
            self.legend_windows = {}

        # Remove previous follow hooks before recreating the legend.
        self._uninstall_legend_follow_behavior(window_id)

        # Remove existing legend window
        if window_id in self.legend_windows:
            old = self.legend_windows[window_id]
            if old:
                old.delete()
            self.legend_windows.pop(window_id, None)

        img_win = self.image_windows.get(window_id)
        if not img_win:
            return

        legend_id = f"{window_id}_legend"

        with self.image_workspace:
            legend_card = ui.card().props(f'id={legend_id}').style(
                'position:absolute; left:400px; top:100px; '
                'width:160px; max-height:300px; overflow:auto; z-index:2001;'
            )

            with legend_card:
                ui.label("Legend").classes("text-sm font-bold")

                for lab in labels:
                    rgb = colors.get(lab, np.array([0, 0, 0], dtype=np.uint8))
                    hexcol = f'#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}'

                    with ui.row().classes("items-center gap-2"):
                        ui.html(
                            f'<div style="width:16px;height:16px;background:{hexcol};border:1px solid #000;"></div>',
                            sanitize=False,
                        )
                        ui.label(lab).classes("text-xs")

                if info:
                    ui.separator()
                    ui.label(info).classes("text-xs text-gray-600")

                # Show Alt. Colors only for full Color Mapping All mode.
                if mode == 'all':
                    ui.separator()
                    ui.button(
                        'Alt. Colors',
                        on_click=lambda wid=window_id: self.toggle_color_scheme(wid),
                    ).props('dense')

        self.legend_windows[window_id] = legend_card

        # CHANGED:
        # Use plain JS timeout instead of ui.timer so it does not depend on the
        # current NiceGUI slot (which may belong to a deleted legend button).
        ui.run_javascript(f"""
            setTimeout(() => {{
                const img = document.getElementById("{window_id}");
                const legend = document.getElementById("{legend_id}");
                if (!img || !legend) return;

                const imgLeft = parseFloat(img.style.left || '0');
                const imgTop = parseFloat(img.style.top || '0');
                const imgWidth = img.offsetWidth || 0;

                legend.style.left = (imgLeft + imgWidth + 10) + 'px';
                legend.style.top = imgTop + 'px';

                if (img._pyfcsLegendMutationObserver) {{
                    img._pyfcsLegendMutationObserver.disconnect();
                    img._pyfcsLegendMutationObserver = null;
                }}
                if (img._pyfcsLegendResizeObserver) {{
                    img._pyfcsLegendResizeObserver.disconnect();
                    img._pyfcsLegendResizeObserver = null;
                }}

                const syncLegend = () => {{
                    const left = parseFloat(img.style.left || '0');
                    const top = parseFloat(img.style.top || '0');
                    const width = img.offsetWidth || 0;
                    legend.style.left = (left + width + 10) + 'px';
                    legend.style.top = top + 'px';
                }};

                const observer = new MutationObserver(syncLegend);
                observer.observe(img, {{
                    attributes: true,
                    attributeFilter: ['style', 'class']
                }});

                const resizeObserver = new ResizeObserver(syncLegend);
                resizeObserver.observe(img);

                img._pyfcsLegendMutationObserver = observer;
                img._pyfcsLegendResizeObserver = resizeObserver;
                img._pyfcsLegendFollowInstalled = true;
            }}, 60);
        """)














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
        pad = 12
        swatch = 18
        line_h = 24
        max_lines = max(1, len(labels))

        legend_h = pad * 2 + line_h * max_lines
        out = Image.new('RGB', (img.width, img.height + legend_h), (255, 255, 255))
        out.paste(img, (0, 0))

        draw = ImageDraw.Draw(out)

        try:
            font = ImageFont.truetype("arial.ttf", 14)
        except Exception:
            font = ImageFont.load_default()

        y0 = img.height + pad
        x0 = pad

        for i, lab in enumerate(labels):
            rgb = colors.get(lab, np.array([0, 0, 0], dtype=np.uint8))
            fill_rgb = (int(rgb[0]), int(rgb[1]), int(rgb[2]))
            y = y0 + i * line_h

            draw.rectangle(
                [x0, y + 3, x0 + swatch, y + 3 + swatch],
                fill=fill_rgb,
                outline=(0, 0, 0)
            )

            draw.text((x0 + swatch + 10, y + 2), lab, fill=(0, 0, 0), font=font)

        return out
    


    def _build_legend_image(self, window_id: str) -> Image.Image:
        labels = list(getattr(self, 'color_matrix', []) or [])
        scheme = self.scheme_cache.get(window_id, 'centroid')
        colors = self._label_colors_centroid() if scheme == 'centroid' else self._label_colors_hsv()

        pad = 12
        swatch = 18
        line_h = 24
        width = 260
        height = pad * 2 + line_h * max(1, len(labels))

        out = Image.new('RGB', (width, height), (255, 255, 255))
        draw = ImageDraw.Draw(out)

        try:
            font = ImageFont.truetype("arial.ttf", 14)
        except Exception:
            font = ImageFont.load_default()

        y0 = pad
        x0 = pad

        for i, lab in enumerate(labels):
            rgb = colors.get(lab, np.array([0, 0, 0], dtype=np.uint8))
            fill_rgb = (int(rgb[0]), int(rgb[1]), int(rgb[2]))
            y = y0 + i * line_h

            draw.rectangle(
                [x0, y + 3, x0 + swatch, y + 3 + swatch],
                fill=fill_rgb,
                outline=(0, 0, 0)
            )
            draw.text((x0 + swatch + 10, y + 2), lab, fill=(0, 0, 0), font=font)

        return out



    def save_image(self):
        """
        Download an image from one of the currently open floating windows.
        """
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
                    if hasattr(self, 'modified_image') and wid in self.modified_image:
                        arr = self.modified_image[wid]
                        pil = Image.fromarray(arr.astype(np.uint8), mode='RGB')
                    else:
                        path = self.image_windows[wid]["path"]
                        pil = Image.open(path).convert('RGB')

                    has_legend_for_window = (
                        (hasattr(self, 'legend_windows') and wid in self.legend_windows)
                        or (hasattr(self, 'scheme_cache') and wid in self.scheme_cache)
                        or (hasattr(self, 'modified_image') and wid in self.modified_image)
                    )

                    legend_pil = None
                    if include_legend.value and has_legend_for_window:
                        legend_pil = self._build_legend_image(wid)

                    # 1) preparar imagen principal
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

                    base = self.image_windows[wid].get("title", wid).replace(' ', '_')
                    filename = f'{base}.{ext}'

                    # 2) descargar imagen principal
                    ui.download(data, filename=filename, media_type=mime)

                    # 3) descargar leyenda aparte si existe
                    if legend_pil is not None:
                        legend_buf = io.BytesIO()
                        legend_pil.save(legend_buf, format='PNG')
                        legend_data = legend_buf.getvalue()

                        legend_filename = f'{base}_legend.png'
                        ui.download(legend_data, filename=legend_filename, media_type='image/png')
                        ui.notify(f'Downloading: {filename} + {legend_filename}')
                    else:
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
