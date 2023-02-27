import os
from pathlib import Path
import time

import tkinter as tk
from tkinter import messagebox, filedialog, ttk

import cv2
import numpy as np

from iseg_labeler.canvas import CanvasImage
from iseg_labeler.controller import InteractiveController
from iseg_labeler.wrappers import (FocusHorizontalScale, FocusButton, FocusLabelFrame)
import traceback

class ISegApp(ttk.Frame):
    def __init__(self, master, args, model):
        super().__init__(master)
        self.master = master
        master.title("Reviving Iterative Training with Mask Guidance for Interactive Segmentation")
        master.withdraw()
        master.update_idletasks()

        self._filedialog_initialdir = os.getcwd()
        self._load_image_initialdir = None  # falls back to _filedialog_initialdir if None
        self._load_mask_initialdir = None    # falls back to _load_image_initialdir if None
        self._save_mask_initialdir = None   # falls back to _save_mask_initialdir if None

        self._mask_mode = False # In mask mode, only the resultant mask is displayed

        self._debug = args.debug
        self._timing = args.timing

        x = (master.winfo_screenwidth() - master.winfo_reqwidth()) / 2
        y = (master.winfo_screenheight() - master.winfo_reqheight()) / 2
        master.geometry(f"+{int(x)}+{int(y)}")
        self.pack(fill="both", expand=True)

        self.brush_modes = ['Foreground', 'Background', 'Erase Brushstrokes']

        self.controller = InteractiveController(model, args.device,
                                                predictor_params={'brs_mode': 'FocalClick'},
                                                # predictor_params={'brs_mode': 'NoBRS'},
                                                update_image_callback=self._update_image)

        self.click_update_size_slider = None

        self._init_state()
        self._add_menu()
        self._add_canvas()
        self._add_buttons()

        self._prev_set_alpha = self.state['alpha_blend'].get()

        master.bind('<KeyPress>', self._keypad_minus_plus)

        master.bind('<space>', lambda event: self.controller.finish_object())
        master.bind('t', lambda event: self._toggle_brush())
        master.bind('<Shift-T>', lambda event: self._show_hide_mask())
        master.bind('1', lambda event: self._change_brush_mode("Foreground"))
        master.bind('2', lambda event: self._change_brush_mode("Background"))
        master.bind('3', lambda event: self._change_brush_mode("Erase Brushstrokes"))
        master.bind('4', lambda event: self._set_alpha(1.0))
        master.bind('5', lambda event: self._increase_alpha())
        master.bind('6', lambda event: self._reduce_alpha())
        master.bind('7', lambda event: self._set_alpha(0.0))

        master.bind('a', lambda event: self.controller.partially_finish_object())

        master.bind('<Shift-MouseWheel>', self._size_wheel)   # Windows/Mac scroll up/down
        master.bind('<Shift-4>', self._size_wheel)  # Linux scroll up
        master.bind('<Shift-5>', self._size_wheel)  # Linux scroll down

        self.state['zoomin_params']['skip_clicks'].trace(mode='w',
                                                         callback=self._update_zoom_in)
        self.state['zoomin_params']['target_size'].trace(mode='w',
                                                         callback=self._update_zoom_in)
        self.state['zoomin_params']['expansion_ratio'].trace(mode='w',
                                                             callback=self._update_zoom_in)

        self.brush_value = 1

        self.image_on_canvas = None
        self._reset_predictor()

        self._image_path = None
        self._mask_path = None

    def _keypad_minus_plus(self, event):
        if self.image_on_canvas is None or event.state not in (4, 5):
            return
        if event.keysym in ("KP_Add", "plus", "equal"):
            # control-plus or shift-control-plus
            self.image_on_canvas.keypad_minus_plus(event, "in")
        elif event.keysym in ("KP_Subtract", "minus", "underscore"):
            # control-minus or shift-control-minus
            self.image_on_canvas.keypad_minus_plus(event, "out")

    def _set_alpha(self, alpha):
        cur_alpha = self.state['alpha_blend'].get()
        if alpha == cur_alpha:
            alpha = self._prev_set_alpha        # toggle back to previous value
        else:
            self._prev_set_alpha = cur_alpha    # set new value and save previous value
        self.state['alpha_blend'].set(alpha)
        self._update_image()

    def _increase_alpha(self):
        cur_alpha = self.state['alpha_blend'].get()
        self.state['alpha_blend'].set(min(1.0, cur_alpha + 0.1))
        self._update_image()

    def _reduce_alpha(self):
        cur_alpha = self.state['alpha_blend'].get()
        self.state['alpha_blend'].set(max(0.0, cur_alpha - 0.1))
        self._update_image()

    def _init_state(self):
        self.state = {
            'zoomin_params': {
                'use_zoom_in': tk.BooleanVar(value=True),
                'fixed_crop': tk.BooleanVar(value=True),
                'skip_clicks': tk.IntVar(value=-1),
                'target_size': tk.IntVar(value=256),
                'expansion_ratio': tk.DoubleVar(value=1.0)
            },

            'prob_thresh': tk.DoubleVar(value=0.5),
            'brush_size': tk.IntVar(value=10), # Initialize brush size to 10
            'alpha_blend': tk.DoubleVar(value=0.5),
            'brush_on': tk.BooleanVar(value=False),
            'click_radius': tk.IntVar(value=3),
            'brush_mode': tk.StringVar(value="Foreground"),
        }

    def _add_menu(self):
        self.menubar = FocusLabelFrame(self, bd=1)
        self.menubar.pack(side=tk.TOP, fill='x')

        button = FocusButton(self.menubar, text='Load image', command=self._load_image_callback)
        button.pack(side=tk.LEFT)
        self.save_mask_btn = FocusButton(self.menubar, text='Save mask',
                                         command=self._save_mask_callback)
        self.save_mask_btn.pack(side=tk.LEFT)
        self.save_mask_btn.configure(state=tk.DISABLED)

        self.load_mask_btn = FocusButton(self.menubar, text='Load mask',
                                         command=self._load_mask_callback)
        self.load_mask_btn.pack(side=tk.LEFT)
        self.load_mask_btn.configure(state=tk.DISABLED)

        button = FocusButton(self.menubar, text='About', command=self._about_callback)
        button.pack(side=tk.LEFT)
        button = FocusButton(self.menubar, text='Exit', command=self.master.quit)
        button.pack(side=tk.LEFT)

    def _add_canvas(self):
        self.canvas_frame = FocusLabelFrame(self, text="Image")
        self.canvas_frame.rowconfigure(0, weight=1)
        self.canvas_frame.columnconfigure(0, weight=1)

        self.canvas = tk.Canvas(self.canvas_frame, highlightthickness=0, cursor="tcross",
                                width=400, height=400)
        self.canvas.grid(row=0, column=0, sticky='nswe', padx=5, pady=5)

        self.image_on_canvas = None
        self.canvas_frame.pack(side=tk.LEFT, fill="both", expand=True, padx=5, pady=5)

    def _add_buttons(self):
        self.control_frame = FocusLabelFrame(self, text="Controls")
        self.control_frame.pack(side=tk.TOP, fill='x', padx=5, pady=5)
        master = self.control_frame

        self.clicks_options_frame = FocusLabelFrame(master, text="Clicks management")
        self.clicks_options_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=3)
        self.undo_click_button = FocusButton(self.clicks_options_frame, text='Undo click',
                                             bg='#ffe599', fg='black', width=10, height=2,
                                             state=tk.DISABLED, command=self.controller.undo_click)
        self.undo_click_button.pack(side=tk.LEFT, fill=tk.X, padx=10, pady=3)
        self.reset_clicks_button = FocusButton(self.clicks_options_frame, text='Reset clicks',
                                               bg='#ea9999', fg='black', width=10, height=2,
                                               state=tk.DISABLED, command=self._reset_last_object)
        self.reset_clicks_button.pack(side=tk.LEFT, fill=tk.X, padx=10, pady=3)

        self.prob_thresh_frame = FocusLabelFrame(master, text="Predictions threshold")
        self.prob_thresh_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=3)
        FocusHorizontalScale(self.prob_thresh_frame, from_=0.0, to=1.0,
                             command=self._update_prob_thresh, variable=self.state['prob_thresh']
                            ).pack(padx=10)

        self.brush_size_frame = FocusLabelFrame(master, text="Brush Size")
        self.brush_size_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=3)
        FocusHorizontalScale(self.brush_size_frame, from_=1, to=20, resolution=1,
                             command=self._update_brush_size, variable=self.state['brush_size']
                            ).pack(padx=10)

        self.click_update_size_frame = FocusLabelFrame(master, text="Size of click update region")
        self.click_update_size_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=3)
        self.click_update_size_slider = FocusHorizontalScale(
            self.click_update_size_frame, from_=32, to=1992, resolution=32,
            command=self._update_click_area, variable=self.state['zoomin_params']['target_size']
        )
        self.click_update_size_slider.pack(padx=10)

        self.alpha_blend_frame = FocusLabelFrame(master, text="Selection Transparency")
        self.alpha_blend_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=3)
        FocusHorizontalScale(self.alpha_blend_frame, from_=0.0, to=1.0, resolution=0.1,
                             command=self._update_blend_alpha, variable=self.state['alpha_blend']
                            ).pack(padx=10, anchor=tk.CENTER)

        self.click_radius_frame = FocusLabelFrame(master, text="Click indicator size")
        self.click_radius_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=3)
        FocusHorizontalScale(self.click_radius_frame, from_=0, to=7, resolution=1,
                             command=self._update_click_radius, variable=self.state['click_radius']
                            ).pack(padx=10, anchor=tk.CENTER)

        self.toggle_brush_button = FocusButton(self.clicks_options_frame, text='Toggle Brush',
                                               bg='#9497e3', fg='black', width=10, height=2,
                                               state=tk.NORMAL, command=self._toggle_brush)
        self.toggle_brush_button.pack(side=tk.LEFT, fill=tk.X, padx=10, pady=3)

        self.brush_options_frame = FocusLabelFrame(master, text="Brush Selection Mode")
        self.brush_options_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=3)
        self.menu = tk.OptionMenu(self.brush_options_frame, self.state['brush_mode'],
                                  *self.brush_modes, command=self._change_brush_mode)
        self.menu.config(width=20, bg='white', fg='black')
        self.menu.grid(rowspan=2, column=0, padx=10)
        self.brush_options_frame.columnconfigure((0, 1), weight=1)

        self.other_options_frame = FocusLabelFrame(master, text="Display Controls")
        self.other_options_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=3)
        self.show_hide_mask_button = FocusButton(self.other_options_frame, text='Show/Hide Mask',
                                             bg='#ffe599', fg='black', width=12, height=2,
                                             state=tk.NORMAL, command=self._show_hide_mask)
        self.show_hide_mask_button.pack(side=tk.LEFT, fill=tk.X, padx=10, pady=3)

    def _get_filedialog_initialdir(self, stage: str):
        initialdir = self._filedialog_initialdir

        if stage == 'load_image':
            if self._load_image_initialdir is not None:
                initialdir = self._load_image_initialdir
        elif stage == 'load_mask':
            if self._load_mask_initialdir is not None:
                initialdir = self._load_mask_initialdir
            else:
                initialdir = self._get_filedialog_initialdir('load_image')
        elif stage == 'save_mask':
            if self._save_mask_initialdir is not None:
                initialdir = self._save_mask_initialdir
            else:
                initialdir = self._get_filedialog_initialdir('load_mask')

        return initialdir

    def _show_hide_mask(self):
        self._mask_mode = not self._mask_mode
        self._update_image()

    def _load_image_callback(self):
        self.menubar.focus_set()
        if self._check_entry(self):
            filename = filedialog.askopenfilename(
                parent=self.master,
                initialdir=self._get_filedialog_initialdir('load_image'),
                filetypes=[
                    ("Images", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"),
                    ("All files", "*.*"),
                ], title="Choose an image"
            )

            if len(filename) > 0:
                self._load_image_initialdir = os.path.dirname(filename)
                self._load_mask_initialdir = None
                self._save_mask_initialdir = None
                self.master.title(os.path.basename(filename))   # set window title
                self._image_path = Path(filename)
                self._mask_path = None
                image = cv2.cvtColor(cv2.imread(filename, 0), cv2.COLOR_GRAY2RGB)
                self.controller.set_image(image)
                self.save_mask_btn.configure(state=tk.NORMAL)
                self.load_mask_btn.configure(state=tk.NORMAL)
                max_click_area = min(min(image.shape[:2]), 1992)
                self.click_update_size_slider.destroy()
                self.click_update_size_slider = FocusHorizontalScale(
                    self.click_update_size_frame, from_=32, to=max_click_area,
                    resolution=32, command=self._update_click_area,
                    variable=self.state['zoomin_params']['target_size']
                )
                self.click_update_size_slider.pack(padx=10)

    def _save_mask_callback(self):
        self.menubar.focus_set()
        if self._check_entry(self):
            mask = self.controller.result_mask
            if mask is None:
                return

            if self._mask_path is not None:
                initial_filename = Path(self._mask_path).name
            elif self._image_path is not None:
                initial_filename = f"{self._image_path.stem}_mask{self._image_path.suffix}"
            else:
                initial_filename = "mask.png"

            filename = filedialog.asksaveasfilename(
                parent=self.master,
                initialdir=self._get_filedialog_initialdir('save_mask'),
                initialfile=initial_filename,
                filetypes=[
                    ("Images", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"),
                    ("All files", "*.*"),
                ], title="Save the current mask as...")

            if len(filename) > 0:
                self._save_mask_initialdir = os.path.dirname(filename)
                mask = self._get_mask_vis(mask)
                cv2.imwrite(filename, mask)

    def _load_mask_callback(self):
        if not self.controller.net.with_prev_mask:
            messagebox.showwarning("Warning",
                                   "The current model doesn't support loading external masks. "
                                   "Please use ITER-M models for that purpose.")
            return

        self.menubar.focus_set()
        if self._check_entry(self):
            filename = filedialog.askopenfilename(
                parent=self.master,
                initialdir=self._get_filedialog_initialdir('load_mask'),
                filetypes=[
                    ("Images", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"),
                    ("All files", "*.*"),
                ], title="Chose an image"
            )

            if len(filename) > 0:
                self._load_mask_initialdir = os.path.dirname(filename)
                self._mask_path = Path(filename)
                mask = cv2.imread(filename)[:, :, 0] > 127
                self.controller.set_mask(mask)
                self._update_image()

    def _get_mask_vis(self, mask=None):
        if mask is None:
            mask = self.controller.result_mask
        if mask is None:
            return

        brush = self.controller.brush
        if brush is not None:
            brush_mask = self.controller.brush.get_brush_mask()[0]
            mask[brush_mask<2] = brush_mask[brush_mask<2]
        if mask.max() < 256:
            mask = mask.astype(np.uint8)
            mask_max = mask.max()
            if mask_max > 0:
                mask *= 255 // mask_max

        return mask

    def _about_callback(self):
        self.menubar.focus_set()

        text = [
            "Developed by:",
            "K.Sofiiuk and I. Petrov",
            "The MIT License, 2021"
        ]

        messagebox.showinfo("About Demo", '\n'.join(text))

    def _reset_last_object(self):
        self.state['alpha_blend'].set(0.5)
        self.state['prob_thresh'].set(0.5)
        self.controller.reset_last_object()

    def _update_prob_thresh(self, value):
        if self.controller.is_incomplete_mask:
            self.controller.prob_thresh = self.state['prob_thresh'].get()
            self._update_image()

    def _size_wheel(self, event):
        # Respond to Linux (event.num) or Windows (event.delta) wheel event
        if event.num == 5 or event.delta == -120 or event.delta == 1:  # scroll down, zoom out, smaller
            self._decrement_size()
        if event.num == 4 or event.delta == 120 or event.delta == -1:  # scroll up, zoom in, bigger
            self._increment_size()

    def _increment_size(self):
        if self.state['brush_on'].get():
            # increment brush size
            cur_size = self.state['brush_size'].get()
            if cur_size < 20:
                self.state['brush_size'].set(cur_size + 1)
        else:
            # increment click update size
            cur_size = self.state['zoomin_params']['target_size'].get()
            if cur_size < 512:
                self.state['zoomin_params']['target_size'].set(cur_size + 32)

    def _decrement_size(self):
        if self.state['brush_on'].get():
            # decrement brush size
            cur_size = self.state['brush_size'].get()
            if cur_size > 1:
                self.state['brush_size'].set(cur_size - 1)
        else:
            # decrement click update size
            cur_size = self.state['zoomin_params']['target_size'].get()
            if cur_size > 32:
                self.state['zoomin_params']['target_size'].set(cur_size - 32)

    def _update_brush_size(self, value):
        self.state['brush_size'] = tk.IntVar(value=value)

    def _update_blend_alpha(self, value):
        self._update_image()

    def _update_click_area(self, value):
        self.state['zoomin_params']['target_size'] = tk.IntVar(value=value)
        self._update_zoom_in()

    def _update_click_radius(self, *args):
        if self.image_on_canvas is None:
            return

        self._update_image()

    def _change_brush_mode(self, value):
        self.state['brush_mode'].set(value)
        self._set_brush_value()

    def _set_brush_value(self):
        brush_mode = self.state['brush_mode'].get()
        if brush_mode == "Foreground":
            self.brush_value = 1
        elif brush_mode == 'Background':
            self.brush_value = 0
        elif brush_mode == 'Erase Brushstrokes':
            self.brush_value = 2

    def _toggle_brush(self):
        brush_on = not self.state['brush_on'].get()
        self.state['brush_on'].set(brush_on)
        if not brush_on:
            self.image_on_canvas.canvas.delete('brush-pointer')
            self.image_on_canvas.canvas.config(cursor="tcross")
        else:
            self.image_on_canvas.canvas.config(cursor="none")

    def _reset_predictor(self, *args, **kwargs):
        zoomin_params = {
            'skip_clicks': self.state['zoomin_params']['skip_clicks'].get(),
            'target_size': self.state['zoomin_params']['target_size'].get(),
            'expansion_ratio': self.state['zoomin_params']['expansion_ratio'].get()
        }
        prob_thresh = self.state['prob_thresh'].get()
        predictor_params = {
            'brs_mode': 'NoBRS',
            'prob_thresh': prob_thresh,
            'zoom_in_params': zoomin_params,
            'net_clicks_limit': None,
            'lbfgs_params': {'maxfun': 20}
        }
        self.controller.reset_predictor(predictor_params)

    def _update_zoom_in(self, *args, **kwargs):
        zoom_in_params = {
            'skip_clicks': self.state['zoomin_params']['skip_clicks'].get(),
            'target_size': self.state['zoomin_params']['target_size'].get(),
            'expansion_ratio': self.state['zoomin_params']['expansion_ratio'].get()
        }
        self.controller.update_zoom_in(zoom_in_params)

    def _click_callback(self, is_positive, x, y):
        if self.state['brush_on'].get():
            self._brush_callback(x, y)
            return

        self.canvas.focus_set()

        if self.image_on_canvas is None:
            messagebox.showwarning("Warning", "Please load an image first")
            return

        if self._check_entry(self):
            self.controller.add_click(x, y, is_positive)

    def _brush_callback(self, x, y):
        if not self.state['brush_on'].get():
            return

        self.canvas.focus_set()

        if self.image_on_canvas is None:
            messagebox.showwarning("Warning", "Please load an image first")
            return

        if self._check_entry(self):
            self.controller.draw_brush(x, y, self.brush_value,
                                       self.state['brush_size'].get())

    def _show_brush_pointer_callback(self, x, y):
        if self.state['brush_on'].get():
            self.image_on_canvas.show_brush_pointer(x, y, self.state['brush_size'].get())

    def _end_brushstroke_callback(self):
        self.controller.end_brushstroke()

    def _update_image(self, reset_canvas=False):
        if (self.image_on_canvas is not None and self.controller.brush is not None
            and self.controller.brush.current_brushstroke is not None):
            canvas_img = self.image_on_canvas.get_full_canvas_image()
        else:
            canvas_img = None

        if self._timing:
            start = time.perf_counter_ns()

        image = self.controller.get_visualization(alpha_blend=self.state['alpha_blend'].get(),
                                                  click_radius=self.state['click_radius'].get(),
                                                  canvas_img=canvas_img,
                                                  brush=self.controller.brush)

        if self._timing:
            end = time.perf_counter_ns()
            print(f"get_visualization() took {(end - start) / 1e6} ms")

        if self.image_on_canvas is None:
            self.image_on_canvas = CanvasImage(self.canvas_frame, self.canvas)
            self.image_on_canvas.register_click_callback(self._click_callback)
            self.image_on_canvas.register_brush_callback(self._brush_callback)
            self.image_on_canvas.register_end_brushstroke_callback(self._end_brushstroke_callback)
            self.image_on_canvas.register_brush_pointer_callback(self._show_brush_pointer_callback)

        self._set_click_dependent_widgets_state()

        if image is not None:
            if self._timing:
                start = time.perf_counter_ns()
            if self._mask_mode:
                image = cv2.cvtColor(self._get_mask_vis(), cv2.COLOR_GRAY2RGB)
            self.image_on_canvas.reload_image(image, reset_canvas)
            if self._timing:
                end = time.perf_counter_ns()
                print(f"reload_image() took {(end - start) / 1e6} ms")
        # print("print update image trace")
        # traceback.print_stack()

    def _set_click_dependent_widgets_state(self):
        after_1st_click_state = tk.NORMAL if self.controller.is_incomplete_mask else tk.DISABLED

        self.undo_click_button.configure(state=after_1st_click_state)
        self.reset_clicks_button.configure(state=after_1st_click_state)

    def _check_entry(self, widget):
        all_checked = True
        if widget.winfo_children is not None:
            for w in widget.winfo_children():
                all_checked = all_checked and self._check_entry(w)

        if getattr(widget, "_check_bounds", None) is not None:
            all_checked = all_checked and widget._check_bounds(widget.get(), '-1')

        return all_checked
