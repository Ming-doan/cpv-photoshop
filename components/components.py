import tkinter as tk
from PIL import ImageTk, Image
import numpy as np
from utils.constants import *
from typing import Any, Union
from logic.state import State

### Components ###

TK_IMG_GLOBAL_VAR = None


class Component:
    def get(self, parent: Union[tk.Tk, tk.Frame]):
        raise NotImplementedError


# Primitive Components
# __init__: component constructor
# get(parent_frame): returns a tkinter widget

class Button(Component):
    def __init__(self, text, on_click, style=DEFAULT_BUTTON_STYLE, is_disabled=False, state: State = None):
        self.text = text
        self.on_click = on_click
        self.style = style
        self.is_disabled = is_disabled
        self.state = state

    def get(self, parent):
        _style = {
            **DEFAULT_BUTTON_STYLE,
            **self.style
        }
        _button_style = {
            'bg': _style['background_color'],
            'fg': _style['color'],
            'activebackground': _style['hover_background_color'],
            'activeforeground': _style['hover_color'],
            'bd': _style['border'],
            'font': (_style['font_family'], _style['font_size']),
            'padx': _style['padding'][0],
            'pady': _style['padding'][1],
        }

        _button_state = tk.DISABLED if self.is_disabled else tk.NORMAL

        _widget = tk.Button(
            parent, text=self.text, command=self.on_click, state=_button_state, **_button_style)

        if self.state is not None:
            self.state.register_func(_widget.configure)

        return _widget


class Text(Component):
    def __init__(self, text, style=DEFAULT_TEXT_STYLE, state: State = None):
        self.text = text
        self.style = style
        self.state = state

    def get(self, parent):
        _style = {
            **DEFAULT_TEXT_STYLE,
            **self.style
        }
        _text_style = {
            'fg': _style['color'],
            'font': (_style['font_family'], _style['font_size']),
        }

        _widget = tk.Label(parent, text=self.text, **_text_style)

        if self.state is not None:
            self.state.register_func(_widget.configure)

        return _widget


class Input(Component):
    def __init__(self, on_change, style=DEFAULT_INPUT_STYLE, type=InputType.TEXT, is_disabled=False, state: State = None):
        self.on_change = on_change
        self.style = style
        self.type = type
        self.is_disabled = is_disabled
        self.state = state

    def get(self, parent):
        _style = {
            **DEFAULT_INPUT_STYLE,
            **self.style
        }
        _text_style = {
            'bg': _style['background_color'],
            'fg': _style['color'],
            'font': (_style['font_family'], _style['font_size']),
            'bd': _style['border'],
        }

        _show_type = '*' if self.type == InputType.PASSWORD else None

        _input_state = tk.DISABLED if self.is_disabled else tk.NORMAL

        _input = tk.Entry(parent, show=_show_type,
                          state=_input_state, **_text_style)
        _input.bind('<KeyRelease>', lambda _: self.on_change(_input.get()))

        if self.state is not None:
            self.state.register_func(_input.configure)

        return _input


class Checkbox(Component):
    def __init__(self, text, on_change, is_check=False, style=DEFAULT_CHECKBOX_STYLE, is_disabled=False):
        self.text = text
        self.on_change = on_change
        self.is_check = is_check
        self.style = style
        self.is_disabled = is_disabled

    def get(self, parent):
        _style = {
            **DEFAULT_CHECKBOX_STYLE,
            **self.style
        }
        _checkbox_style = {
            'bg': _style['background_color'],
            'fg': _style['color'],
            'activebackground': _style['focus_background_color'],
            'activeforeground': _style['focus_color'],
            'bd': _style['border'],
            'font': (_style['font_family'], _style['font_size']),
            'padx': _style['padding'][0],
            'pady': _style['padding'][1],
        }

        _checkbox_state = tk.DISABLED if self.is_disabled else tk.NORMAL

        _checkbox = tk.Checkbutton(
            parent, text=self.text, command=self.on_change, state=_checkbox_state, **_checkbox_style)
        _checkbox.select() if self.is_check else _checkbox.deselect()
        return _checkbox


class Picture(Component):
    def __init__(self, img: np.ndarray, width=CANVAS_WIDTH, heigth=CANVAS_HEIGHT, state: State = None):
        self.img = img
        self.width = width
        self.height = heigth
        self.state = state

    def get(self, parent):
        global TK_IMG_GLOBAL_VAR
        TK_IMG_GLOBAL_VAR = ImageTk.PhotoImage(image=Image.fromarray(self.img))
        # Create widget
        _widget = tk.Label(parent, image=TK_IMG_GLOBAL_VAR,
                           width=self.width, height=self.height)

        if self.state is not None:
            self.state.register_func(_widget.configure)

        return _widget


# Composite Components
# __init__: component constructor
# get(parent_frame): renders child components and returns a frame variable

class Container(Component):
    def __init__(self, width=0, height=0, style=DEFAULT_CONTAINER_STYLE, child: Component = None, state: State = None):
        self.width = width
        self.height = height
        self.style = style
        self.child = child
        self.state = state

    def get(self, parent):
        _style = {
            **DEFAULT_CONTAINER_STYLE,
            **self.style
        }
        _container_style = {
            'bg': _style['background_color'],
            'bd': _style['border']
        }

        _frame = tk.Frame(parent, width=self.width,
                          height=self.height, **_container_style)

        if self.child is not None:
            _child = self.child.get(_frame)
            _child.pack(fill=tk.BOTH)

        if self.state is not None:
            self.state.register_func(_frame.configure)

        return _frame


class Row(Component):
    def __init__(self, children: list[Component], justify=Justify.START, align=Align.CENTER, state: State = None):
        self.children = children
        self.justify = justify
        self.align = align
        self.state = state

    def get(self, parent):
        _frame = tk.Frame(parent)

        _children = [child.get(_frame) for child in self.children]

        _range = reversed(
            range(len(_children))) if self.justify == Justify.END else range(len(_children))

        for i in _range:
            _child = _children[i]
            if isinstance(self.children[i], Column) or isinstance(self.children[i], Row) or self.justify == Justify.STRETCH or self.align == Align.STRETCH:
                _fill_param = tk.BOTH
            else:
                _fill_param = tk.NONE

            if self.justify == Justify.END:
                _side_param = tk.RIGHT
            else:
                _side_param = tk.LEFT

            if self.justify == Justify.STRETCH:
                _expand_param = True
            else:
                _expand_param = False

            _child.pack(side=_side_param, fill=_fill_param,
                        expand=_expand_param)

        if self.state is not None:
            self.state.register_func(_frame.configure)

        return _frame


class Column(Component):
    def __init__(self, children: list[Component], justify=Justify.START, align=Align.CENTER, state: State = None):
        self.children = children
        self.justify = justify
        self.align = align
        self.state = state

    def get(self, parent):
        _frame = tk.Frame(parent)

        _children = [child.get(_frame) for child in self.children]

        _range = reversed(
            range(len(_children))) if self.justify == Justify.END else range(len(_children))

        for i in _range:
            _child = _children[i]
            if isinstance(self.children[i], Row) or isinstance(self.children[i], Column) or self.justify == Justify.STRETCH or self.align == Align.STRETCH:
                _fill_param = tk.BOTH
            else:
                _fill_param = tk.NONE

            if self.justify == Justify.END:
                _side_param = tk.BOTTOM
            else:
                _side_param = tk.TOP

            if self.justify == Justify.STRETCH:
                _expand_param = True
            else:
                _expand_param = False

            _child.pack(side=_side_param, fill=_fill_param,
                        expand=_expand_param)

        if self.state is not None:
            self.state.register_func(_frame.configure)

        return _frame


### Main Widgets ###


class Window:
    def __init__(self, title, width=None, height=None):
        self.title = title
        self.width = width
        self.height = height
        self.root = tk.Tk()

    def render(self, child):
        # Initialize Tk app
        self.root.title(self.title)
        if self.width and self.height:
            self.root.geometry(f"{self.width}x{self.height}")

        # Render child
        if isinstance(child, Component):
            _child = child.get(self.root)
        else:
            _child = child.render(self.root)

        # Pack child
        _child.pack(fill=tk.BOTH)

        # Start Tk app
        self.root.mainloop()


class PopupWindow:
    def __init__(self, parent: tk.Tk, title, width=None, height=None):
        self.parent = parent
        self.title = title
        self.width = width
        self.height = height
        self.root = tk.Toplevel(self.parent)

    def render(self, child: Component = None):
        # Create window
        self.root.title(self.title)
        if self.width and self.height:
            self.root.geometry(f"{self.width}x{self.height}")

        # Render child
        if child is not None:
            _child = child.get(self.root)
            _child.pack(fill=tk.BOTH)
