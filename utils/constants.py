import enum
from typing import TypedDict
import numpy as np
import cv2 as cv

CANVAS_WIDTH = 1000
CANVAS_HEIGHT = 562


WHITE_IMAGE = np.ones((CANVAS_HEIGHT, CANVAS_WIDTH, 3), dtype=np.uint8) * 0
WORKSPACE_IMAGE = WHITE_IMAGE.copy()


class Justify(enum.Enum):
    START = 'start'
    END = 'end'
    STRETCH = 'stretch'


class Align(enum.Enum):
    CENTER = 'center'
    STRETCH = 'stretch'


class ButtonStyle(TypedDict):
    background_color: str
    color: str
    hover_background_color: str
    hover_color: str
    border: int
    font_family: str
    font_size: int
    padding: tuple[int, int]


class TextStyle(TypedDict):
    color: str
    font_family: str
    font_size: int


class ContainerStyle(TypedDict):
    background_color: str
    border: int


class InputStyle(TypedDict):
    background_color: str
    color: str
    focus_color: str
    font_family: str
    font_size: int
    border: int


class InputType(enum.Enum):
    TEXT = 'text'
    PASSWORD = 'password'


class CheckboxStyle(TypedDict):
    color: str
    background_color: str
    focus_color: str
    focus_background_color: str
    font_family: str
    font_size: int
    border: int
    padding: tuple[int, int]


### Default styles ###
DEFAULT_BUTTON_STYLE = ButtonStyle(
    background_color='#ffffff',
    color='#000000',
    hover_background_color='#eeeeee',
    hover_color='#000000',
    border=0,
    font_family='Arial',
    font_size=12,
    padding=(0, 0)
)

DEFAULT_TEXT_STYLE = TextStyle(
    color='#000000',
    font_family='Arial',
    font_size=12,
)

DEFAULT_INPUT_STYLE = InputStyle(
    background_color='#ffffff',
    color='#000000',
    focus_color='#000000',
    font_family='Arial',
    font_size=12,
    border=0,
)

DEFAULT_CHECKBOX_STYLE = CheckboxStyle(
    color='#000000',
    background_color='#ffffff',
    focus_color='#000000',
    focus_background_color='#ffffff',
    font_family='Arial',
    font_size=12,
    border=0,
    padding=(5, 2)
)

DEFAULT_CONTAINER_STYLE = ContainerStyle(
    background_color='#ffffff',
    border=0
)
