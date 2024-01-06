from components.components import *
from utils.constants import *
from logic.state import *
from logic.logic import *

# Code start here 👇

app = Window("Photoshop", 1000, 650)


# Space Component


def Spacer(space):
    return Container(width=space, height=space, style=ContainerStyle(background_color=None))


BUTTON_STYLE_PRIMARY = ButtonStyle(
    background_color='#1D267D',
    color="#FFFFFF",
    hover_background_color="#0C134F",
    hover_color="#FFFFFF",
    padding=(20, 5),
)

BUTTON_STYLE_SECONDARY = ButtonStyle(
    background_color='#E76161',
    color='#FFFFFF',
    hover_background_color='#B04759',
    hover_color='#FFFFFF',
    padding=(20, 5)
)

BUTTON_STYLE_LAYERS = ButtonStyle(
    background_color='#E76161',
    color='#FFFFFF',
    hover_background_color='#B04759',
    hover_color='#FFFFFF',
    padding=(40, 5),
)


# Canvas Component


def Canvas():
    return Picture(img=WORKSPACE_IMAGE, width=CANVAS_WIDTH, heigth=CANVAS_HEIGHT, state=canvas_color)


# Select target object window


def SelectObjectPopup():
    window = PopupWindow(app.root, "Select Object")

    def GetObjectButtonList():
        object_button = []
        for i, object in enumerate(object_list.state):
            print(object)
            object_button.append(
                Row(
                    children=[
                        Spacer(space=20),
                        Column(
                            children=[
                                Button(f"Layer {object.name}",
                                       on_click=lambda: select_object(i, object, window.root), style=BUTTON_STYLE_LAYERS),
                                Spacer(space=5),
                            ]
                        ),
                        Spacer(space=20),
                    ]
                )
            )
        return object_button

    window.render(
        Column(
            children=[
                Spacer(space=20),
                Column(
                    children=GetObjectButtonList()
                ),
                Spacer(space=15),
            ]
        )
    )


# Translate window


def TranslatePopup():
    window = PopupWindow(app.root, "Translate")
    x = 0
    y = 0

    def set_x(new_x):
        nonlocal x
        x = int(new_x)

    def set_y(new_y):
        nonlocal y
        y = int(new_y)

    def handleApplyPress():
        transform_translate(x, y)
        window.root.destroy()

    window.render(
        Column(
            children=[
                Spacer(space=20),
                Row(
                    children=[
                        Spacer(space=20),
                        Text("X: "),
                        Spacer(space=5),
                        Input(on_change=lambda e: set_x(e)),
                        Spacer(space=20),
                    ]
                ),
                Spacer(space=10),
                Row(
                    children=[
                        Spacer(space=20),
                        Text("Y: "),
                        Spacer(space=5),
                        Input(on_change=lambda e: set_y(e)),
                        Spacer(space=20),
                    ]
                ),
                Spacer(space=10),
                Button("Apply", on_click=lambda: handleApplyPress()),
                Spacer(space=20),
            ]
        ))


# Shear window


def ShearPopup():
    window = PopupWindow(app.root, "Shear")
    x = 0
    y = 0

    def set_x(new_x):
        nonlocal x
        x = float(new_x)

    def set_y(new_y):
        nonlocal y
        y = float(new_y)

    def handleApplyPress():
        transform_shear(x, y)
        window.root.destroy()

    window.render(
        Column(
            children=[
                Spacer(space=20),
                Row(
                    children=[
                        Spacer(space=20),
                        Text("X: "),
                        Spacer(space=5),
                        Input(on_change=lambda e: set_x(e)),
                        Spacer(space=20),
                    ]
                ),
                Spacer(space=10),
                Row(
                    children=[
                        Spacer(space=20),
                        Text("Y: "),
                        Spacer(space=5),
                        Input(on_change=lambda e: set_y(e)),
                        Spacer(space=20),
                    ]
                ),
                Spacer(space=10),
                Button("Apply", on_click=lambda: handleApplyPress()),
                Spacer(space=20),
            ]
        ))


# Perspective window


def PerspectivePopup():
    global selected_object
    window = PopupWindow(app.root, "Perspective")
    x1 = 0
    y1 = 0
    x2 = 0
    y2 = 0
    x3 = 0
    y3 = 0
    x4 = 0
    y4 = 0

    def set_x1(new_x1):
        nonlocal x1
        x1 = int(new_x1)

    def set_y1(new_y1):
        nonlocal y1
        y1 = int(new_y1)

    def set_x2(new_x2):
        nonlocal x2
        x2 = int(new_x2)

    def set_y2(new_y2):
        nonlocal y2
        y2 = int(new_y2)

    def set_x3(new_x3):
        nonlocal x3
        x3 = int(new_x3)

    def set_y3(new_y3):
        nonlocal y3
        y3 = int(new_y3)

    def set_x4(new_x4):
        nonlocal x4
        x4 = int(new_x4)

    def set_y4(new_y4):
        nonlocal y4
        y4 = int(new_y4)

    def handleApplyPress():
        original = [
            [x1, y1],
            [x2, y2],
            [x3, y3],
            [x4, y4]
        ]
        new = [
            [0, 0],
            [CANVAS_WIDTH, 0],
            [0, CANVAS_HEIGHT],
            [CANVAS_WIDTH, CANVAS_HEIGHT]
        ]
        transform_perspective(original, new)
        window.root.destroy()

    def handleSelectOnCV():
        x1, y1, x2, y2, x3, y3, x4, y4 = select_corrdinate()
        set_x1(x1)
        set_y1(y1)
        set_x2(x2)
        set_y2(y2)
        set_x3(x3)
        set_y3(y3)
        set_x4(x4)
        set_y4(y4)
        handleApplyPress()

    window.render(
        Column(
            children=[
                Spacer(space=20),
                Row(
                    children=[
                        Spacer(space=20),
                        Column(
                            children=[
                                Row(
                                    children=[
                                        Text("X1: "),
                                        Spacer(space=5),
                                        Input(on_change=lambda e: set_x1(e)),
                                    ]
                                ),
                                Spacer(space=5),
                                Row(
                                    children=[
                                        Text("Y1: "),
                                        Spacer(space=5),
                                        Input(on_change=lambda e: set_y1(e)),
                                    ]
                                )
                            ]
                        ),
                        Spacer(space=10),
                        Column(
                            children=[
                                Row(
                                    children=[
                                        Text("X2: "),
                                        Spacer(space=5),
                                        Input(on_change=lambda e: set_x2(e)),
                                    ]
                                ),
                                Spacer(space=5),
                                Row(
                                    children=[
                                        Text("Y2: "),
                                        Spacer(space=5),
                                        Input(on_change=lambda e: set_y2(e)),
                                    ]
                                )
                            ]
                        ),
                        Spacer(space=20),
                    ]
                ),
                Spacer(space=20),
                Row(
                    children=[
                        Spacer(space=20),
                        Column(
                            children=[
                                Row(
                                    children=[
                                        Text("X3: "),
                                        Spacer(space=5),
                                        Input(on_change=lambda e: set_x3(e)),
                                    ]
                                ),
                                Spacer(space=5),
                                Row(
                                    children=[
                                        Text("Y3: "),
                                        Spacer(space=5),
                                        Input(on_change=lambda e: set_y3(e)),
                                    ]
                                )
                            ]
                        ),
                        Spacer(space=10),
                        Column(
                            children=[
                                Row(
                                    children=[
                                        Text("X4: "),
                                        Spacer(space=5),
                                        Input(on_change=lambda e: set_x4(e)),
                                    ]
                                ),
                                Spacer(space=5),
                                Row(
                                    children=[
                                        Text("Y4: "),
                                        Spacer(space=5),
                                        Input(on_change=lambda e: set_y4(e)),
                                    ]
                                )
                            ]
                        ),
                        Spacer(space=20),
                    ]
                ),
                Spacer(space=10),
                Row(
                    children=[
                        Spacer(space=20),
                        Button("Select on Canvas",
                               on_click=lambda: handleSelectOnCV()),
                        Spacer(space=20),
                        Button("Apply", on_click=lambda: handleApplyPress()),
                        Spacer(space=20),
                    ]
                ),
                Spacer(space=20),
            ]
        ))


# Single Input Popup


def SingleInputPopup(title, label, on_apply):
    window = PopupWindow(app.root, title)
    value = 0

    def set_value(new_value):
        nonlocal value
        value = float(new_value)

    def handleApplyPress():
        on_apply(value)
        window.root.destroy()

    window.render(
        Column(
            children=[
                Spacer(space=20),
                Row(
                    children=[
                        Spacer(space=20),
                        Text(label),
                        Spacer(space=5),
                        Input(on_change=lambda e: set_value(e)),
                        Spacer(space=20),
                    ]
                ),
                Spacer(space=10),
                Button("Apply", on_click=lambda: handleApplyPress()),
                Spacer(space=20),
            ]
        ))


# Select transform window


def SelectTransformMethod():
    window = PopupWindow(app.root, "Select Effects")

    def handleTranslatePress():
        window.root.destroy()
        TranslatePopup()

    def handleRotatePress():
        window.root.destroy()
        SingleInputPopup("Rotate", "Degree", transform_rotate)

    def handleScalePress():
        window.root.destroy()
        SingleInputPopup("Scale", "Scale", transform_scale)

    def handleShearPress():
        window.root.destroy()
        ShearPopup()

    def handlePerspectivePress():
        window.root.destroy()
        PerspectivePopup()

    def handleBrightnessPress():
        window.root.destroy()
        SingleInputPopup("Brightness", "Value", brightness)

    def handleContrastPress():
        window.root.destroy()
        SingleInputPopup("Contrast", "Value", contrast)

    def handleGammaCorrectionPress():
        window.root.destroy()
        SingleInputPopup("Gamma Correction", "Value", gamma_correction)

    def handleAutoBrightnessPress():
        window.root.destroy()
        auto_brightness()

    def handleMedianPress():
        window.root.destroy()
        SingleInputPopup("Median", "Kernel size", median_blur)

    def handleMeanPress():
        window.root.destroy()
        SingleInputPopup("Mean", "Kernel size", mean_blur)

    def handleGaussianPress():
        window.root.destroy()
        SingleInputPopup("Gaussian", "Kernel size", gaussian_blur)

    def handleEdgePress():
        window.root.destroy()
        edge_detection()

    def handleHistogramPress():
        window.root.destroy()
        show_histogram()

    def handleHistogramEqualizationPress():
        window.root.destroy()
        histogram_equalization()

    def handleHarrisCornerDetectorPress():
        window.root.destroy()
        SingleInputPopup("Harris Corner Detector",
                         "Threshold", harris_corner_detector)

    def handleCannyEdgeDetectorPress():
        window.root.destroy()
        SingleInputPopup("Canny Edge Detector",
                         "Threshold", canny_edge_detector)

    def handleHoughLineDetectorPress():
        window.root.destroy()
        SingleInputPopup("Hough Line Detector",
                         "Threshold", hough_line_detector)

    def handleHOGPress():
        window.root.destroy()
        SingleInputPopup("Histogram of Oriented Gradients",
                         "Bins", show_histogram_hog)

    def handleFaceDetectorPress():
        window.root.destroy()
        face_detection()

    def handleSnakeSegmentationPress():
        window.root.destroy()
        snake_segmentation()

    def handleWatershedSegmentationPress():
        window.root.destroy()
        SingleInputPopup("Watershed Segmentation", "Threshold",
                         watershed_segmentation)

    def handleKMeansSegmentationPress():
        window.root.destroy()
        SingleInputPopup("KMeans Segmentation", "K", kmeans_segmentation)

    def handleMeanShiftSegmentationPress():
        window.root.destroy()
        mean_shift_segmentation()

    window.render(
        Row(
            children=[
                Column(
                    justify=Justify.START,
                    children=[
                        Spacer(space=20),
                        Column(
                            children=[
                                Text('Transformations'),
                                Spacer(space=5),
                                Button("Translate",
                                       on_click=lambda: handleTranslatePress(), style=BUTTON_STYLE_PRIMARY),
                                Spacer(space=5),
                                Button(
                                    "Rotate", on_click=lambda: handleRotatePress(), style=BUTTON_STYLE_PRIMARY),
                                Spacer(space=5),
                                Button(
                                    "Scale", on_click=lambda: handleScalePress(), style=BUTTON_STYLE_PRIMARY),
                                Spacer(space=5),
                                Button(
                                    "Shear", on_click=lambda: handleShearPress(), style=BUTTON_STYLE_PRIMARY),
                                Spacer(space=5),
                                Button("Perspective",
                                       on_click=lambda: handlePerspectivePress(), style=BUTTON_STYLE_PRIMARY),
                            ]
                        ),
                        Spacer(space=20),
                    ]
                ),
                Spacer(space=20),
                Column(
                    justify=Justify.START,
                    children=[
                        Spacer(space=20),
                        Column(
                            children=[
                                Text('Color Adjustment'),
                                Spacer(space=5),
                                Button('Brightness',
                                       on_click=lambda: handleBrightnessPress(), style=BUTTON_STYLE_PRIMARY),
                                Spacer(space=5),
                                Button(
                                    'Contrast', on_click=lambda: handleContrastPress(), style=BUTTON_STYLE_PRIMARY),
                                Spacer(space=5),
                                Button('Gamma Correction',
                                       on_click=lambda: handleGammaCorrectionPress(), style=BUTTON_STYLE_PRIMARY),
                                Spacer(space=5),
                                Button('Auto Brightness',
                                       on_click=lambda: handleAutoBrightnessPress(), style=BUTTON_STYLE_PRIMARY),
                                Spacer(space=5),
                                Button('Histogram',
                                       on_click=lambda: handleHistogramPress(), style=BUTTON_STYLE_PRIMARY),
                            ]
                        ),
                        Spacer(space=20),
                    ]
                ),
                Spacer(space=20),
                Column(
                    justify=Justify.START,
                    children=[
                        Spacer(space=20),
                        Column(
                            children=[
                                Text('Filters'),
                                Spacer(space=5),
                                Button(
                                    'Median', on_click=lambda: handleMedianPress(), style=BUTTON_STYLE_PRIMARY),
                                Spacer(space=5),
                                Button(
                                    'Mean', on_click=lambda: handleMeanPress(), style=BUTTON_STYLE_PRIMARY),
                                Spacer(space=5),
                                Button(
                                    'Gaussian', on_click=lambda: handleGaussianPress(), style=BUTTON_STYLE_PRIMARY),
                                Spacer(space=5),
                                Button('Edge extractor',
                                       on_click=lambda: handleEdgePress(), style=BUTTON_STYLE_PRIMARY),
                                Spacer(space=5),
                                Button('Histogram Equalization',
                                       on_click=lambda: handleHistogramEqualizationPress(), style=BUTTON_STYLE_PRIMARY),
                            ]
                        ),
                        Spacer(space=20),
                    ]
                ),
                Spacer(space=20),
                Column(
                    justify=Justify.START,
                    children=[
                        Spacer(space=20),
                        Column(
                            children=[
                                Text('Detectors'),
                                Spacer(space=5),
                                Button(
                                    'Harris Corner Detector', on_click=lambda: handleHarrisCornerDetectorPress(), style=BUTTON_STYLE_PRIMARY),
                                Spacer(space=5),
                                Button(
                                    'Canny Edge Detector', on_click=lambda: handleCannyEdgeDetectorPress(), style=BUTTON_STYLE_PRIMARY),
                                Spacer(space=5),
                                Button(
                                    'Hough Line Detector', on_click=lambda: handleHoughLineDetectorPress(), style=BUTTON_STYLE_PRIMARY),
                                Spacer(space=5),
                                Button(
                                    'Histogram of Oriented Gradients', on_click=lambda: handleHOGPress(), style=BUTTON_STYLE_PRIMARY),
                                Spacer(space=5),
                                Button(
                                    'Face Detector', on_click=lambda: handleFaceDetectorPress(), style=BUTTON_STYLE_PRIMARY),
                            ]
                        ),
                        Spacer(space=20),
                    ]
                ),
                Spacer(space=20),
                Column(
                    justify=Justify.START,
                    children=[
                        Spacer(space=20),
                        Column(
                            children=[
                                Text('Segmentation'),
                                Spacer(space=5),
                                Button(
                                    'Snake Segmentation', on_click=lambda: handleSnakeSegmentationPress(), style=BUTTON_STYLE_PRIMARY),
                                Spacer(space=5),
                                Button(
                                    'Watershed Segmentation', on_click=lambda: handleWatershedSegmentationPress(), style=BUTTON_STYLE_PRIMARY),
                                Spacer(space=5),
                                Button(
                                    'KMean Segmentation', on_click=lambda: handleKMeansSegmentationPress(), style=BUTTON_STYLE_PRIMARY),
                                Spacer(space=5),
                                Button(
                                    'Mean Shift Segmentation', on_click=lambda: handleMeanShiftSegmentationPress(), style=BUTTON_STYLE_PRIMARY),
                            ]
                        ),
                        Spacer(space=20),
                    ]
                ),
            ]
        )
    )


def NewItemPopup():
    window = PopupWindow(app.root, "New Item")

    def handleShapePress():
        window.root.destroy()
        create_shape()

    def handleImagePress():
        window.root.destroy()
        import_image()

    def handlePanoramaImagePress():
        window.root.destroy()
        import_panorama_image()

    window.render(
        Column(
            justify=Justify.START,
            children=[
                Spacer(space=20),
                Column(
                    children=[
                        Button(
                            'Shape', on_click=lambda: handleShapePress(), style=BUTTON_STYLE_PRIMARY),
                        Spacer(space=5),
                        Button(
                            'Image', on_click=lambda: handleImagePress(), style=BUTTON_STYLE_PRIMARY),
                        Spacer(space=5),
                        Button(
                            'Panorama Image', on_click=lambda: handlePanoramaImagePress(), style=BUTTON_STYLE_PRIMARY),
                        Spacer(space=5),
                    ]
                ),
                Spacer(space=20),
            ]
        ),
    )


# Main screen
app.render(
    Column(
        children=[
            Canvas(),
            Spacer(space=20),
            Row(
                children=[
                    Spacer(space=20),
                    Button('New Item', on_click=lambda: NewItemPopup(),
                           style=BUTTON_STYLE_PRIMARY),
                    Spacer(space=20),
                    Button('Select Object', on_click=lambda: SelectObjectPopup(),
                           style=BUTTON_STYLE_SECONDARY),
                    Spacer(space=10),
                    Text(f'Object: {selected_object_text.state}',
                         state=selected_object_text),
                    Spacer(space=20),
                    Button('Edit Console', on_click=lambda: SelectTransformMethod(),
                           style=BUTTON_STYLE_SECONDARY, is_disabled=selected_object_edit_btn.state, state=selected_object_edit_btn),
                    Spacer(space=20),
                    Button('Export', on_click=lambda: export_image(),
                           style=BUTTON_STYLE_SECONDARY)
                ]
            )
        ]
    )
)


# Code end here 👆
