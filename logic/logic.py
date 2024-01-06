from logic.state import *
from utils.constants import *
from models.object import *
from logic.state import *
from PIL import ImageTk, Image
import tkinter as tk
from tkinter import filedialog


TK_IMG_GLOBAL_VAR = None
SHAPE_IDX = 0
IMAGE_IDX = 0


### Canvas logic ###

# Create image Object, get image, return Object
def create_shape_object(img):
    global SHAPE_IDX
    return Object(f"Shape {SHAPE_IDX}", img, ObjectType.SHAPE)


# Create shape Object, get image, return Object
def create_image_object(img):
    global IMAGE_IDX
    return Object(f"Image {IMAGE_IDX}", img, ObjectType.IMAGE)


# Put img2 on img1, get 2 images, position, return img1
def stack_img(first_image: np.ndarray, second_image: np.ndarray, pos: tuple[int, int] = (0, 0)):
    x, y = pos
    # Define ROI
    rows, cols, _ = second_image.shape
    bg_rows, bg_cols, _ = first_image.shape
    c_rows, c_cols = rows+x, cols+y
    roi = first_image[x:c_rows, y:c_cols]
    # Cut img
    img_x = min(rows, bg_rows-x)
    img_y = min(cols, bg_cols-y)
    second_image = second_image[0:img_x, 0:img_y]
    # Create mask and it inverse
    img_gray = cv.cvtColor(second_image, cv.COLOR_BGR2GRAY)
    _, mask = cv.threshold(img_gray, 10, 255, cv.THRESH_BINARY)
    mask_inv = cv.bitwise_not(mask)
    # Black out area of ROI
    _bg = cv.bitwise_and(roi, roi, mask=mask_inv)
    # Take only region of first img
    img_fg = cv.bitwise_and(second_image, second_image, mask=mask)
    # Place 2 img
    dst = cv.add(_bg, img_fg, dtype=cv.CV_32F)
    first_image[x:c_rows, y:c_cols] = dst
    return first_image


# def stack_img(first_image: np.ndarray, second_image: np.ndarray):
#     x1, y1 = first_image.shape[:2]
#     x2, y2 = second_image.shape[:2]
#     x = min(x1, x2)
#     y = min(y1, y2)
#     first_image[:x, :y] = second_image[:x, :y]
#     return first_image


# Append object in canvas, get object, return None
def append_canvas(object: Object):
    global WORKSPACE_IMAGE, TK_IMG_GLOBAL_VAR
    _working_img = object.img.copy()

    # Apply filters
    _working_img = object.filter.apply_kernel_filter(_working_img)

    # Append object in transparant image
    # If image shape is larger than workspace, change _transparent to image shape
    if _working_img.shape[0] > WORKSPACE_IMAGE.shape[0] or _working_img.shape[1] > WORKSPACE_IMAGE.shape[1]:
        _transparent = np.full(
            (_working_img.shape[0], _working_img.shape[1], 4), (0, 0, 0, 0), dtype=np.uint8)
    else:
        _transparent = np.full(
            (WORKSPACE_IMAGE.shape[0], WORKSPACE_IMAGE.shape[1], 4), (0, 0, 0, 0), dtype=np.uint8)

    # Overlay object on transparant image
    _working_img = stack_img(_transparent, _working_img)

    # Get transform of object
    object.transform.get_transform()
    _working_img = object.transform.apply_transform(_working_img)
    object.transform.get_perspective()
    _working_img = object.transform.apply_perspective(_working_img)

    # Stack image
    _working_img = stack_img(WORKSPACE_IMAGE, _working_img[:, :, :3])

    # Get object in canvas
    WORKSPACE_IMAGE = _working_img
    TK_IMG_GLOBAL_VAR = ImageTk.PhotoImage(
        image=Image.fromarray(WORKSPACE_IMAGE))
    state.get_state('canvas_color').config(image=TK_IMG_GLOBAL_VAR)


# Create shape Object, get color, return None
def create_shape(color=(255, 0, 0)):
    global WORKSPACE_IMAGE, object_list, SHAPE_IDX
    pt1 = (0, 0)
    pt2 = (0, 0)
    completed = False
    drawing = False
    # Mouse callback

    def draw_rectangle(event, x, y, flags, param):
        nonlocal pt1, pt2, completed, drawing
        if event == cv.EVENT_LBUTTONDOWN:
            pt1 = (x, y)
            drawing = True
        elif event == cv.EVENT_LBUTTONUP:
            pt2 = (x, y)
            completed = True
        elif event == cv.EVENT_MOUSEMOVE:
            pt2 = (x, y)

    # Popup CV
    cv.namedWindow('Draw Rectangle')
    cv.setMouseCallback('Draw Rectangle', draw_rectangle)
    while True:
        _image = WORKSPACE_IMAGE.copy()
        if drawing:
            cv.rectangle(_image, pt1, pt2, color, -1)
        cv.imshow('Draw Rectangle', cv.cvtColor(_image, cv.COLOR_RGB2BGR))
        cv.waitKey(1)
        if completed:
            cv.rectangle(_image, pt1, pt2, color, -1)
            break
    cv.destroyAllWindows()

    # Create object
    w = pt2[0] - pt1[0]
    h = pt2[1] - pt1[1]
    c = (color[0], color[1], color[2], 255)
    new_image = np.full((h, w, 4), c, dtype=np.uint8)
    SHAPE_IDX += 1
    new_object = create_shape_object(new_image)
    new_object.transform.add_step(Translate(pt1[0], pt1[1]))

    # Set state for new object
    object_list.set(lambda prev: [*prev, new_object])

    # Modify image
    append_canvas(new_object)


# Create image Object, get None, return None
def import_image():
    global object_list, IMAGE_IDX
    # Open file dialog
    file_path = filedialog.askopenfilename()
    _image = cv.imread(file_path)

    # Change color space
    _image = cv.cvtColor(_image, cv.COLOR_BGR2RGBA)

    # Create object
    new_object = create_image_object(_image)

    # Set state for new object
    IMAGE_IDX += 1
    object_list.set(lambda prev: [*prev, new_object])

    # Modify image
    append_canvas(new_object)


# Create objects on layer, get object index, object instantce, window, return None
def select_object(id: int, object: Object, window):
    global object_list, selected_object

    # Set state for current object
    selected_object.set(lambda _: object_list.state[id])

    print(object_list.state)

    # Update UI
    state.get_state('selected_object_text').config(text=f"Layer {object.name}")
    state.get_state('selected_object_edit_btn').config(state=tk.NORMAL)

    # Destroy window
    window.destroy()


# Re-render canvas, get None, return None
def refresh_canvas():
    global object_list, WORKSPACE_IMAGE, TK_IMG_GLOBAL_VAR

    # Reset WORKSPACE_IMAGE
    WORKSPACE_IMAGE = WHITE_IMAGE.copy()

    TK_IMG_GLOBAL_VAR = ImageTk.PhotoImage(
        image=Image.fromarray(WORKSPACE_IMAGE))
    state.get_state('canvas_color').config(image=TK_IMG_GLOBAL_VAR)

    # Re-render object
    for object in object_list.state:
        append_canvas(object)


### Transform logic ###


def transform_translate(x, y):
    global selected_object
    selected_object.state.transform.add_step(Translate(x, y))
    refresh_canvas()


def transform_rotate(angle):
    global selected_object
    selected_object.state.transform.add_step(Rotation(angle))
    refresh_canvas()


def transform_scale(s):
    global selected_object
    selected_object.state.transform.add_step(Scale(s))
    refresh_canvas()


def transform_shear(sx, sy):
    global selected_object
    selected_object.state.transform.add_step(Shear(sx, sy))
    refresh_canvas()


def transform_perspective(original, new):
    global selected_object
    selected_object.state.transform.add_step(Perspective(original, new))
    refresh_canvas()


def select_corrdinate():
    selected_corrdinate = []
    completed = False

    def select_corrdinate_callback(event, x, y, flags, param):
        nonlocal selected_corrdinate, completed
        if event == cv.EVENT_LBUTTONDOWN:
            selected_corrdinate.append((x, y))
            if len(selected_corrdinate) == 4:
                completed = True

    cv.namedWindow('Select Coordinate')
    cv.setMouseCallback('Select Coordinate', select_corrdinate_callback)
    while True:
        cv.imshow('Select Coordinate', cv.cvtColor(
            WORKSPACE_IMAGE, cv.COLOR_RGB2BGR))
        cv.waitKey(1)
        if completed:
            break
    cv.destroyAllWindows()

    x1, y1 = selected_corrdinate[0]
    x2, y2 = selected_corrdinate[1]
    x3, y3 = selected_corrdinate[2]
    x4, y4 = selected_corrdinate[3]

    return x1, y1, x2, y2, x3, y3, x4, y4


### Filter logic ###


def brightness(brightness):
    global selected_object
    selected_object.state.filter.add_step(Brightness(brightness))
    refresh_canvas()


def contrast(contrast):
    global selected_object
    selected_object.state.filter.add_step(Contrast(contrast))
    refresh_canvas()


def gamma_correction(gamma):
    global selected_object
    selected_object.state.filter.add_step(GammaCorrection(gamma))
    refresh_canvas()


def auto_brightness():
    global selected_object
    selected_object.state.filter.add_step(AutoBrightness())
    refresh_canvas()


def median_blur(size):
    global selected_object
    selected_object.state.filter.add_step(MedianFilter(int(size)))
    refresh_canvas()


def mean_blur(size):
    global selected_object
    selected_object.state.filter.add_step(MeanFilter(int(size)))
    refresh_canvas()


def gaussian_blur(size):
    global selected_object
    selected_object.state.filter.add_step(GaussianFilter(int(size)))
    refresh_canvas()


def edge_detection():
    global selected_object
    selected_object.state.filter.add_step(EdgeExtractor())
    refresh_canvas()


def histogram_equalization():
    global selected_object
    selected_object.state.filter.add_step(HistogramEqualization())
    refresh_canvas()


def show_histogram():
    global selected_object
    _img = apply_filters(selected_object.state)
    histogram_title = f"Histogram of {selected_object.state.name}"
    histogram(_img, histogram_title)


### Feature detection logic ###


def harris_corner_detector(threshold):
    global selected_object
    selected_object.state.filter.add_step(HarrisCornerDetector(threshold))
    refresh_canvas()


def canny_edge_detector(threshold):
    global selected_object
    selected_object.state.filter.add_step(
        CannyEdgeDetector(0, threshold))
    refresh_canvas()


def hough_line_detector(threshold):
    global selected_object
    selected_object.state.filter.add_step(HoughLineDetector(int(threshold)))
    refresh_canvas()


def apply_filters(object: Object = None):
    global selected_object
    return object.filter.apply_kernel_filter(object.img)


def show_histogram_hog(bins):
    global selected_object
    _img = apply_filters(selected_object.state)
    histogram_title = f"Histogram of {selected_object.state.name}"
    histogram_hog(_img, bins=int(bins), title=histogram_title)


def face_detection():
    global selected_object
    selected_object.state.filter.add_step(FaceDetection())
    refresh_canvas()


### Segmentation logic ###


def snake_segmentation(color=(0, 255, 255)):
    global selected_object, WORKSPACE_IMAGE
    pt1 = (0, 0)
    pt2 = (0, 0)
    completed = False
    drawing = False

    def draw_rectangle(event, x, y, flags, param):
        nonlocal pt1, pt2, completed, drawing
        if event == cv.EVENT_LBUTTONDOWN:
            pt1 = (x, y)
            drawing = True
        elif event == cv.EVENT_LBUTTONUP:
            pt2 = (x, y)
            completed = True
        elif event == cv.EVENT_MOUSEMOVE:
            pt2 = (x, y)

    cv.namedWindow('Draw Rectangle')
    cv.setMouseCallback('Draw Rectangle', draw_rectangle)
    while True:
        _image = WORKSPACE_IMAGE.copy()
        if drawing:
            cv.rectangle(_image, pt1, pt2, color, 2)
        cv.imshow('Draw Rectangle', cv.cvtColor(_image, cv.COLOR_RGB2BGR))
        cv.waitKey(1)
        if completed:
            cv.rectangle(_image, pt1, pt2, color, -1)
            break
    cv.destroyAllWindows()

    selected_object.state.filter.add_step(
        SnakeSegmentation(topleft=pt1, bottomright=pt2))
    refresh_canvas()


def watershed_segmentation(threshold):
    global selected_object
    selected_object.state.filter.add_step(WatershedSegmentation(threshold))
    refresh_canvas()


def kmeans_segmentation(k):
    global selected_object
    selected_object.state.filter.add_step(KMeansSegmentation(int(k)))
    refresh_canvas()


def mean_shift_segmentation():
    global selected_object
    selected_object.state.filter.add_step(MeanShiftSegmentation())
    refresh_canvas()


### Panorama logic ###


def stitching_right(anchor_img: np.ndarray, pano_img: np.ndarray):
    # Copy image
    _anchor_img = anchor_img.copy()
    _pano_img = pano_img.copy()
    # Convert to gray
    anchor_gray = cv.cvtColor(_anchor_img, cv.COLOR_BGR2GRAY)
    pano_gray = cv.cvtColor(_pano_img, cv.COLOR_BGR2GRAY)
    # Init SIFT
    sift = cv.SIFT_create()
    # Find keypoints and descriptors
    kp1, des1 = sift.detectAndCompute(pano_gray, None)
    kp2, des2 = sift.detectAndCompute(anchor_gray, None)
    # BFMatcher
    bf = cv.BFMatcher()
    matches = bf.match(des1, des2)
    # Find homography
    src_pts = np.float32(
        [kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32(
        [kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    H, _ = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
    # Warp perspective
    panorama = cv.warpPerspective(
        _pano_img, H, (_pano_img.shape[1] + _anchor_img.shape[1], _pano_img.shape[0]))
    # Merge image
    panorama = stack_img(panorama, _anchor_img)
    # Crop black border
    return panorama


def stitching_left(anchor_img: np.ndarray, pano_img: np.ndarray):
    # Copy image
    _pano_img = anchor_img.copy()
    _anchor_img = pano_img.copy()
    # Convert to gray
    anchor_gray = cv.cvtColor(_anchor_img, cv.COLOR_BGR2GRAY)
    pano_gray = cv.cvtColor(_pano_img, cv.COLOR_BGR2GRAY)
    # Init SIFT
    sift = cv.SIFT_create()
    # Find keypoints and descriptors
    kp1, des1 = sift.detectAndCompute(anchor_gray, None)
    kp2, des2 = sift.detectAndCompute(pano_gray, None)
    # BFMatcher
    bf = cv.BFMatcher()
    matches = bf.match(des1, des2)
    # Find homography
    src_pts = np.float32(
        [kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32(
        [kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    T = np.array([[1, 0, _pano_img.shape[1]], [0, 1, 0], [0, 0, 1]])
    H, _ = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
    # Warp perspective
    panorama = cv.warpPerspective(_anchor_img, np.matmul(
        T, H), (_pano_img.shape[1] + _anchor_img.shape[1], _pano_img.shape[0]))
    # Merge image
    panorama = stack_img(panorama, _pano_img, pos=(0, _anchor_img.shape[1]))
    # Crop black border
    return panorama


def panorama(imgs: list[str]):
    def resize(img):
        return cv.resize(img, (0, 0), fx=0.5, fy=0.5)
    # Get center image
    center_idx = len(imgs) // 2
    center_img = resize(cv.imread(imgs[center_idx]))
    center_img_right = center_img.copy()
    center_img_left = center_img.copy()
    # Create panorama from center to right
    for i in range(center_idx+1, len(imgs)):
        center_img_right = stitching_right(
            center_img_right, resize(cv.imread(imgs[i])))
    # Create panorama from center to left
    for i in range(center_idx-1, -1, -1):
        center_img_left = stitching_left(
            center_img_left, resize(cv.imread(imgs[i])))
    # Merge 2 panorama
    panorama = np.hstack(
        (center_img_left, center_img_right[:, center_img.shape[1]:]))
    return panorama


def import_panorama_image():
    global object_list, IMAGE_IDX
    # Open file dialog
    file_paths = filedialog.askopenfilenames()

    # Create panorama
    _image = panorama(file_paths)

    # Change color space
    _image = cv.cvtColor(_image, cv.COLOR_BGR2RGBA)

    # Create object
    new_object = create_image_object(_image)

    # Set state for new object
    IMAGE_IDX += 1
    object_list.set(lambda prev: [*prev, new_object])

    # Modify image
    append_canvas(new_object)


### Export logic ###

def export_image():
    global WORKSPACE_IMAGE

    # Get file path
    file_path = filedialog.asksaveasfilename()
    file_path = file_path + '.jpg'

    # Convert color space
    _img = cv.cvtColor(WORKSPACE_IMAGE, cv.COLOR_RGBA2BGR)

    # Save image
    cv.imwrite(file_path, _img)
