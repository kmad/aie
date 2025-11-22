import cv2
import dspy
import numpy as np
from PIL import Image


def crop_region_by_grid_cells(image, grid_shape, cell_numbers):
    """
    Crop the region of `image` that covers the `cell_numbers`.
    Args:
        image (np.ndarray): Image array.
        grid_shape (tuple): (rows, cols) as used in your grid code.
        cell_numbers (list): List of cell ids (left-to-right, top-to-bottom, starting at 1).
    Returns:
        np.ndarray: Cropped region.
    """
    h, w = image.shape[:2]
    rows, cols = grid_shape

    # Convert cell_number to (row, col): cell_id = r * cols + c + 1
    coords = []
    for cell_id in cell_numbers:
        cell_id -= 1  # zero based
        row = cell_id // cols
        col = cell_id % cols
        x1 = int(col * w / cols)
        x2 = int((col + 1) * w / cols)
        y1 = int(row * h / rows)
        y2 = int((row + 1) * h / rows)
        coords.append((x1, y1, x2, y2))

    # Get bounding box that contains all selected cells
    x1s, y1s, x2s, y2s = zip(*coords)
    min_x, min_y = min(x1s), min(y1s)
    max_x, max_y = max(x2s), max(y2s)

    cropped = image[min_y:max_y, min_x:max_x]
    return cropped


def overlay_grid_on_image_bytes(
    image_bytes,
    grid_shape=(3, 3),
    color=(0, 255, 0),
    thickness=2,
    alpha=0.4,
    number_color=(255, 255, 255),
    number_scale=1.2,
    number_thickness=2,
    font=cv2.FONT_HERSHEY_SIMPLEX,
):
    """
    Overlay a grid on the image (provided as bytes), and add a single ID to each cell.
    Cells are numbered left-to-right, top-to-bottom, starting at 1.
    """
    # Load image from bytes
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image from bytes.")
    overlay = img.copy()
    h, w = img.shape[:2]
    rows, cols = grid_shape

    # draw grid lines
    for r in range(1, rows):
        y = int(r * h / rows)
        cv2.line(overlay, (0, y), (w, y), color, thickness)
    for c in range(1, cols):
        x = int(c * w / cols)
        cv2.line(overlay, (x, 0), (x, h), color, thickness)

    # single running ID
    cell_id = 1
    for r in range(rows):
        for c in range(cols):
            x1 = int(c * w / cols)
            x2 = int((c + 1) * w / cols)
            y1 = int(r * h / rows)
            y2 = int((r + 1) * h / rows)
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            label = str(cell_id)
            (text_w, text_h), _ = cv2.getTextSize(
                label, font, number_scale, number_thickness
            )
            text_x = cx - text_w // 2
            text_y = cy + text_h // 2

            # white halo
            cv2.putText(
                overlay,
                label,
                (text_x, text_y),
                font,
                number_scale,
                (255, 255, 255),
                number_thickness + 2,
                cv2.LINE_AA,
            )
            # coloured text
            cv2.putText(
                overlay,
                label,
                (text_x, text_y),
                font,
                number_scale,
                number_color,
                number_thickness,
                cv2.LINE_AA,
            )
            cell_id += 1

    out = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
    return out


class HandicapSignDetector(dspy.Signature):
    """Return ONLY the numbers associated with ALL applicable regions that capture the FULL sign with the handicap logo in the provided image. Add some buffer to the top and bottom of the sign to ensure you get the full sign."""

    image: dspy.Image = dspy.InputField()
    applicable_regions: list[int] = dspy.OutputField()


class VisualAnalyzer(dspy.Module):
    """Visual analyzer module with multimodal support."""

    def __init__(self):
        super().__init__()
        self.predictor = dspy.Predict(HandicapSignDetector)

    def forward(self, image_bytes: bytes) -> list[int]:
        IMG_SHAPE = (12, 12)

        # overlay the grid on the image represented as bytes
        img = overlay_grid_on_image_bytes(
            image_bytes,
            grid_shape=IMG_SHAPE,
            color=(255, 255, 255),
            thickness=1,
            alpha=0.3,
            number_color=(0, 0, 0),
        )
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        cv2.imwrite("output/overlayed_image.png", img)
        img_dspy = dspy.Image(img_pil)

        sig_region = dspy.Predict(HandicapSignDetector)
        pred = sig_region(image=img_dspy)
        print(pred.applicable_regions)

        #
        cropped_image = crop_region_by_grid_cells(
            img, IMG_SHAPE, pred.applicable_regions
        )
        # save to data/cropped_image.png
        cv2.imwrite("output/cropped_image.png", cropped_image)

        # Now crop the original image based on the image_bytes input
        original_img_array = cv2.imdecode(
            np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR
        )
        cropped_original_image = crop_region_by_grid_cells(
            original_img_array, IMG_SHAPE, pred.applicable_regions
        )
        # save to data/cropped_original_image.png
        cv2.imwrite("output/cropped_original_image.png", cropped_original_image)
        return pred.applicable_regions
