from datetime import datetime
from typing import List, Tuple

import cv2
import dspy
import numpy as np


def crop_region_by_grid_cells(
    image: np.ndarray, grid_shape: Tuple[int, int], cell_numbers: List[int]
) -> np.ndarray:
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
    coords: List[Tuple[int, int, int, int]] = []
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
) -> np.ndarray:
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


class VisualAnalyzerWithTools(dspy.Module):
    """Visual analyzer module with multimodal support."""

    def __init__(self, image_bytes: bytes):
        super().__init__()
        self.image_bytes = image_bytes
        self.img_dspy = dspy.Image(image_bytes)
        self.img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        if self.img is None:
            raise ValueError("Could not decode image from bytes.")

    def _numpy_to_bytes(self, img_array: np.ndarray) -> bytes:
        """Convert numpy array to bytes for image processing functions."""
        _, encoded_img = cv2.imencode(".png", img_array)
        return encoded_img.tobytes()

    # Allow the LLM to determine the appropriate parameters
    def _overlay_grid_on_image(
        self, grid_shape, grid_color, grid_thickness, grid_alpha, number_color
    ) -> dspy.Image:
        """
        Overlay a rectangular grid on the input image.

        Args:
            grid_shape (tuple): The number of grid rows and columns, specified as (rows, cols).
            grid_color (tuple): The BGR color value for the grid lines, e.g., (255, 255, 255) for white.
            grid_thickness (int): Thickness of the grid lines.
            grid_alpha (float): Transparency factor for blending the grid over the image (0.0 to 1.0).
            number_color (tuple): The BGR color value used to label the grid cells with cell numbers.

        Returns:
            The original image with the overlaid grid.

        Notes:
            - Each cell in the grid may be optionally labeled with its index using the specified `number_color`.
            - Useful for visually inspecting or locating regions for subsequent processing tasks.
        """
        print(
            f"Called with grid_shape: {grid_shape}, grid_color: {grid_color}, grid_thickness: {grid_thickness}, grid_alpha: {grid_alpha}, number_color: {number_color}"
        )
        # Convert current image state (numpy array) to bytes for the function
        img_bytes = self._numpy_to_bytes(self.img)
        result = overlay_grid_on_image_bytes(
            img_bytes,
            grid_shape=grid_shape,
            color=grid_color,
            thickness=grid_thickness,
            alpha=grid_alpha,
            number_color=number_color,
        )
        # convert to bytes
        img_bytes = self._numpy_to_bytes(result)
        # save to data/tool_image_{timestamp}.png
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        cv2.imwrite(f"output/tool_grid_image{timestamp}.png", result)
        return dspy.Image(img_bytes)

    def _crop_region_by_grid_cells(self, grid_shape, cell_numbers) -> dspy.Image:
        """Crop a region from the original image based on grid cell numbers."""
        print(
            f"_crop_region_by_grid_cells called with grid_shape: {grid_shape}, cell_numbers: {cell_numbers}"
        )
        result = crop_region_by_grid_cells(self.img, grid_shape, cell_numbers)
        # convert to bytes
        img_bytes = self._numpy_to_bytes(result)
        # save to data/tool_image_{timestamp}.png
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        cv2.imwrite(f"output/tool_crop_image_{timestamp}.png", result)
        return dspy.Image(img_bytes)

    def forward(self) -> list[int]:
        sig_grid = dspy.Signature("image -> image_with_grid").with_instructions(
            """
            - Overlay a rectangular grid on the input image so that it is sufficient to capture the FULL sign with the handicap logo in the provided image.
            - Add some buffer to the top and bottom of the sign to ensure you get the full sign.
            - You may use multiple iterations to find the best parameters for the next LLM call to ensure you get the full sign. Experiment with things like color and grid size to optimize capture of the full street sign.
            - ONLY after you are confident you have the right parameters, crop the region of the image that corresponds to the grid cells that contain the handicap sign.
            - Do this by calling _crop_region_by_grid_cells tool, which will return the cropped image. This should only be called once you are confident you have the proper variables
            - Return the cropped image.
            """
        )
        img_grid = dspy.ReAct(
            sig_grid,
            tools=[self._overlay_grid_on_image, self._crop_region_by_grid_cells],
            max_iters=8,
        )
        response = img_grid(image=self.img_dspy)
        print(response.trajectory)

        return response
