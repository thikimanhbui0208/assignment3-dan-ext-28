import random
import cv2
import numpy as np

# Difference classes show inheritance and polymorphism.
class Difference:
    """Parent class for one hidden difference."""

    def __init__(self, x, y, w, h):
        self._x = x
        self._y = y
        self._w = w
        self._h = h
        self._found = False

    @property
    def found(self):
        return self._found

    @property
    def bbox(self):
        return self._x, self._y, self._w, self._h

    def mark_found(self):
        self._found = True

    def center(self):
        return self._x + self._w // 2, self._y + self._h // 2

    def radius(self):
        return max(self._w, self._h) // 2 + 10  # Extra space makes tiny differences easier to click

    def contains(self, px, py):
        cx, cy = self.center()
        tolerance = max(15, int(self.radius() * 0.5))  # Gives the player a fair click area
        distance = ((px - cx) ** 2 + (py - cy) ** 2) ** 0.5
        return distance <= self.radius() + tolerance

    def overlaps(self, other, min_center_distance=0):
        x1, y1, w1, h1 = self.bbox
        x2, y2, w2, h2 = other.bbox
        padding = 5  # Keeps the hidden rectangles from touching
        rectangles_overlap = not (x1 + w1 + padding < x2 or
                                  x2 + w2 + padding < x1 or
                                  y1 + h1 + padding < y2 or
                                  y2 + h2 + padding < y1)
        if rectangles_overlap:
            return True

        cx1, cy1 = self.center()
        cx2, cy2 = other.center()
        distance = ((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2) ** 0.5
        return distance < min_center_distance  # Keeps the fixed marker circles from crowding each other

    def change_amount(self, roi, dark_amount, normal_amount, bright_amount):
        brightness = roi.mean()  # Average brightness of the selected area
        if brightness > 170:
            return bright_amount
        if brightness > 100:
            return normal_amount
        return dark_amount

    def apply(self, image):
        raise NotImplementedError("Child classes must define apply().")


class ColorShiftDifference(Difference):
    """Changes one color channel in the selected area."""

    def apply(self, image):
        x, y, w, h = self.bbox
        roi = image[y:y+h, x:x+w].copy()
        channel = random.randint(0, 2)  # OpenCV stores colors as B, G, R.
        amount = self.change_amount(roi, 65, 50, 35)  # Bright areas need a softer change
        if roi[:, :, channel].mean() >= 128:
            amount = -amount  # Move the color away from its current value
        roi[:, :, channel] = np.clip(
            roi[:, :, channel].astype(np.int16) + amount, 0, 255
        ).astype(np.uint8)
        image[y:y+h, x:x+w] = roi
        return image


class BlurDifference(Difference):
    """Blurs the selected area."""

    def apply(self, image):
        x, y, w, h = self.bbox
        roi = image[y:y+h, x:x+w].copy()
        kernel_size = max(7, (min(w, h) // 2) | 1)  # Gaussian blur needs an odd kernel size
        blurred = cv2.GaussianBlur(roi, (kernel_size, kernel_size), 0)
        if cv2.absdiff(roi, blurred).mean() < 10:
            return BrightnessDifference(x, y, w, h).apply(image)  # Use a clearer change if blur is too subtle
        image[y:y+h, x:x+w] = blurred
        return image


class BrightnessDifference(Difference):
    """Makes the selected area brighter or darker."""

    def apply(self, image):
        x, y, w, h = self.bbox
        roi = image[y:y+h, x:x+w].copy()
        amount = self.change_amount(roi, 45, 35, 22)  # Bright areas need a softer change
        if roi.mean() >= 128:
            amount = -amount  # Darken bright areas and brighten dark areas
        roi = np.clip(roi.astype(np.int16) + amount, 0, 255).astype(np.uint8)
        image[y:y+h, x:x+w] = roi
        return image


class PatchCopyDifference(Difference):
    """Copies a nearby patch into the selected area."""

    def apply(self, image):
        x, y, w, h = self.bbox
        image_h, image_w = image.shape[:2]
        original_roi = image[y:y+h, x:x+w].copy()

        for _ in range(20):  # Try a few nearby patches before using the fallback
            dx = random.choice([-1, 1]) * random.randint(w + 10, w * 4)
            dy = random.choice([-1, 1]) * random.randint(h + 10, h * 4)
            sx, sy = x + dx, y + dy
            if not (0 <= sx <= image_w - w and 0 <= sy <= image_h - h):
                continue

            patch = image[sy:sy+h, sx:sx+w].copy()
            patch_difference = cv2.absdiff(original_roi, patch).mean()  # Avoid patches that are identical or too obvious
            if 10 <= patch_difference <= 45:
                image[y:y+h, x:x+w] = patch
                return image

        return BrightnessDifference(x, y, w, h).apply(image)
