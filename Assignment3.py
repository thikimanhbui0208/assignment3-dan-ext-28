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
        class Difference:
"""
Represents one difference area in the image.
This class stores the x, y position, width, height,
and whether the difference has already been found.
"""

def __init__(self, x, y, width, height):
    self.x = x
    self.y = y
    self.width = width
    self.height = height
    self.found = False

def contains_click(self, click_x, click_y):
    """
    Checks if the player's click is inside this difference area.
    """
    return (
        self.x <= click_x <= self.x + self.width
        and self.y <= click_y <= self.y + self.height
    )

def get_center(self):
    """
    Returns the centre point of the difference area.
    Useful for drawing circles in the GUI.
    """
    center_x = self.x + self.width // 2
    center_y = self.y + self.height // 2
    return center_x, center_y

def get_radius(self):
    """
    Returns a suitable circle radius for marking the difference.
    """
    return max(self.width, self.height) // 2 + 10
class GameLogic:
"""
Handles the main game rules:
- checking clicks
- counting found differences
- counting mistakes
- stopping after 3 mistakes
- revealing unfound differences
- resetting for a new image
"""

def __init__(self, difference_regions):
    """
    difference_regions can be a list of Difference objects
    or a list of dictionaries from the image processing part.

    Example dictionary format:
    {"x": 100, "y": 80, "w": 50, "h": 40}
    """

    self.max_mistakes = 3
    self.load_differences(difference_regions)

def load_differences(self, difference_regions):
    """
    Loads the 5 generated difference regions into the game.
    This allows Part A and Part B to send the generated differences here.
    """

    self.differences = []

    for region in difference_regions:
        if isinstance(region, Difference):
            self.differences.append(region)
        else:
            x = region.get("x")
            y = region.get("y")
            width = region.get("w", region.get("width"))
            height = region.get("h", region.get("height"))

            self.differences.append(Difference(x, y, width, height))

    self.total_differences = len(self.differences)
    self.found_count = 0
    self.mistakes = 0
    self.game_over = False
    self.revealed = False

def check_click(self, click_x, click_y):
    """
    Checks the player's click on the modified image.

    Returns a dictionary so the GUI can easily understand what happened.
    """

    if self.game_over:
        return {
            "result": "game_over",
            "message": "No more guesses allowed.",
            "difference": None
        }

    for difference in self.differences:
        if difference.contains_click(click_x, click_y):
            if difference.found:
                return {
                    "result": "already_found",
                    "message": "This difference was already found.",
                    "difference": difference
                }

            difference.found = True
            self.found_count += 1

            if self.found_count == self.total_differences:
                self.game_over = True
                return {
                    "result": "win",
                    "message": "All differences found. You win!",
                    "difference": difference
                }

            return {
                "result": "correct",
                "message": "Correct difference found.",
                "difference": difference
            }

    self.mistakes += 1

    if self.mistakes >= self.max_mistakes:
        self.game_over = True
        return {
            "result": "lost",
            "message": "Maximum 3 mistakes reached. Game over.",
            "difference": None
        }

    return {
        "result": "wrong",
        "message": "Wrong click. Try again.",
        "difference": None
    }

def get_remaining_count(self):
    """
    Returns how many differences are still not found.
    """
    return self.total_differences - self.found_count

def get_found_count(self):
    """
    Returns how many differences have been found.
    """
    return self.found_count

def get_mistake_count(self):
    """
    Returns the current number of mistakes.
    """
    return self.mistakes

def get_score(self):
    """
    Returns a simple score.
    Correct findings increase the score.
    Mistakes reduce the score slightly.
    """
    score = self.found_count * 10 - self.mistakes * 2
    return max(score, 0)

def reveal_unfound_differences(self):
    """
    Reveals all differences that were not found yet.
    The GUI should draw these revealed differences with blue circles.
    """

    self.revealed = True
    self.game_over = True

    unfound_differences = []

    for difference in self.differences:
        if not difference.found:
            difference.found = True
            unfound_differences.append(difference)

    self.found_count = self.total_differences

    return unfound_differences

def get_all_differences(self):
    """
    Returns all difference regions.
    Useful when the GUI needs to draw circles.
    """
    return self.differences

def get_found_differences(self):
    """
    Returns only the differences already found by the player.
    These should be marked with red circles.
    """
    return [difference for difference in self.differences if difference.found]

def is_game_over(self):
    """
    Returns True if the game has ended.
    """
    return self.game_over

def has_won(self):
    """
    Returns True if the player found all differences before losing.
    """
    return self.found_count == self.total_differences and not self.revealed

def has_lost(self):
    """
    Returns True if the player reached 3 mistakes.
    """
    return self.mistakes >= self.max_mistakes

def reset_game(self, new_difference_regions):
    """
    Resets the game when a new image is loaded.
    """
    self.load_differences(new_difference_regions)
