import random
import cv2
import numpy as np

from PIL import Image, ImageTk
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk


CANVAS_SIZE = 550
DIFFERENCES_PER_IMAGE = 5
MARKER_RADIUS = 20
MARKER_GAP = 8
MAX_MISTAKES = 3

BG = "#0f172a"          # dark navy
PANEL = "#111827"       # charcoal
BORDER = "#334155"      # slate
TEXT = "#e5e7eb"        # light gray
MUTED = "#94a3b8"       # gray-blue
BLUE = "#60a5fa"        # blue
LIGHT_BLUE = "#93c5fd"  # light blue
GREEN = "#34d399"       # green
RED = "#ef4444"         # red
PINK = "#fb7185"        # pink-red

FONT_TITLE = ("Segoe UI", 22, "bold")
FONT_HEADER = ("Segoe UI", 12, "bold")
FONT_NORMAL = ("Segoe UI", 11)
FONT_SMALL = ("Segoe UI", 10)


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


class ImageProcessor:
    """Loads images and creates the modified copy."""

    DIFF_TYPES = [
        ColorShiftDifference,
        BlurDifference,
        BrightnessDifference,
        PatchCopyDifference,
    ]

    def __init__(self, max_display_size=(CANVAS_SIZE, CANVAS_SIZE)):
        self.max_display_size = max_display_size

    def load_image(self, path):
        allowed_extensions = [".jpg", ".jpeg", ".png", ".bmp"]  # File types required for the assignment
        file_extension = Path(path).suffix.lower()
        if file_extension not in allowed_extensions:
            raise ValueError("Please choose a JPG, JPEG, PNG, or BMP image file.")

        image = cv2.imread(path)
        if image is None:
            raise ValueError("The selected file has an image extension, but OpenCV could not read it. The file may be damaged or unsupported.")
        return image

    def generate_differences(self, image, count=DIFFERENCES_PER_IMAGE):
        h, w = image.shape[:2]
        short_side = min(w, h)
        scale = min(self.max_display_size[0] / w, self.max_display_size[1] / h)  # Display scale

        min_display_size = 6  # Smallest visible difference size
        max_display_size = 10  # Largest visible difference size

        min_size = max(3, int(min_display_size / scale))  # Smallest difference in original pixels
        max_size = max(min_size + 4, int(max_display_size / scale))  # Largest difference in original pixels
        margin = max(5, int(short_side * 0.02), int(MARKER_RADIUS / scale) + 1)  # Keeps marker circles inside the image edge
        marker_spacing = int(((MARKER_RADIUS * 2) + MARKER_GAP) / scale) + 1  # Circle spacing in original pixels

        if w < max_size * 3 or h < max_size * 3:
            raise ValueError(f"Image too small for {count} distinct differences.")

        diffs = []
        type_counts = {}
        attempts = 0
        while len(diffs) < count and attempts < 5000:
            attempts += 1
            available = [cls for cls in self.DIFF_TYPES if type_counts.get(cls.__name__, 0) < 2]
            diff_class = random.choice(available or self.DIFF_TYPES)

            dw = random.randint(min_size, max_size)
            dh = random.randint(min_size, max_size)
            x = random.randint(margin, w - dw - margin)
            y = random.randint(margin, h - dh - margin)
            candidate = diff_class(x, y, dw, dh)

            if any(candidate.overlaps(diff, marker_spacing) for diff in diffs):  # Guarantees the five regions do not overlap
                continue

            diffs.append(candidate)
            type_counts[diff_class.__name__] = type_counts.get(diff_class.__name__, 0) + 1

        if len(diffs) < count:
            raise RuntimeError("Could not place all differences without overlap.")

        modified = image.copy()
        for diff in diffs:
            diff.apply(modified)  # Polymorphism: each child class changes the image differently
        return modified, diffs

    def fit_to_display(self, image):
        h, w = image.shape[:2]
        max_w, max_h = self.max_display_size
        scale = min(max_w / w, max_h / h)  # Fit inside the display box
        new_w = max(1, int(w * scale))  # Display width
        new_h = max(1, int(h * scale))  # Display height
        interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC  # Shrink or enlarge nicely
        image = cv2.resize(image, (new_w, new_h), interpolation=interpolation)
        return image, scale

    def to_photoimage(self, bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)  # Tkinter expects RGB, OpenCV uses BGR
        return ImageTk.PhotoImage(Image.fromarray(rgb))


class GameState:
    """Tracks the current round."""

    def __init__(self):
        self.total_found = 0  # Cumulative score across multiple images
        self.reset_for_new_image([])

    @property
    def remaining(self):
        return sum(1 for diff in self.differences if not diff.found)

    def reset_for_new_image(self, diffs):
        self.differences = diffs
        self.mistakes = 0
        self.locked = False  # Allows clicks again for the new image

    def register_click(self, x, y):
        if self.locked:
            return None
        for diff in self.differences:
            if not diff.found and diff.contains(x, y):
                diff.mark_found()
                self.total_found += 1
                return diff
        self.mistakes += 1
        if self.mistakes >= MAX_MISTAKES:
            self.locked = True  # Stop image clicks after three wrong guesses
        return None

    def reveal_all(self):
        unfound = [diff for diff in self.differences if not diff.found]
        for diff in unfound:
            diff.mark_found()
        self.locked = True
        return unfound


class SpotTheDifferenceApp(tk.Tk):
    """Main Tkinter app."""

    def __init__(self):
        super().__init__()
        self.title("Spot the Differences")
        self.configure(bg=BG)
        self.resizable(False, False)

        self.processor = ImageProcessor()
        self.state = GameState()
        self.original_bgr = None
        self.modified_bgr = None
        self.scale = 1.0
        self.photo_left = None
        self.photo_right = None

        self.setup_style()
        self.build_ui()
        self.update_status()

    def setup_style(self):
        style = ttk.Style(self)
        try:
            style.theme_use("clam")  # Helps custom button colors show on more systems
        except tk.TclError:
            pass
        style.configure("App.TButton", font=FONT_NORMAL, padding=(14, 7), background=BLUE, foreground=BG)
        style.map("App.TButton", background=[("active", LIGHT_BLUE), ("disabled", PANEL)], foreground=[("disabled", MUTED)])

    def build_ui(self):
        tk.Label(self, text="Spot the Differences", fg=TEXT, bg=BG, font=FONT_TITLE).pack(pady=(15, 5))

        button_row = tk.Frame(self, bg=BG)
        button_row.pack(pady=5)
        ttk.Button(button_row, text="Load Image", width=14, style="App.TButton", command=self.load_image).pack(side=tk.LEFT, padx=10)
        self.reveal_btn = ttk.Button(button_row, text="Reveal", width=14, style="App.TButton", command=self.reveal, state=tk.DISABLED)
        self.reveal_btn.pack(side=tk.LEFT, padx=10)

        self.message_var = tk.StringVar(value="Load an image to start.")
        self.message_label = tk.Label(self, textvariable=self.message_var, fg=BLUE, bg=BG, font=FONT_SMALL)
        self.message_label.pack(pady=(8, 2))

        self.status_var = tk.StringVar()
        tk.Label(self, textvariable=self.status_var, fg=LIGHT_BLUE, bg=BG, font=FONT_HEADER).pack(pady=(0, 10))

        image_row = tk.Frame(self, bg=BG)
        image_row.pack(pady=(12, 16))
        self.canvas_left = self.make_image_column(image_row, "Original Image\n")
        self.canvas_right = self.make_image_column(image_row, "Modified Image\nClick here to find differences")
        self.canvas_right.config(cursor="crosshair")
        self.canvas_right.bind("<Button-1>", self.on_click_modified)

    def make_image_column(self, parent, title):
        column = tk.Frame(parent, bg=BG)
        column.pack(side=tk.LEFT, padx=20)
        tk.Label(column, text=title, fg=TEXT, bg=BG, font=FONT_HEADER, justify="center").pack(pady=(0, 6))
        canvas = tk.Canvas(
            column,
            width=CANVAS_SIZE,
            height=CANVAS_SIZE,
            bg=PANEL,
            highlightbackground=BORDER,
            highlightcolor=BORDER,
            highlightthickness=1,
            bd=0,
        )
        canvas.pack()
        return canvas

    def load_image(self):
        path = filedialog.askopenfilename(
            title="Choose an image",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.JPG *.JPEG *.PNG *.BMP"), ("All files", "*.*")]
        )
        if not path:
            return

        try:
            full_original = self.processor.load_image(path)
            full_modified, diffs = self.processor.generate_differences(full_original)
        except Exception as error:
            messagebox.showerror("Error", str(error))
            return

        self.original_bgr, self.scale = self.processor.fit_to_display(full_original)  # Save scale for click mapping
        self.modified_bgr, _ = self.processor.fit_to_display(full_modified)
        self.state.reset_for_new_image(diffs)
        self.message_var.set("Game started. Find all 5 differences.")
        self.message_label.config(fg=BLUE)
        self.reveal_btn.config(state=tk.NORMAL)
        self.refresh_canvases()
        self.update_status()

    def reveal(self):
        if not self.state.differences:
            return
        for diff in self.state.reveal_all():
            self.draw_circle_on_both(diff, BLUE)
        self.message_var.set("Revealed all remaining differences. Load a new image to play again.")
        self.message_label.config(fg=LIGHT_BLUE)
        self.reveal_btn.config(state=tk.DISABLED)
        self.update_status()

    def on_click_modified(self, event):
        if not self.state.differences or self.state.locked:
            return

        offset_x = (CANVAS_SIZE - self.modified_bgr.shape[1]) // 2  # Image is centered inside the canvas
        offset_y = (CANVAS_SIZE - self.modified_bgr.shape[0]) // 2  # Image is centered inside the canvas
        if not (0 <= event.x - offset_x < self.modified_bgr.shape[1] and 0 <= event.y - offset_y < self.modified_bgr.shape[0]):
            return

        image_x = int((event.x - offset_x) / self.scale)  # Convert canvas click to original image x
        image_y = int((event.y - offset_y) / self.scale)  # Convert canvas click to original image y
        match = self.state.register_click(image_x, image_y)

        if match is not None:
            self.draw_circle_on_both(match, RED)
            if self.state.remaining == 0:
                self.finish_round()
        else:
            self.show_wrong_click(event.x, event.y)
            if self.state.locked:
                found = DIFFERENCES_PER_IMAGE - self.state.remaining
                self.message_var.set(f"Too many mistakes! You found {found}/{DIFFERENCES_PER_IMAGE} differences. Use Reveal or load a new image to keep playing.")
                self.message_label.config(fg=PINK)
                self.update_status()
                self.update_idletasks()  # Let Tkinter draw the wrong-click mark before the popup appears
                messagebox.showinfo(
                    "Too many mistakes",
                    f"You found {found}/{DIFFERENCES_PER_IMAGE} differences.\n"
                    "Use Reveal to see the remaining answers or load a new image to keep playing."
                )
        self.update_status()

    def finish_round(self):
        self.state.locked = True
        self.reveal_btn.config(state=tk.DISABLED)
        self.message_var.set("You found all 5 differences. Load another image to keep playing.")
        self.message_label.config(fg=GREEN)
        self.update_status()
        self.update_idletasks()  # Let Tkinter draw the last circle before the popup appears
        self.after(100, lambda: messagebox.showinfo("Well done!", f"You found all {DIFFERENCES_PER_IMAGE} differences!\nTotal found: {self.state.total_found}"))

    def refresh_canvases(self):
        self.photo_left = self.processor.to_photoimage(self.original_bgr)
        self.photo_right = self.processor.to_photoimage(self.modified_bgr)
        self.canvas_left.delete("all")
        self.canvas_right.delete("all")
        for canvas, photo in ((self.canvas_left, self.photo_left), (self.canvas_right, self.photo_right)):
            x = (CANVAS_SIZE - photo.width()) // 2  # Center image horizontally
            y = (CANVAS_SIZE - photo.height()) // 2  # Center image vertically
            canvas.create_image(x, y, image=photo, anchor=tk.NW)

    def draw_circle_on_both(self, diff, color):
        cx, cy = diff.center()
        radius = MARKER_RADIUS
        x = cx * self.scale + (CANVAS_SIZE - self.original_bgr.shape[1]) // 2  # Display x position
        y = cy * self.scale + (CANVAS_SIZE - self.original_bgr.shape[0]) // 2  # Display y position
        for canvas in (self.canvas_left, self.canvas_right):
            canvas.create_oval(x - radius, y - radius, x + radius, y + radius, outline=color, width=3)

    def show_wrong_click(self, x, y):
        size = 12
        line1 = self.canvas_right.create_line(x - size, y - size, x + size, y + size, fill=PINK, width=3)
        line2 = self.canvas_right.create_line(x - size, y + size, x + size, y - size, fill=PINK, width=3)
        self.after(600, lambda: (self.canvas_right.delete(line1), self.canvas_right.delete(line2)))  # Remove the wrong-click mark after a short moment

    def update_status(self):
        self.status_var.set(
            f"Remaining: {self.state.remaining}  |  "
            f"Mistakes: {self.state.mistakes} / {MAX_MISTAKES}  |  "
            f"Total Found: {self.state.total_found}"
        )


if __name__ == "__main__":
    SpotTheDifferenceApp().mainloop()
