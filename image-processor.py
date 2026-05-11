import cv2
import numpy as np
import random

class Alteration:

      def __init__(self, name: str):
            self.name = name

      def apply (self, image:np.ndarray, x: int, y:int, w:int, h: int) -> np.ndarray:

            raise NotImplementedError ("Subclasses must implement apply()")
      
      def __repr__(self):
            return f"Alteration(name={self.name})"

class ColourShiftAlteration(Alteration):

      def __init__(self):
            super().__init__("colour_shift")
      #random hue shift amount between 30 and 90 deg
            self.hue_shift = random.randint (30,90)
      
      def apply(self, image:np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
            modified = image.copy()
            region = modified[y:y+h, x:x+w]

            hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV).astype(np.int32)
            hsv[:, :, 0] = (hsv [:, :, 0] + self.hue_shift) % 180
            hsv = hsv.astype(np.uint8)

            modified[y:y+h, x:x+w] = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            return modified
      
class BrightnessAlteration(Alteration) :
      # change in brightness region using HSV value channel to preserve color tone
      def __init__(self):
            super().__init__("brightness")
            self.delta = random. choice([-60, -50, 50,60])
      

      def apply(self, image: np.ndarray, x: int , y: int, w: int , h: int) -> np.ndarray:
            modified = image.copy()
            region = modified [y: y+h, x:x+w]

            hsv = cv2. cvtColor(region, cv2.COLOR_BGR2HSV).astype(np.int32)
            hsv[:, :, 2] = np.clip(hsv[:, :, 2] + self.delta,0,255)
            hsv = hsv.astype(np.uint8)
            modified[y:y+h, x:x+w] = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            return modified
      
class NoiseAlteration(Alteration):
      #Adding visible Gaussian noise to create a grainy patch
      def __init__(self):
            super().__init__("noise")
            self.intensity = random.randint(40,70)
      
      def apply(self, image: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
            modified = image.copy()
            region = modified [y:y+h, x:x+w].astype(np.int32)

            noise = np.random.randint(-self.intensity,self.intensity, region.shape).astype = (np.int32)
            noisy = np.clip(region + noise, 0, 255).astype(np.uint8)

            modified[y:y+h, x:x+w] = noisy
            return modified
      

class DifferenceRegion:

      def __init__(self, x: int, y: int, w: int, h: int, alteration_name: str):
            self.x = x
            self.y = y
            self.w = w
            self.h = h
            self.alteration_name = alteration_name
            self.found = False 

      def center(self) -> tuple:
            return (self.x + self.w // 2, self.y + self.h //2)
      
      def contains_point(self,px: int,py: int, tolerance: int = 0) -> bool:

            return (self.x - tolerance <= px <= self.x + self.w + tolerance and
                    self.y - tolerance <= py <= self.y + self.h + tolerance)
      def __repr__(self):
            return (f"DifferenceRegion(x={self.x}, y={self.y}, w={self.w}, h={self.h},"
                    f"type={self.alteration_name}, found={self.found}")
      
class ImageProcessor:
      NUM_DIFFERENCES = 5
      MIN_PATCH = 40
      MAX_PATCH = 80
      MAX_RETRIES = 200

      ALTERATION_CLASSES = [ColourShiftAlteration , BrightnessAlteration, NoiseAlteration]
      def __init__(self):
            self.original: np.ndarray | None = None
            self.modified: np.ndarray | None = None
            self.differences: list[DifferenceRegion] = []

      
      def load_image(self,filepath:str)-> bool:
            # Loads an image from the same folder and creates 5 random differences within the image
            img = cv2.imread(filepath)
            if img is None:
                  return False
            self.original = img
            self.modified, self.differences = self._generate_differences(img)
            return True
      def get_original_rgb(self) -> np.ndarray: 
            return cv2.cvtColor(self.original, cv2.COLOR_BGR2RGB)
      
      def get_modified_rgb(self) -> np.ndarray: 
            return cv2.cvtColor(self.modified, cv2.COLOR_BGR2RGB)
      

      

      def _generate_differences(self, image: np.ndarray):
            # Applying  NUM_DIFFERENCES non-overlapping alterations by cloning the image
            h, w = image.shape[:2]
            clone = image.copy()
            placed: list[DifferenceRegion] = []

            attempts = 0
            while len(placed) < self.NUM_DIFFERENCES and attempts < self.MAX_RETRIES:
                  attempts +=1

                  pw = random.randint(self.MIN_PATCH, self.MAX_PATCH)
                  ph = random.randint(self.MIN_PATCH, self.MAX_PATCH)

                  rx = random.randint(0, w - pw)
                  ry = random.randint(0, h - ph)
                  if self._overlaps_any(rx, ry, pw, ph, placed):
                         continue

                  alteration = random.choice(self.ALTERATION_CLASSES)()
                  clone = alteration.apply(clone,rx, ry, pw, ph)

                  region = DifferenceRegion(rx, ry, pw, ph, alteration.name)
                  placed.append(region)

            if len(placed) < self.NUM_DIFFERENCES:
                  print(f"Warning: only placed {len(placed)} differences (image may be too small).")
            return clone, placed
      def _overlaps_any(self,x: int, y: int, w: int, h: int,
                        placed: list[DifferenceRegion]) -> bool:
            # checking the overlapping of two rectangles and then added padding.
            PAD = 10
            for r in placed:
                  if (x < r.x + r.w + PAD and x + w + PAD > r.x and 
                         y < r.y + r.h + PAD and y + h + PAD > r.y):
                      return True
            return False
      

if __name__ == "__main__":
      import sys

      if len(sys.argv) < 2:
            print("Usage: python image_processor.py <path_to_image>")
            print("Example: python image_processor.py <path_to_image>")
          

      processor = ImageProcessor()
      success = processor.load_image(sys.argv[1])

      if not success:
            print (f"Could not load image: {sys.argv[1]}")
            sys.exit(1)
      print(f"Image loaded successfully")
      print(f"Placed{len(processor.differences)} differences:")
      for i, d in enumerate(processor.differences, 1):
            print(f" {i}. {d}")
      # Original and modified images are saved in the same folder
      cv2.imwrite("original_output.jpg", processor.original)
      cv2.imwrite("modified_output.jpg",processor.modified)
      print("Saved: Original Image file and Modified Image file are saved in the same folder")
      print("You can open both files side by side to find out the differences")                  
      


