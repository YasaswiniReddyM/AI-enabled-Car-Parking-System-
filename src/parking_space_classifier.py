import cv2
import numpy as np
import pickle
import urllib.request

class ParkClassifier:
    def __init__(self, car_park_positions_path, rect_width=None, rect_height=None):
        self.car_park_positions = self._read_positions(car_park_positions_path)
        self.rect_height = 48 if rect_height is None else rect_height
        self.rect_width = 107 if rect_width is None else rect_width

    def _read_positions(self, car_park_positions_path):
        car_park_positions = None
        try:
            if car_park_positions_path.startswith("http://") or car_park_positions_path.startswith("https://"):
                response = urllib.request.urlopen(car_park_positions_path)
                car_park_positions = pickle.load(response)
            else:
                car_park_positions = pickle.load(open(car_park_positions_path, 'rb'))
        except Exception as e:
            print(f"Error: {e}\n It raised while reading the car park positions file.")
        return car_park_positions

    def classify(self, image, processed_image, threshold=900):
        empty_car_park = 0
        for x, y in self.car_park_positions:
            col_start, col_stop = x, x + self.rect_width
            row_start, row_stop = y, y + self.rect_height
            crop = processed_image[row_start:row_stop, col_start:col_stop]
            count = cv2.countNonZero(crop)
            empty_car_park, color, thick = (
                [empty_car_park + 1, (0, 255, 0), 5]
                if count < threshold
                else [empty_car_park, (0, 0, 255), 2]
            )
            start_point, stop_point = (x, y), (x + self.rect_width, y + self.rect_height)
            cv2.rectangle(image, start_point, stop_point, color, thick)

        cv2.rectangle(image, (45, 30), (250, 75), (180, 0, 180), -1)
        ratio_text = f'Free: {empty_car_park}/{len(self.car_park_positions)}'
        cv2.putText(image, ratio_text, (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        return image

    def implement_process(self, image):
        kernel_size = np.ones((3, 3), np.uint8)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (3, 3), 1)
        thresholded = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 16)
        blur = cv2.medianBlur(thresholded, 5)
        dilate = cv2.dilate(blur, kernel_size, iterations=1)
        return dilate
