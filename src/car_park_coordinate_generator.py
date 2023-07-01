import cv2
import pickle

class CoordinateDenoter:
    def __init__(self, rect_width=107, rect_height=48, car_park_positions_path="C:/Users/yawir/Downloads/car-parking-finder-main/data/source/CarParkPos"):
        self.rect_width = rect_width
        self.rect_height = rect_height
        self.car_park_positions_path = car_park_positions_path
        self.car_park_positions = []

    def read_positions(self):
        try:
            with open(self.car_park_positions_path, 'rb') as f:
                self.car_park_positions = pickle.load(f)
        except Exception as e:
            print(f"Error: {e}\n It raised while reading the car park positions file.")
            self.car_park_positions = []

    def mouse_click(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONUP:
            self.car_park_positions.append((x, y))

    def demonstration(self):
        # image_path = "C:/Users/yawir/Downloads/car-parking-finder-main/data/source/example_image.png"
        image_path = "carParking.png"
        cv2.namedWindow("Image")
        cv2.setMouseCallback("Image", self.mouse_click)

        while True:
            image = cv2.imread(image_path)
            for pos in self.car_park_positions:
                start = pos
                end = (pos[0] + self.rect_width, pos[1] + self.rect_height)
                cv2.rectangle(image, start, end, (0, 0, 255), 2)

            cv2.imshow("Image", image)
            key = cv2.waitKey(1)

            if key == ord("q"):
                break

        cv2.destroyAllWindows()

        with open(self.car_park_positions_path, 'wb') as f:
            pickle.dump(self.car_park_positions, f)
