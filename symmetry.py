import cv2
import numpy as np
import dlib

class FaceSymmetryTester:
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    def detect_landmarks(self, image):
        if image is None:
            raise ValueError("Image is None, cannot process.")

        # Convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Verify the image data type and shape
        if gray.dtype != np.uint8:
            raise ValueError("Grayscale image must be of type uint8.")
        if len(gray.shape) != 2:
            raise ValueError("Grayscale image must be 2D.")

        # Detect faces
        faces = self.detector(gray)
        if len(faces) == 0:
            return None

        landmarks = []
        for face in faces:
            shape = self.predictor(gray, face)
            for i in range(68):  # 68 landmarks
                landmarks.append((shape.part(i).x, shape.part(i).y))
        return landmarks

    def calculate_distance(self, landmarks, indices):
        if landmarks is None:
            return None

        points = np.array([landmarks[i] for i in indices])

        # Calculate the mean distance between specified points
        distances = np.linalg.norm(points - np.flip(points, axis=0), axis=1)
        distance = np.mean(distances)
        return distance

    def calculate_symmetry_ratios(self, original_landmarks, mirrored_landmarks):
        if original_landmarks is None or mirrored_landmarks is None:
            return None

        # Define indices for different facial features
        left_indices = list(range(0, 17))  # Left side of the face
        right_indices = list(range(16, -1, -1))  # Right side of the face
        forehead_indices = list(range(17, 27))  # Forehead
        chin_to_ear_indices = left_indices + right_indices  # Chin to ear
        lip_corner_to_eye_and_ear_indices = list(range(48, 54)) + list(range(60, 64))  # Lip corner to eye and ear
        nose_to_ear_indices = list(range(27, 36)) + list(range(27, 35))  # Nose to ear

        # Calculate distances for each facial feature
        chin_to_ear_distance_original = self.calculate_distance(original_landmarks, chin_to_ear_indices)
        chin_to_ear_distance_mirrored = self.calculate_distance(mirrored_landmarks, chin_to_ear_indices)

        lip_corner_to_eye_and_ear_distance_original = self.calculate_distance(original_landmarks, lip_corner_to_eye_and_ear_indices)
        lip_corner_to_eye_and_ear_distance_mirrored = self.calculate_distance(mirrored_landmarks, lip_corner_to_eye_and_ear_indices)

        nose_to_ear_distance_original = self.calculate_distance(original_landmarks, nose_to_ear_indices)
        nose_to_ear_distance_mirrored = self.calculate_distance(mirrored_landmarks, nose_to_ear_indices)

        forehead_distance_original = self.calculate_distance(original_landmarks, forehead_indices)
        forehead_distance_mirrored = self.calculate_distance(mirrored_landmarks, forehead_indices)

        # Calculate symmetry ratios for each facial feature
        chin_to_ear_symmetry_ratio = chin_to_ear_distance_mirrored / chin_to_ear_distance_original if chin_to_ear_distance_original else None
        lip_corner_to_eye_and_ear_symmetry_ratio = lip_corner_to_eye_and_ear_distance_mirrored / lip_corner_to_eye_and_ear_distance_original if lip_corner_to_eye_and_ear_distance_original else None
        nose_to_ear_symmetry_ratio = nose_to_ear_distance_mirrored / nose_to_ear_distance_original if nose_to_ear_distance_original else None
        forehead_symmetry_ratio = forehead_distance_mirrored / forehead_distance_original if forehead_distance_original else None

        return chin_to_ear_symmetry_ratio, lip_corner_to_eye_and_ear_symmetry_ratio, nose_to_ear_symmetry_ratio, forehead_symmetry_ratio

    def test_symmetry(self):
        original_image_path = input("Enter the path to the original image: ")
        original_image = cv2.imread(original_image_path)

        if original_image is None:
            raise ValueError("Error: Unable to read the image.")

        # Create mirror image by flipping horizontally
        mirrored_image = cv2.flip(original_image, 1)

        original_landmarks = self.detect_landmarks(original_image)
        mirrored_landmarks = self.detect_landmarks(mirrored_image)

        if original_landmarks is None or mirrored_landmarks is None:
            raise ValueError("Face not detected in one or both images.")

        symmetry_ratios = self.calculate_symmetry_ratios(original_landmarks, mirrored_landmarks)

        return original_image, mirrored_image, symmetry_ratios

    def draw_symmetry_lines(self, image, landmarks):
        if landmarks is None:
            return

        # Draw lines for facial features
        chin_to_ear_indices = list(range(0, 17)) + list(range(26, 16, -1))
        for i in range(len(chin_to_ear_indices) - 1):
            cv2.line(image, landmarks[chin_to_ear_indices[i]], landmarks[chin_to_ear_indices[i + 1]], (0, 255, 0), 2)

        lip_corner_to_eye_and_ear_indices = list(range(48, 54)) + list(range(60, 64))
        for i in range(len(lip_corner_to_eye_and_ear_indices) - 1):
            cv2.line(image, landmarks[lip_corner_to_eye_and_ear_indices[i]], landmarks[lip_corner_to_eye_and_ear_indices[i + 1]], (0, 255, 0), 2)

        nose_to_ear_indices = list(range(27, 36)) + list(range(27, 35))
        for i in range(len(nose_to_ear_indices) - 1):
            cv2.line(image, landmarks[nose_to_ear_indices[i]], landmarks[nose_to_ear_indices[i + 1]], (0, 255, 0), 2)

if __name__ == "__main__":
    tester = FaceSymmetryTester()

    try:
        original_image, mirrored_image, symmetry_ratios = tester.test_symmetry()

        if original_image is not None and mirrored_image is not None:
            print("Symmetry Percentages:")
            if symmetry_ratios[0] is not None:
                print("Chin to Ear Symmetry Percentage: {:.2f}%".format((1 - symmetry_ratios[0]) * 100))
            else:
                print("Chin to Ear Symmetry Percentage: Not calculable")
            if symmetry_ratios[1] is not None:
                print("Lip Corner to Eye and Ear Symmetry Percentage: {:.2f}%".format((1 - symmetry_ratios[1]) * 100))
            else:
                print("Lip Corner to Eye and Ear Symmetry Percentage: Not calculable")
            if symmetry_ratios[2] is not None:
                print("Nose to Ear Symmetry Percentage: {:.2f}%".format((1 - symmetry_ratios[2]) * 100))
            else:
                print("Nose to Ear Symmetry Percentage: Not calculable")
            if symmetry_ratios[3] is not None:
                print("Forehead Symmetry Percentage: {:.2f}%".format((1 - symmetry_ratios[3]) * 100))
            else:
                print("Forehead Symmetry Percentage: Not calculable")

            # Draw symmetry lines on original image
            original_landmarks = tester.detect_landmarks(original_image)
            tester.draw_symmetry_lines(original_image, original_landmarks)

            # Draw symmetry lines on mirrored image
            mirrored_landmarks = tester.detect_landmarks(mirrored_image)
            tester.draw_symmetry_lines(mirrored_image, mirrored_landmarks)

            # Resize images for display (optional)
            scale_percent = 50  # percent of original size
            width = int(original_image.shape[1] * scale_percent / 100)
            height = int(original_image.shape[0] * scale_percent / 100)
            original_image = cv2.resize(original_image, (width, height))
            mirrored_image = cv2.resize(mirrored_image, (width, height))

            # Display images
            cv2.imshow("Original Image", original_image)
            cv2.imshow("Mirrored Image", mirrored_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("Face not detected in one or both images.")
    except Exception as e:
        print(f"An error occurred: {e}")
