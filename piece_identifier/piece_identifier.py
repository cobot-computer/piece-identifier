import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from std_msgs.msg import String
import cv2
from PIL import Image as PILImage
import numpy as np
import json
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont

text_size = 8

debug_mul = 4

text_size *= debug_mul


def draw_bb(image, left, upper, right, lower, label):
    # Create a drawing context
    draw = ImageDraw.Draw(image)

    # Draw the rectangle (bounding box)
    draw.rectangle((left, upper, right, lower), outline="red", width=2)

    try:
        font = ImageFont.truetype("arial.ttf", text_size)
    except IOError:
        font = ImageFont.load_default()

    # Position for the label. This example positions it above the bounding box.
    label_pos = (left, upper - text_size - 2)

    # Draw the label
    draw.text(label_pos, label, fill="cyan", font=font)

    return image


B_PERCENT = 22 / 377

RESIZE_TO = (100, 100)


class YOLOModel:
    def __init__(self, model_path):
        # Load the model
        self.model = YOLO(model_path)

    def predict(self, img):
        # Inference
        results = self.model(img, verbose=False)

        return results

    def crop_and_run(self, image: Image):
        width, height = image.size
        crop_x = int(width * B_PERCENT)
        crop_y = int(height * B_PERCENT)
        crop_width = int(width - (2 * B_PERCENT * width))
        crop_height = int(height - (2 * B_PERCENT * height))
        checkerboard_img = image.crop((crop_x, crop_y, crop_x + crop_width, crop_y + crop_height))

        checkerboard_width, checkerboard_height = checkerboard_img.size

        # Calculate the size of each cell
        cell_width = checkerboard_width / 8
        cell_height = checkerboard_height / 8

        output_matrix = []
        images = []

        # Iterate over each row and column
        for row in range(8):
            for col in range(8):
                # Calculate the left, upper, right, and lower coordinates for each cell
                left = round(col * cell_width)
                upper = round(row * cell_height)
                right = round(left + cell_width)
                lower = round(upper + cell_height)
                crop_box = (left, upper, right, lower)

                # Crop the cell from the image
                cell_image = checkerboard_img.crop(crop_box)

                cell_image = cell_image.resize(RESIZE_TO)

                images.append(cell_image)

        results = self.predict(images)

        draw_img = image.copy().resize((image.size[0] * debug_mul, image.size[1] * debug_mul))

        offset_x = round(B_PERCENT * draw_img.size[0])
        offset_y = round(B_PERCENT * draw_img.size[1])

        cell_width *= debug_mul
        cell_height *= debug_mul
        # Iterate over each row and column
        index = 0
        for row in range(8):
            row_classifications = [].copy()
            for col in range(8):
                left = round(col * cell_width) + offset_x
                upper = round(row * cell_height) + offset_y
                right = round(left + cell_width)
                lower = round(upper + cell_height)

                classifications = results[index]
                index += 1

                names = classifications.names
                probs = classifications.probs.data

                actual_names = []
                for key in range(13):
                    actual_names.append(names[key])

                confidence_scores = {}
                for label, confidence in zip(actual_names, probs):
                    confidence_scores[label] = float(confidence)

                row_classifications.append(confidence_scores)

                label = sorted(confidence_scores, key=confidence_scores.get)[-1]

                draw_img = draw_bb(draw_img, left, upper, right, lower, label)
            output_matrix.append(row_classifications)

        return draw_img, output_matrix


class PieceIdentifier(Node):
    def __init__(self):
        super().__init__("piece_identifier")
        self.model = YOLOModel("cobot_best.pt")
        self.subscription = self.create_subscription(
            Image, "/chessboard/image_raw", self.listener_callback, 10
        )
        self.image_publisher = self.create_publisher(Image, "/chessboard/annotated/image_raw", 10)
        self.matrix_publisher = self.create_publisher(
            String, "/chessboard/piece_identifier/inference_results", 10
        )
        self.br = CvBridge()

    def listener_callback(self, data):
        # Convert ROS Image message to OpenCV image
        current_frame = self.br.imgmsg_to_cv2(data)

        # Convert OpenCV image to Pillow Image
        pil_image = PILImage.fromarray(cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB))

        # Send to yolo model
        preview_image, inference_matrix = self.model.crop_and_run(pil_image)

        # Convert Pillow image back to OpenCV image
        processed_image = np.array(preview_image)

        # Ensure it's in the correct format (mono8 for grayscale)
        processed_image = processed_image[:, :, np.newaxis]

        # Convert back to ROS Image message and publish
        self.image_publisher.publish(self.br.cv2_to_imgmsg(processed_image, "mono8"))

        self.matrix_publisher.publish(str(inference_matrix))


def main(args=None):
    rclpy.init(args=args)
    image_processor = PieceIdentifier()
    rclpy.spin(image_processor)
    # Cleanup
    image_processor.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
