import rclpy
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from std_msgs.msg import String
import cv2
from PIL import Image as PILImage
import numpy as np
from yolomodel import YOLOModel
from chess_msgs.msg import FullFEN, GameConfig, ClockButtons
import chess

WHITE = 0
BLACK = 1


def get_piece_name(board, x, y):
    # Convert x and y (1-based index) into a square index (0-based index)
    square = chess.square(7 - y, 7 - x)

    # Get the piece at the given square
    piece = board.piece_at(square)

    if piece:
        # Determine the yor of the piece
        color = "white" if piece.color == chess.WHITE else "black"

        # Determine the type of the piece
        piece_type = {
            chess.PAWN: "pawn",
            chess.KNIGHT: "knight",
            chess.BISHOP: "bishop",
            chess.ROOK: "rook",
            chess.QUEEN: "queen",
            chess.KING: "king",
        }.get(piece.piece_type, "Unknown")

        return f"{piece_type}-{color}"
    else:
        return "empty"


def check_probability(board, inference_matrix):
    final_prob = 1

    for y in range(8):
        for x in range(8):
            board_label = get_piece_name(board, x, y)

            final_prob *= inference_matrix[y][x][board_label]

    return final_prob


class PieceIdentifier(Node):
    def __init__(self):
        super().__init__("piece_identifier")
        
        self.declare_parameter(
            "initial_game_state",
            chess.STARTING_FEN,
            ParameterDescriptor(description="FEN string describing the initial game state"),
        )

        self.model = YOLOModel("cobot_best.pt")
        self.br = CvBridge()

        self.last_img = None
        self.last_depth_img = None
        self.board = chess.Board(self.get_parameter('initial_game_state').value)

        # Subscriptions
        self.board_img_sub = self.create_subscription(
            Image, "chessboard/image_raw", self.board_img_cb, 10
        )
        self.depth_img_sub = self.create_subscription(
            Image, "kinect2/depth/image_raw", self.color_img_cb, 10
        )
        self.restart_game_sub = self.create_subscription(
            GameConfig, "chess/restart_game", self.restart_game_cb, 10
        )
        self.clock_btn_sub = self.create_subscription(ClockButtons, "chess/clock_buttons", 10)

        # Publishers
        self.game_state_pub = self.create_publisher(
            FullFEN,
            "chess/game_state",
            10,
        )
        self.image_publisher = self.create_publisher(Image, "/chessboard/annotated/image_raw", 10)

        self.game_state_pub.publish(FullFEN(fen=self.board.fen()))

    def 


def main(args=None):
    rclpy.init(args=args)
    image_processor = PieceIdentifier()
    rclpy.spin(image_processor)
    # Cleanup
    image_processor.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
