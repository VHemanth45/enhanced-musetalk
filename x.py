import argparse
import os
import cv2
import numpy as np
from gfpgan.utils import GFPGANer
from tqdm import tqdm
import logging

class VideoEnhancer:
    def __init__(self, model_type="GFPGAN", model_path=None):
        self.model_type = model_type
        self.model_path = model_path or './external_models/GFPGAN/experiments/pretrained_models/GFPGANv1.3.pth'
        self.superres_model = None
        self.logger = self._setup_logger()

    def _setup_logger(self):
        """Set up logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('video_enhancement.log')
            ]
        )
        return logging.getLogger(__name__)

    def load_model(self):
        """Load the super-resolution model"""
        try:
            if self.model_type == "GFPGAN":
                if not os.path.exists(self.model_path):
                    raise FileNotFoundError(f"Model not found at {self.model_path}")
                
                self.superres_model = GFPGANer(
                    model_path=self.model_path,
                    upscale=2,
                    arch='clean',
                    channel_multiplier=2
                )
                self.logger.info(f"Successfully loaded {self.model_type} model")
            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise

    def process_frame(self, frame):
        """Process a single frame"""
        if frame is None:
            raise ValueError("Invalid frame: Frame is None")

        try:
            enhance_result = self.superres_model.enhance(
                frame,
                has_aligned=False,
                only_center_face=False
            )

            if isinstance(enhance_result, tuple):
                enhanced_frame = enhance_result[0]
            else:
                enhanced_frame = enhance_result

            if enhanced_frame is None or not isinstance(enhanced_frame, np.ndarray):
                raise ValueError("Invalid enhanced frame")

            return enhanced_frame

        except Exception as e:
            self.logger.warning(f"Frame enhancement failed: {str(e)}")
            return frame

    def enhance_video(self, input_path, output_path):
        """Enhance the entire video"""
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input video not found: {input_path}")

        video = cv2.VideoCapture(input_path)
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(video.get(cv2.CAP_PROP_FPS))
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Ensure output directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        output_video = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*'mp4v'),
            fps,
            (width, height)
        )

        self.logger.info(f"Starting video enhancement: {input_path}")
        self.logger.info(f"Video properties: {total_frames} frames, {fps} FPS, {width}x{height}")

        try:
            with tqdm(total=total_frames, desc="Processing frames") as pbar:
                while True:
                    ret, frame = video.read()
                    if not ret:
                        break

                    enhanced_frame = self.process_frame(frame)
                    
                    # Ensure correct dimensions
                    if enhanced_frame.shape[:2] != (height, width):
                        enhanced_frame = cv2.resize(enhanced_frame, (width, height))

                    output_video.write(enhanced_frame)
                    pbar.update(1)

        except Exception as e:
            self.logger.error(f"Error during video processing: {str(e)}")
            raise
        finally:
            video.release()
            output_video.release()
            self.logger.info(f"Enhanced video saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Video enhancement using GFPGAN")
    parser.add_argument(
        "--superres",
        choices=["GFPGAN"],
        default="GFPGAN",
        help="Super-resolution method to use"
    )
    parser.add_argument(
        "-i",
        "--input_video",
        default="./results/input_inputa.mp4",
        help="Path to input video file"
    )
    parser.add_argument(
        "-o",
        "--output_video",
        required=True,
        help="Path to output video file"
    )
    parser.add_argument(
        "--model_path",
        help="Path to model file (optional)"
    )

    args = parser.parse_args()

    try:
        enhancer = VideoEnhancer(
            model_type=args.superres,
            model_path=args.model_path
        )
        enhancer.load_model()
        enhancer.enhance_video(args.input_video, args.output_video)
    except Exception as e:
        logging.error(f"Enhancement failed: {str(e)}")
        exit(1)

if __name__ == "__main__":
    main()