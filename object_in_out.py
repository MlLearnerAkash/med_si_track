
import argparse
from pathlib import Path
import os
import cv2
import numpy as np
from shapely.geometry import Polygon
from shapely.geometry.point import Point

from ultralytics import YOLO, solutions
from ultralytics.utils.files import increment_path
from ultralytics.utils.plotting import Annotator, colors

import json



# Directory paths for saving images and JSON files
output_dir = "output"
image_dir = os.path.join(output_dir, "images")
json_dir = os.path.join(output_dir, "json")

# Create the directories if they don't exist
os.makedirs(image_dir, exist_ok=True)
os.makedirs(json_dir, exist_ok=True)

def mouse_callback(event, x, y, flags, param):
    """Mouse event callback for manipulating regions."""
    pass  # Keep the mouse callback if required later




def run(
    weights="/root/ws/med_si_track/custom_needle_large/large/weights/best.pt",
    source="/mnt/data/training_6.avi",
    device="cpu",
    view_img=False,
    save_img=False,
    exist_ok=False,
    classes=None,
    line_thickness=2,
    track_thickness=2,
    region_thickness=2,
):
    """
    Run YOLOv8 with object counting and tracking using the ultralytics `solutions.ObjectCounter`.

    Args:
        weights (str): Path to model weights.
        source (str): Path to the video file.
        device (str): Device to use: 'cpu' or 'cuda'.
        view_img (bool): Whether to show the video while processing.
        save_img (bool): Whether to save the annotated output video.
        exist_ok (bool): Whether to overwrite existing output.
        classes (list): Classes to detect and track.
        line_thickness (int): Line thickness for annotations.
        track_thickness (int): Line thickness for tracking lines.
        region_thickness (int): Thickness of region boundary lines.
    """
    # Check environment and dependencies

    # Initialize the YOLO model
    model = YOLO(weights)

    # Load video
    cap = cv2.VideoCapture(source)
    assert cap.isOpened(), "Error reading video file"
    
    # Get video properties
    w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

    # Define points for a region of interest (ROI) or line
    #qAdd 30 to x and y coordinates to shift the ROI to the right and down
    line_points = [(50, 20), (50, 420), (550, 420), (550, 20)]  # ROI coordinates

    # Define object classes to count (use class IDs)
    classes_to_count = []  # For example, counting class 0 (persons, etc.)

    # Initialize video writer if saving
    if save_img:
        save_dir = increment_path(Path("ultralytics_rc_output") / "exp", exist_ok)
        save_dir.mkdir(parents=True, exist_ok=True)
        video_writer = cv2.VideoWriter(str(save_dir / f"{Path(source).stem}.mp4"), 
                                       cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    # Initialize ObjectCounter from ultralytics.solutions
    counter = solutions.ObjectCounter(
        view_img=view_img,  # Display the video frame while processing
        reg_pts=line_points,  # ROI for counting objects
        classes_names=model.names,  # YOLO class names
        draw_tracks=True,  # Draw tracking lines on objects
        line_thickness=line_thickness  # Line thickness for tracking
    )

    frame_index = 0  # Initialize frame index

    # Process video frame-by-frame
    while cap.isOpened():
        success, im0 = cap.read()
        if not success:
            print("Video frame is empty or video processing has completed.")
            break

        # Perform object tracking with YOLO, filtering by specified classes
        tracks = model.track(im0, persist=True, conf=0.25, iou=0.25, show=False, imgsz=2480, tracker="botsort.yaml")

        # Count objects in the current frame using the ObjectCounter
        im0, item_status = counter.start_counting(im0, tracks)

        # Save the JSON file for the current frame
        json_path = os.path.join(json_dir, f"frame_{frame_index:05d}.json")
        with open(json_path, "w") as f:
            json.dump(item_status, f, indent=4)
        
        # Save the annotated frame as an image
        image_path = os.path.join(image_dir, f"frame_{frame_index:05d}.jpg")
        cv2.imwrite(image_path, im0)
        
        print(f"Frame {frame_index} processed and saved: {json_path}, {image_path}")

        # Save the annotated frame to output video (optional)
        if save_img:
            video_writer.write(im0)

        # Increment frame index for the next iteration
        frame_index += 1

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release resources
    cap.release()
    if save_img:
        video_writer.release()
    cv2.destroyAllWindows()


def parse_opt():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="/root/ws/med_si_track/custom_needle/train7/weights/best.pt", help="model weights path")
    parser.add_argument("--source", type=str, default="/root/ws/med_si_track/test_data/training_6.avi", help="path to video file")
    parser.add_argument("--device", type=str, default="cpu", help="device to use (cpu or cuda)")
    parser.add_argument("--view-img", action="store_true", help="display results")
    parser.add_argument("--save-img", action="store_true", help="save results")
    parser.add_argument("--exist-ok", action="store_true", help="overwrite existing results")
    parser.add_argument("--classes", nargs="+", type=int, help="filter by class")
    parser.add_argument("--line-thickness", type=int, default=2, help="bounding box thickness")
    parser.add_argument("--track-thickness", type=int, default=2, help="tracking line thickness")
    parser.add_argument("--region-thickness", type=int, default=2, help="region boundary thickness")
    return parser.parse_args()


def main(opt):
    """Main function."""
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
