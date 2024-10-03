import argparse
from pathlib import Path

import cv2
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from sahi.utils.yolov8 import download_yolov8s_model

from ultralytics.utils.files import increment_path


def run(weights="/mnt/new/weights/train/weights/best.pt", source="images/", view_img=False, save_img=False, exist_ok=False):
    """
    Run object detection on images in a directory using YOLOv8 and SAHI.

    Args:
        weights (str): Model weights path.
        source (str): Directory containing image files.
        view_img (bool): Show results.
        save_img (bool): Save results.
        exist_ok (bool): Overwrite existing files.
    """

    # Check source path
    source_path = Path(source)
    if not source_path.exists() or not source_path.is_dir():
        raise FileNotFoundError(f"Source path '{source}' does not exist or is not a directory.")

    yolov8_model_path = f"{weights}"
    download_yolov8s_model(yolov8_model_path)
    detection_model = AutoDetectionModel.from_pretrained(
        model_type="yolov8", model_path=yolov8_model_path, confidence_threshold=0.05, device="0", image_size = 2048
    )

    # Output setup
    save_dir = increment_path(Path("ultralytics_results_with_sahi") / "exp", exist_ok)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Process each image in the directory
    for image_path in source_path.glob("*.png"):
        frame = cv2.imread(str(image_path))
        if frame is None:
            print(f"Failed to load image '{image_path}'. Skipping.")
            continue

        results = get_sliced_prediction(
            frame, detection_model, slice_height=312, slice_width=312, overlap_height_ratio=0.2, overlap_width_ratio=0.2
        )
        object_prediction_list = results.object_prediction_list
        print(object_prediction_list)
        boxes_list = []
        clss_list = []
        for ind, _ in enumerate(object_prediction_list):
            boxes = (
                object_prediction_list[ind].bbox.minx,
                object_prediction_list[ind].bbox.miny,
                object_prediction_list[ind].bbox.maxx,
                object_prediction_list[ind].bbox.maxy,
            )
            clss = object_prediction_list[ind].category.name
            print(object_prediction_list[ind])
            boxes_list.append(boxes)
            clss_list.append(clss)

        for box, cls in zip(boxes_list, clss_list):
            x1, y1, x2, y2 = box
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (56, 56, 255), 2)
            label = str(cls)
            t_size = cv2.getTextSize(label, 0, fontScale=0.6, thickness=1)[0]
            cv2.rectangle(
                frame, (int(x1), int(y1) - t_size[1] - 3), (int(x1) + t_size[0], int(y1) + 3), (56, 56, 255), -1
            )
            cv2.putText(
                frame, label, (int(x1), int(y1) - 2), 0, 0.6, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA
            )

        if view_img:
            cv2.imshow(image_path.stem, frame)
        if save_img:
            output_path = save_dir / f"{image_path.stem}_output.jpg"
            cv2.imwrite(str(output_path), frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()


def parse_opt():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="yolov8n.pt", help="initial weights path")
    parser.add_argument("--source", type=str, required=True, help="directory containing images")
    parser.add_argument("--view-img", action="store_true", help="show results")
    parser.add_argument("--save-img", action="store_true", help="save results")
    parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok, do not increment")
    return parser.parse_args()


def main(opt):
    """Main function."""
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
