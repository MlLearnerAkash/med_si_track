# from ultralytics import YOLO
# from map_label import map_classes, map_model_output
# # Load your trained models
# model_base = YOLO('/root/ws/med_si_track/weights_2806/best.pt')
# # model_needle = YOLO('/root/ws/med_si_track/opervu-v8-140924-only-needle-training/train/weights/best.pt')
# model_needle_2 = YOLO('/root/ws/med_si_track/opervu-220824-needle-training/train2/weights/best.pt')
# # model_needle_sponge = YOLO('/root/ws/med_si_track/opervu-310824-needle-sponge-training/train/weights/best.pt')



# # print("Model Base::", model_base.model.names)
# # print("model_needle::", model_needle_2.model.names)

# map_classes = map_classes(model_base.model.names, model_needle_2.model.names)


# # Run inference on an image for each model
# results_base = model_base.predict('/mnt/data/needle_images/9798.jpg')
# # results_needle = model_needle.predict('path/to/image.jpg')
# results_needle_2 = model_needle_2.predict('/mnt/data/needle_images/9798.jpg')
# # results_needle_sponge = model_needle_sponge.predict('path/to/image.jpg')

# # Extract labels and scores
# def extract_labels_and_scores(results):
#     labels = results[0].boxes.cls.cpu().numpy()
#     scores = results[0].boxes.conf.cpu().numpy()
#     return labels, scores

# labels_base, scores_base = extract_labels_and_scores(results_base)
# # labels_needle, scores_needle = extract_labels_and_scores(results_needle)
# labels_needle_2, scores_needle_2 = extract_labels_and_scores(results_needle_2)
# labels_needle_2 = map_model_output(labels_needle_2, map_classes)
# # labels_needle_sponge, scores_needle_sponge = extract_labels_and_scores(results_needle_sponge)

# # Combine results (example: weighted average based on confidence scores)
# def combine_results(labels_list, scores_list):
#     combined_labels = []
#     combined_scores = []
#     for i in range(len(labels_list[0])):
#         combined_label = labels_list[0][i]  # Assuming same label across models
#         combined_score = (scores_list[0][i] + scores_list[1][i] ) / 2
#         combined_labels.append(combined_label)
#         combined_scores.append(combined_score)
#     return combined_labels, combined_scores

# combined_labels, combined_scores = combine_results([labels_base, labels_needle_2],
#                                                    [scores_base, scores_needle_2])

# print("Combined Labels:", combined_labels)
# print("Combined Scores:", combined_scores)

##NOTE:V-2
'''
from ultralytics import YOLO
from map_label import map_classes, map_model_output
import numpy as np

# Load your trained models
model_base = YOLO('/root/ws/med_si_track/weights_2806/best.pt')
model_needle_2 = YOLO('/root/ws/med_si_track/opervu-v8-140924-only-needle-training/train/weights/best.pt')

# Map classes between the models
map_classes = map_classes(model_base.model.names, model_needle_2.model.names)

# Run inference on an image for each model
results_base = model_base.predict('/mnt/data/needle_images/9798.jpg')
results_needle_2 = model_needle_2.predict('/mnt/data/needle_images/9798.jpg')

# Function to extract labels and scores
def extract_labels_and_scores(results):
    labels = results[0].boxes.cls.cpu().numpy()
    scores = results[0].boxes.conf.cpu().numpy()
    return labels, scores

# Extract labels and scores for both models
labels_base, scores_base = extract_labels_and_scores(results_base)
labels_needle_2, scores_needle_2 = extract_labels_and_scores(results_needle_2)

# Map the labels from the needle model to the base model's classes
labels_needle_2 = map_model_output(labels_needle_2, map_classes)

# Function to combine results from both models
def combine_results(labels1, scores1, labels2, scores2):
    combined_labels = []
    combined_scores = []

    # Dictionary to store label and aggregated score
    score_dict = {}

    # Process first set of labels and scores
    for i in range(len(labels1)):
        label = labels1[i]
        score = scores1[i]
        if label in score_dict:
            score_dict[label].append(score)
        else:
            score_dict[label] = [score]

    # Process second set of labels and scores
    for i in range(len(labels2)):
        label = labels2[i]
        score = scores2[i]
        if label in score_dict:
            score_dict[label].append(score)
        else:
            score_dict[label] = [score]

    # Combine results by averaging scores for common labels
    for label, score_list in score_dict.items():
        combined_labels.append(label)
        combined_scores.append(np.mean(score_list))  # Average scores for the same label

    return combined_labels, combined_scores

# Combine labels and scores from both models
combined_labels, combined_scores = combine_results(labels_base, scores_base, labels_needle_2, scores_needle_2)

print("Combined Labels:", combined_labels)
print("Combined Scores:", combined_scores)
'''
from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.cfg import get_cfg

args = get_cfg("./test_cfg.yaml")

validator = DetectionValidator(args=args)
validator(model=dict(args)['model'])
