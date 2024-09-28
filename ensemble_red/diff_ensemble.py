from ultralytics import YOLO
# from map_label import map_classes, map_model_output
# Load your trained models
model_base = YOLO('/root/ws/med_si_track/weights_2806/best.pt')
# model_needle = YOLO('/root/ws/med_si_track/opervu-v8-140924-only-needle-training/train/weights/best.pt')
model_needle_2 = YOLO('/root/ws/med_si_track/opervu-v8-140924-only-needle-training/train/weights/best.pt')
# model_needle_sponge = YOLO('/root/ws/med_si_track/opervu-310824-needle-sponge-training/train/weights/best.pt')



# print("Model Base::", model_base.model.names)
# print("model_needle::", model_needle_2.model.names)

# map_classes = map_classes(model_base.model.names, model_needle_2.model.names)


# Run inference on an image for each model
results_base = model_base.predict('/mnt/data/needle_images/9798.jpg')
print("Model names>>base_model>>>", results_base)
# results_needle = model_needle.predict('path/to/image.jpg')
results_needle_2 = model_needle_2.predict('/mnt/data/needle_images/9798.jpg')
print("Model names>>needle_2>>>", results_needle_2)

# results_needle_sponge = model_needle_sponge.predict('path/to/image.jpg')

# Extract labels and scores
def extract_labels_and_scores(results):
    labels = results[0].boxes.cls.cpu().numpy()
    scores = results[0].boxes.conf.cpu().numpy()
    return labels, scores

labels_base, scores_base = extract_labels_and_scores(results_base)
print(">>>>>", labels_base)
# labels_needle, scores_needle = extract_labels_and_scores(results_needle)
labels_needle_2, scores_needle_2 = extract_labels_and_scores(results_needle_2)
print("<<<<<<<<", labels_needle_2)
# labels_needle_2 = map_model_output(labels_needle_2, map_classes)
# labels_needle_sponge, scores_needle_sponge = extract_labels_and_scores(results_needle_sponge)

# Combine results (example: weighted average based on confidence scores)
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