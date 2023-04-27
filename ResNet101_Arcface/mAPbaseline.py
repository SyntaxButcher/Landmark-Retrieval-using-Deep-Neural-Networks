import csv
from metrics import MeanAveragePrecision
from metrics import MeanPrecisions
import numpy as np

def read_ground_truth_csv(file_path):
    ground_truth = {}
    with open(file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip header
        for row in reader:
            ground_truth[row[0]] = row[1].split()
    return ground_truth


def read_predicted_csv(file_path):
    predictions = {}
    with open(file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip header
        for row in reader:
            # Remove '.jpg' extension from the image name in the first column
            key = row[0][:-4]

            # Remove '.jpg' extension from image names in the second column
            image_names = [img_name[:-4] for img_name in row[1].split()]
            predictions[key] = image_names
    return predictions

def main():
    # Read ground truth and predicted CSV files
    ground_truth = read_ground_truth_csv('groundTruth.csv')
    predictions = read_predicted_csv('predictions.csv')

    # Compute Mean Average Precision @ 10
    map_10 = MeanAveragePrecision(predictions, ground_truth, 10) * 100
    print('Mean Average Precision: %f' % map_10)

    # Compute mean precision for all queries
    mean_precisions = MeanPrecisions(predictions, ground_truth, 10)
    mean_precision = np.mean(mean_precisions) * 100
    print('Mean Precision @ 10: %f' % mean_precision)

    


if __name__ == '__main__':
    main()