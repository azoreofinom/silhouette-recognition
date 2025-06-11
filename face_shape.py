import cv2
import numpy as np
import os
import math
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from align import align_images
from sklearn.neighbors import KNeighborsClassifier
from pyefd import elliptic_fourier_descriptors
from sklearn.metrics import classification_report, accuracy_score, roc_curve, roc_auc_score, auc
import matplotlib.pyplot as plt
from itertools import combinations
from scipy.spatial.distance import euclidean
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from skimage.morphology import skeletonize
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import mediapipe as mp
from skimage.feature import hog
import cv2
import numpy as np

def find_head_box(binary_image):
    x=0
    y=0
    # Iterate through the rows of the image
    for row_index, row in enumerate(binary_image):
        # Check if the row contains any white pixel
        if np.any(row == 255):
            # Find the index of the first white pixel in the row
            y=max(row_index -10,0)
            column_index = np.argmax(row == 255)
            x = column_index - 125
            break

    h = 210
    w = 250

    return x,y,w,h

def get_head_features(img):
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)
    M = cv2.moments(contour)
    hu_moments = cv2.HuMoments(M).flatten()
    coeffs = elliptic_fourier_descriptors(np.squeeze(contour), order=2, normalize=True)
    # features = hu_moments
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    [vx, vy, x, y] = cv2.fitLine(contour, cv2.DIST_L2, 0, 0.01, 0.01)

    x, y, width, height = cv2.boundingRect(contour)

    bounding_features = np.array([height,width,area,perimeter])
    features = np.concatenate((bounding_features, hu_moments, coeffs.flatten()[3:]))
    return features


def pose_estimation(image):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=True)
    # Convert the image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Perform pose estimation
    results = pose.process(image_rgb)

    # Extract keypoints
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        height, width, _ = image.shape

        # Convert keypoints to pixel coordinates
        keypoints = np.array([(int(landmark.x * width), int(landmark.y * height)) for landmark in landmarks])

        # Define limb pairs (e.g., shoulder to elbow, elbow to wrist, hip to knee, knee to ankle)
        limbs = {
            'torso'          :(mp_pose.PoseLandmark.RIGHT_SHOULDER,mp_pose.PoseLandmark.RIGHT_HIP),
            'right_upper_leg': (mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_KNEE),
            'right_lower_leg': (mp_pose.PoseLandmark.RIGHT_KNEE, mp_pose.PoseLandmark.RIGHT_ANKLE)
        }

        # Calculate limb lengths
        limb_lengths = {}
        for limb_name, (point1, point2) in limbs.items():
            p1 = keypoints[point1.value]
            p2 = keypoints[point2.value]
            length = np.linalg.norm(np.array(p1) - np.array(p2))
            limb_lengths[limb_name] = length

        # Print limb lengths
        for limb_name, length in limb_lengths.items():
            print(f'{limb_name}: {length:.2f} pixels')

    # Release the pose model
    pose.close()

    # Optionally, display the image with keypoints
    for point in keypoints:
        cv2.circle(image, point, 5, (0, 255, 0), -1)

    return limb_lengths


def remove_artifacts(binary_image):
    # Find all connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image, connectivity=8)

    # The first label is the background, so we start with the second one
    largest_component = 1
    largest_size = stats[1, cv2.CC_STAT_AREA]

    # Iterate through the components to find the largest one
    for i in range(2, num_labels):
        if stats[i, cv2.CC_STAT_AREA] > largest_size:
            largest_size = stats[i, cv2.CC_STAT_AREA]
            largest_component = i

    # Create an output image to display only the largest component
    largest_component_mask = np.zeros(binary_image.shape, dtype=np.uint8)
    largest_component_mask[labels == largest_component] = 255
    return largest_component_mask


def extract_features(body):
    contours, _ = cv2.findContours(body, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    [vx, vy, x, y] = cv2.fitLine(contour, cv2.DIST_L2, 0, 0.01, 0.01)

    # Calculate moments
    M = cv2.moments(contour)

    # Calculate centroid coordinates
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    else:
        # If the area (m00) is zero, which means the binary image has no white pixels,
        # set the centroid to some default value or handle the situation as needed.
        cX, cY = 0, 0

    # Calculate Hu Moments
    hu_moments = cv2.HuMoments(M).flatten()
    coeffs = elliptic_fourier_descriptors(np.squeeze(contour), order=10, normalize=True)

    # Get bounding box of the largest contour
    x, y, width, height = cv2.boundingRect(contour)
    # Calculate the ratio of height to width
    ratio = height / width
    if ratio>3.8:
        orientation = 1
    else:
        orientation = 0


    bounding_features = np.array([height])
    features = bounding_features
    return features


def extract_body(image):
    #  Read the Image
    background = cv2.imread('background.jpg')

    image, h = align_images(background, image)
    foreground_mask = cv2.absdiff(image, background)
    foreground_mask = cv2.cvtColor(foreground_mask,cv2.COLOR_BGR2GRAY)

    #cropping to only include green background (cutting shoes off as can't extract them properly)
    foreground_mask = foreground_mask[320:1580, 750:1800]

    threshold = 40 
    _, foreground_mask_binary = cv2.threshold(foreground_mask,threshold, 255, cv2.THRESH_BINARY)
    return foreground_mask_binary

def extract_full_features(image):
   
    fd, hog_image = hog(
        image,
        orientations=8,
        pixels_per_cell=(16, 16),
        cells_per_block=(1, 1),
        visualize=True,
        channel_axis=-1,
    )
    print(fd.shape)

    # Flatten HOG features
    hog_features_flat = fd.flatten()

    feature_vector = hog_features_flat
    return feature_vector

def eval(model,standard_scaler):
    predictions = model.predict(X_test_pca)
    print(predictions)
    print(classification_report(test_labels, predictions))
    print(model.score(X_test_pca,test_labels))

labels = []
features = []
folder_path = 'training'
i = 0
# Loop through each file in the folder
for filename in os.listdir(folder_path):
    if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff','.JPG')):
        # Construct the full path to the image
        image_path = os.path.join(folder_path, filename)
        code = filename.split(".")[0]
        label = code[4:7]
        labels.append(label)
        # Load the image
        image = cv2.imread(image_path)
        # Check if the image was loaded correctly
        if image is None:
            print(f"Failed to load image {image_path}")
            continue



        processed_image = remove_artifacts(extract_body(image))
        x,y,w,h = find_head_box(processed_image)
        head = processed_image[y:y+h, x:x+w]


        head_features = get_head_features(head)


        limbs = list(pose_estimation(image).values())
        feature_vector = extract_features(processed_image)
        subject_image = image[320:1580, 750:1800]

        feature_vector = np.concatenate((feature_vector, head_features))
        features.append(feature_vector)

# Close all OpenCV windows
cv2.destroyAllWindows()
predictions = []
test_features = []
test_labels = ['001', '001', '002', '002', '003', '003', '004', '004', '005', '005', '006', '006', '007', '007',
               '008', '008', '009', '009', '010', '010', '011', '011']


folder_path = 'test'
for filename in os.listdir(folder_path):
    if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.JPG')):
        image_path = os.path.join(folder_path, filename)
        image = cv2.imread(image_path)

        processed_image = remove_artifacts(extract_body(image))

        x, y, w, h = find_head_box(processed_image)
        head = processed_image[y:y + h, x:x + w]

        head_features = get_head_features(head)


        feature_vector = extract_features(processed_image)
        limbs = list(pose_estimation(image).values())
        subject_image = image[320:1580, 750:1800]


        feature_vector = np.concatenate((feature_vector, head_features))
        test_features.append(feature_vector)


scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)
scaled_test_features = scaler.transform(test_features)
print(scaled_features[0])

# Perform PCA
pca = PCA(n_components=20)  # Reduce to 20 principal components

X_train_pca = scaled_features
X_test_pca = scaled_test_features

param_grid = {'C': [1e-06,0.00001,0.0001,0.001,0.01,0.1, 1, 10, 100, 1000],
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001,'auto'],
              'kernel': ['rbf','linear','poly']}


svm = SVC()
best_accuracy = 0
best_params = None

# Iterate over parameter combinations
for C in param_grid['C']:
    for gamma in param_grid['gamma']:
        for kernel in param_grid['kernel']:
            # Train SVM model with current parameters
            svm = SVC(C=C, gamma=gamma, kernel=kernel)
            svm.fit(X_train_pca,labels)

            # Evaluate model
            y_pred = svm.predict(X_test_pca)
            accuracy = accuracy_score(test_labels, y_pred)

            # Print current parameters and accuracy
            print(f'Parameters: C={C}, gamma={gamma}, kernel={kernel}, Accuracy={accuracy}')

            # Update best accuracy and parameters if current model is better
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_params = {'C': C, 'gamma': gamma, 'kernel': kernel}

print(f'Best parameters: {best_params}, Best accuracy: {best_accuracy}')
svm = SVC(probability=True, C=1e-6,gamma=1, kernel='rbf')
svm.fit(X_train_pca,labels)
predictions = svm.predict(X_test_pca)
print(classification_report(test_labels, predictions))

y_probs = svm.predict_proba(X_test_pca)
print(y_probs[:5])
# Calculate cumulative match characteristic curve
def cmc_curve(y_true, y_scores):
    n_samples, n_classes = y_scores.shape
    cmc = np.zeros(n_classes)
    for i in range(n_samples):
        sorted_indices = np.argsort(y_scores[i])[::-1]
        for j in range(n_classes):
            if y_true[i] == sorted_indices[j]:
                cmc[j:] += 1
                break
    cmc /= n_samples
    return cmc
# Compute CMC curve
cmc = cmc_curve(test_labels, y_probs)

# Plot CMC curve
plt.plot(np.arange(1, len(cmc)+1), cmc, marker='o')
plt.xlabel('Rank')
plt.ylabel('Recognition Rate')
plt.title('Cumulative Match Characteristic (CMC) Curve')
plt.grid(True)
plt.show()








knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train_pca, labels)
eval(knn,scaler)

intra_class_distances = []
for subject in set(labels):
    subject_features = [features[i] for i in range(len(labels)) if labels[i] == subject]
    for f1, f2 in combinations(subject_features, 2):
        intra_class_distances.append(euclidean(f1, f2))

# Calculate Inter-Class Variations
inter_class_distances = []
for subject1, subject2 in combinations(set(labels), 2):
    subject1_features = [features[i] for i in range(len(labels)) if labels[i] == subject1]
    subject2_features = [features[i] for i in range(len(labels)) if labels[i] == subject2]
    for f1 in subject1_features:
        for f2 in subject2_features:
            inter_class_distances.append(euclidean(f1, f2))

# Plot Histograms
plt.hist(intra_class_distances, bins=50, alpha=0.5, label='Intra-Class', color='blue', density=True)
plt.hist(inter_class_distances, bins=50, alpha=0.5, label='Inter-Class', color='red', density=True)
plt.xlabel('Distance')
plt.ylabel('Frequency')
plt.legend(loc='upper right')
plt.title('Intra-Class and Inter-Class Distance Distributions')
plt.show()

def create_pairs_and_labels2(features, labels):
    pairs = []
    pair_labels = []

    for class_id in np.unique(labels):
        class_indices = [index for index, value in enumerate(labels) if value == class_id]
        other_indices = [index for index, value in enumerate(labels) if value != class_id]

        # Generate genuine pair (same class)
        genuine_pair_generated = False
        for idx1, idx2 in combinations(class_indices, 2):
            pairs.append((features[idx1], features[idx2]))
            pair_labels.append(1)  # Same class
            genuine_pair_generated = True
            break

        # Generate impostor pair (different classes)
        impostor_pair_generated = False
        for idx1 in class_indices:
            for idx2 in other_indices:
                pairs.append((features[idx1], features[idx2]))
                pair_labels.append(0)  # Different classes
                impostor_pair_generated = True
                break
            if impostor_pair_generated:
                break

    return np.array(pairs), np.array(pair_labels)


def create_verification_data2(train_features, train_labels, test_features, test_labels):
    pairs = []
    pair_labels = []

    for class_id in np.unique(test_labels):
        test_class_indices = [index for index, value in enumerate(test_labels) if value == class_id]
        train_class_indices = [index for index, value in enumerate(train_labels) if value == class_id]
        other_train_indices = [index for index, value in enumerate(train_labels) if value != class_id]

        genuine_pair_generated = False
        for test_idx in test_class_indices:
            for train_idx in train_class_indices:
                pairs.append((test_features[test_idx], train_features[train_idx]))
                pair_labels.append(1)  # Different classes
                genuine_pair_generated = True
                break
            if genuine_pair_generated:
                break
        # Generate impostor pair (different classes)
        impostor_pair_generated = False
        for test_idx in test_class_indices:
            for train_idx in other_train_indices:
                pairs.append((test_features[test_idx], train_features[train_idx]))
                pair_labels.append(0)  # Different classes
                impostor_pair_generated = True
                break
            if impostor_pair_generated:
                break

    return np.array(pairs), np.array(pair_labels)

# Create pairs and labels
pairs, pair_labels = create_pairs_and_labels2(X_train_pca, labels)
test_pairs, test_pair_labels = create_verification_data2(X_train_pca, labels,X_test_pca, test_labels)

# test_pairs, test_pair_labels = create_pairs_and_labels2(X_test_pca, test_labels)

# Prepare data for binary classification
X_train = np.array([np.concatenate([pair[0], pair[1]]) for pair in pairs])
X_test = np.array([np.concatenate([pair[0], pair[1]]) for pair in test_pairs])

print(X_train.shape)
print(X_test.shape)
# Train a binary classifier
svm = SVC(probability=True, C=1e-6,gamma=1, kernel='rbf')
svm.fit(X_train, pair_labels)


# Get decision scores (probability estimates) - distance of the samples to the decision boundary
y_scores = svm.decision_function(X_test)

# Compute false positive rate, true positive rate, and thresholds
fpr, tpr, thresholds = roc_curve(test_pair_labels,y_scores)

# Compute EER
eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
eer_threshold = interp1d(fpr, thresholds)(eer)
print(f'EER: {eer}, Threshold at EER: {eer_threshold}')


y_pred_eer = (y_scores >= eer_threshold).astype(int)
print(y_pred_eer)
print(test_pair_labels)
ccr_at_eer = np.mean(y_pred_eer == test_pair_labels)
print(f'CCR at EER: {ccr_at_eer}')



# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = %0.2f)' % auc(fpr, tpr))
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
plt.scatter(eer, 1-eer, color='red', label=f'EER = {eer:.2f}')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()


# Predict and evaluate
y_pred = svm.predict(X_test)
accuracy = accuracy_score(test_pair_labels, y_pred)
roc_auc = roc_auc_score(test_pair_labels, svm.predict_proba(X_test)[:, 1])

print(f'Accuracy: {accuracy}')
print(f'ROC AUC: {roc_auc}')