import cv2
import numpy as np
import os
#Import math Library
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
            # 'left_upper_arm': (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW),
            # 'left_lower_arm': (mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST),
            # 'right_upper_arm': (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW),
            # 'right_lower_arm': (mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_WRIST),
            # 'left_upper_leg': (mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_KNEE),
            # 'left_lower_leg': (mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.LEFT_ANKLE),
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
    # for i in range(0, 7):
    #     hu_moments[i] = -1 * math.copysign(1.0, hu_moments[i]) * math.log10(abs(hu_moments[i]))


    coeffs = elliptic_fourier_descriptors(np.squeeze(contour), order=10, normalize=True)

    # Get bounding box of the largest contour
    x, y, width, height = cv2.boundingRect(contour)
    # Calculate the ratio of height to width
    ratio = height / width
    # print(f"ratio:{ratio}")

    if ratio>3.8:
        orientation = 1
    else:
        orientation = 0


    bounding_features = np.array([height])
    # features = bounding_features
    # features = np.concatenate((bounding_features, hu_moments))
    # features = np.concatenate((bounding_features, coeffs.flatten()[3:]))
    features = np.concatenate((bounding_features,hu_moments))
    features = np.concatenate((bounding_features,hu_moments, coeffs.flatten()[3:]))
    return x,y,width,height, features


def extract_body(image):
    # Step 1: Read the Image
    # image = cv2.imread('training/020z071pf.jpg')
    background = cv2.imread('background.jpg')

    image, h = align_images(background, image)
    # img_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    # back_gray = cv2.cvtColor(background,cv2.COLOR_BGR2GRAY)

    foreground_mask = cv2.absdiff(image, background)
    # foreground_mask = cv2.absdiff(img_gray, back_gray)

    foreground_mask = cv2.cvtColor(foreground_mask,cv2.COLOR_BGR2GRAY)

    #cropping to only include green background (cutting shoes off as can't extract them properly...
    # foreground_mask = foreground_mask[320:, 750:1800]
    foreground_mask = foreground_mask[320:1580, 750:1800]
    # imS = cv2.resize(foreground_mask, (960, 540))

    # threshold = 40 works pretty well, use adaptive??
    _, foreground_mask_binary = cv2.threshold(foreground_mask,40, 255, cv2.THRESH_BINARY)
    return foreground_mask_binary

def extract_full_features(image):
    # print(image.shape)
    # sift = cv2.SIFT_create()
    # kp, des = sift.detectAndCompute(image, None)
    # print(des.shape)

    fd, hog_image = hog(
        image,
        orientations=8,
        pixels_per_cell=(16, 16),
        cells_per_block=(1, 1),
        visualize=True,
        channel_axis=-1,
    )
    print(fd.shape)

    # # Flatten SIFT descriptors
    # sift_descriptors_flat = des.reshape(-1)

    # Flatten HOG features
    hog_features_flat = fd.flatten()

    # Concatenate SIFT and HOG feature vectors
    # feature_vector = np.concatenate((sift_descriptors_flat, hog_features_flat))
    feature_vector = hog_features_flat
    return feature_vector

def eval(model,standard_scaler):
    predictions = model.predict(X_test_pca)
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

        limbs = list(pose_estimation(image).values())
        processed_image = remove_artifacts(extract_body(image))
        x,y,w,h,feature_vector = extract_features(processed_image)

        # subject_image = image[y:y + h, x:x + w]
        subject_image = image[320:1580, 750:1800]
        # other_features = extract_full_features(subject_image)
        # feature_vector = np.concatenate((feature_vector,np.array(limbs), other_features))
        feature_vector = np.concatenate((feature_vector, np.array(limbs)))
        features.append(feature_vector)

        print(i)
        i+=1


        # image = cv2.resize(image, (960, 740))
        # processed_image = cv2.resize(processed_image, (960, 740))
        # # Display the original and processed images
        # cv2.imshow('Original Image', image)
        # cv2.imshow('Processed Image', processed_image)
        # # cv2.imwrite('pose.jpg', image)
        # # cv2.imwrite('subtraction.jpg', processed_image)
        # # Wait for a key press to move to the next image
        # print(f"Displaying {filename}. Press any key to continue...")
        # cv2.waitKey(0)
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
        limbs = list(pose_estimation(image).values())
        processed_image = remove_artifacts(extract_body(image))
        x, y, w, h,feature_vector = extract_features(processed_image)
        # subject_image = image[y:y + h, x:x + w]
        subject_image = image[320:1580, 750:1800]
        # other_features = extract_full_features(subject_image)

        # feature_vector = np.concatenate((feature_vector, np.array(limbs),other_features))
        feature_vector = np.concatenate((feature_vector, np.array(limbs)))


        test_features.append(feature_vector)



scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)
scaled_test_features = scaler.transform(test_features)
print(scaled_features[0])

# Step 4: Perform PCA
pca = PCA(n_components=40)  # Reduce to 10 principal components
X_train_pca = pca.fit_transform(scaled_features)
X_test_pca = pca.transform(scaled_test_features)

# X_train_pca = scaled_features
# X_test_pca = scaled_test_features
param_grid = {'C': [1e-06,0.00001,0.0001,0.001,0.01,0.1, 1, 10, 100, 1000],
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001,'auto'],
              'kernel': ['rbf','linear','poly']}

# knn = KNeighborsClassifier(n_neighbors=1)
# knn.fit(X_train_pca, labels)
# eval(knn,scaler)

svm = SVC()
best_accuracy = 0
best_params = None

# Iterate over parameter combinations
for C in param_grid['C']:
    for gamma in param_grid['gamma']:
        for kernel in param_grid['kernel']:
            # Train SVM model with current parameters
            svm = SVC(C=C, gamma=gamma, kernel=kernel)
            # svm.fit(scaled_features, labels)
            svm.fit(X_train_pca,labels)

            # Evaluate model
            # y_pred = svm.predict(scaled_test_features)
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


intra_class_distances = []
for subject in set(labels):
    subject_features = [features[i] for i in range(len(labels)) if labels[i] == subject]
    for f1, f2 in combinations(subject_features, 2):
        intra_class_distances.append(euclidean(f1, f2))

# Step 2: Calculate Inter-Class Variations
inter_class_distances = []
for subject1, subject2 in combinations(set(labels), 2):
    subject1_features = [features[i] for i in range(len(labels)) if labels[i] == subject1]
    subject2_features = [features[i] for i in range(len(labels)) if labels[i] == subject2]
    for f1 in subject1_features:
        for f2 in subject2_features:
            inter_class_distances.append(euclidean(f1, f2))

# Step 3: Plot Histograms
plt.hist(intra_class_distances, bins=50, alpha=0.5, label='Intra-Class', color='blue', density=True)
plt.hist(inter_class_distances, bins=50, alpha=0.5, label='Inter-Class', color='red', density=True)
plt.xlabel('Distance')
plt.ylabel('Frequency')
plt.legend(loc='upper right')
plt.title('Intra-Class and Inter-Class Distance Distributions')
plt.show()



# svm = SVC(C=best_params['C'], gamma=best_params['gamma'], kernel=best_params['kernel'])
# svm.fit(scaled_features, labels)

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

# Create pairs and labels for binary classification
def create_pairs_and_labels(features, labels):
    pairs = []
    pair_labels = []
    for class_id in np.unique(labels):
        subject_id = str(class_id)

        class_indices = [index for index, value in enumerate(labels) if value == subject_id]
        other_indices = [index for index, value in enumerate(labels) if value != subject_id]

        # Generate genuine pairs (same class)
        for idx1, idx2 in combinations(class_indices, 2):
            pairs.append((features[idx1], features[idx2]))
            pair_labels.append(1)  # Same class
        # Generate impostor pairs (different classes)
        for idx1 in class_indices:
            for idx2 in other_indices:
                pairs.append((features[idx1], features[idx2]))
                pair_labels.append(0)  # Different classes

    return np.array(pairs), np.array(pair_labels)


def create_verification_data(train_features, train_labels, test_features, test_labels):
    pairs = []
    pair_labels = []

    for train_class_id in np.unique(train_labels):
        train_subject_id = str(train_class_id)

        train_class_indices = [index for index, value in enumerate(train_labels) if value == train_subject_id]
        test_class_indices = [index for index, value in enumerate(test_labels) if value == train_subject_id]
        other_train_indices = [index for index, value in enumerate(train_labels) if value != train_subject_id]
        other_test_indices = [index for index, value in enumerate(test_labels) if value != train_subject_id]

        # Generate genuine pairs (same class)
        for train_idx in train_class_indices:
            for test_idx in test_class_indices:
                pairs.append((train_features[train_idx], test_features[test_idx]))
                pair_labels.append(1)  # Same class

        # Generate impostor pairs (different classes)
        for train_idx in train_class_indices:
            for other_train_idx in other_train_indices:
                pairs.append((train_features[train_idx], train_features[other_train_idx]))
                pair_labels.append(0)  # Different classes

            for other_test_idx in other_test_indices:
                pairs.append((train_features[train_idx], test_features[other_test_idx]))
                pair_labels.append(0)  # Different classes

    return np.array(pairs), np.array(pair_labels)


# Create pairs and labels
pairs, pair_labels = create_pairs_and_labels2(X_train_pca, labels)
test_pairs, test_pair_labels = create_verification_data2(X_train_pca, labels,X_test_pca, test_labels)

# test_pairs, test_pair_labels = create_pairs_and_labels2(X_test_pca, test_labels)

# Prepare data for binary classification
X_train = np.array([np.concatenate([pair[0], pair[1]]) for pair in pairs])
X_test = np.array([np.concatenate([pair[0], pair[1]]) for pair in test_pairs])


# Train a binary classifier
# svm = SVC(C=best_params['C'], gamma=best_params['gamma'], kernel=best_params['kernel'],probability=True)
svm = SVC(probability=True)
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

