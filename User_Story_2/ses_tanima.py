import os
import numpy as np
import librosa
import librosa.display
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report

# Path to the folder containing sound samples
DATASET_FOLDER = 'samples'

# Categories (subfolders) in your dataset
CATEGORIES = ['drums', 'piano', 'speech']

# Feature extraction function
def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=None)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        return np.mean(mfccs.T, axis=0)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# Prepare dataset
features, labels = [], []

for category in CATEGORIES:
    category_path = os.path.join(DATASET_FOLDER, category)
    for file_name in os.listdir(category_path):
        if file_name.endswith('.wav'):
            file_path = os.path.join(category_path, file_name)
            feature = extract_features(file_path)
            if feature is not None:
                features.append(feature)
                labels.append(category)

# Convert to numpy arrays
X = np.array(features)
y = np.array(labels)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"Accuracy: {accuracy:.2f}")
print(f"F1 Score: {f1:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
