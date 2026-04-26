from sklearn import svm
import numpy as np

# Fake feature data (simulate LGS output)
# 1 = real, 0 = spoof
X = np.array([
    [0.9, 0.8, 0.85],
    [0.2, 0.3, 0.25],
    [0.88, 0.75, 0.9],
    [0.1, 0.2, 0.15]
])

y = np.array([1, 0, 1, 0])

# Train SVM
model = svm.SVC()
model.fit(X, y)

# Predict new sample
test = np.array([[0.85, 0.8, 0.88]])
prediction = model.predict(test)

print("Prediction:", "Real" if prediction[0] == 1 else "Spoof")