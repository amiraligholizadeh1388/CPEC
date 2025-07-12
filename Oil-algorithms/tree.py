import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
import numpy as np
# Load data
data = pd.read_csv("data\\Merged file1.csv")
# Prepare data
specified_country = data[data['Year'] == int(input('Enter your target year: '))]

X = specified_country['Oil consumption - TWh']
y = specified_country['Cluster']  # Assuming you have a column indicating the true clusters
X = np.array([int(i) for i in X]).reshape(-1 , 1)
y = np.array([int(i) for i in y]).reshape(-1 , 1)
# Train decision tree
clf = DecisionTreeClassifier(max_depth=3).fit(X, y)

# Plot decision tree
plt.figure(figsize=(10, 6))
plot_tree(clf, filled=True, feature_names=y, class_names=[str(i) for i in clf.classes_])
print(clf.predict(X))
plt.show()
