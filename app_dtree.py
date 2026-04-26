# Loading library python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

# Read dataset
df_net = pd.read_csv('citrus.csv')
df_net.head()

# Describe data
df_net.describe()

# data count
df_net['name'].value_counts()

# Diameter distribution
sns.displot(df_net['diameter'])
plt.title('Diameter Distribution')
plt.xlabel('Diameter')
plt.ylabel('Frequency')
plt.show()

# Weight distribution
sns.displot(df_net['weight'])
plt.title('Weight Distribution')
plt.xlabel('Weight')
plt.ylabel('Frequency')
plt.show()

# Label encoding, grapefruit = 0, orange = 1
le = LabelEncoder()
df_net['name']= le.fit_transform(df_net['name'])
print('Label Encoding results', dict(zip(['grapefruit', 'orange'], le.transform(['grapefruit', 'orange']))))

# Correlation matrix
corr_matrix = df_net.corr()

# Heatmap correlation matrix
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.show()

# Relationship between Diameter and Weight
plt.scatter(df_net['diameter'], df_net['weight'], c=df_net['name'], cmap='bwr', alpha=0.4)
plt.xlabel('Diameter')
plt.ylabel('Weight')
plt.title('Relationship between Diameter and Weight')
plt.show()

# Split data into independent/dependent variables
X  = df_net[['diameter', 'weight', 'red', 'green', 'blue']].values
y = df_net['name'].values

# Split data into Train/Test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

print('Length of X_train:', len(X_train))
print('Length of X_test:', len(X_test))

# Scale dataset
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Train Decision Tree model
classifier = DecisionTreeClassifier(criterion='entropy', random_state=0)
classifier.fit(X_train, y_train)

# Prediction
y_pred = classifier.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))

# Accuracy
print(f"Accuracy : {accuracy_score(y_test, y_pred)}")

# F1 score
print(f"F1 Score : {f1_score(y_test, y_pred)}")

# Classification report
print(f'Classification Report: \n{classification_report(y_test, y_pred)}')

# Confusion matrix
cf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(cf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.show()

# Plot Precision-Recall Curve
y_pred_proba = classifier.predict_proba(X_test)[:,1]
precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)

fig, ax = plt.subplots()
ax.plot(recall, precision, label='Decision Tree')
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.set_title('Precision-Recall Curve')
ax.legend()
plt.tight_layout()
plt.show()

# Plot AUC/ROC curve
y_pred_proba = classifier.predict_proba(X_test)[:,1]
fpr, tpr, thresholds = metrics.roc_curve(y_test,  y_pred_proba)

fig, ax = plt.subplots()
ax.plot(fpr, tpr, label='Decision Tree Classification', color = 'firebrick')
ax.set_title('ROC Curve')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
plt.box(False)
ax.legend()
plt.tight_layout()
plt.show()

