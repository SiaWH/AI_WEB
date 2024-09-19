# train_models.py
from sklearn.cluster import KMeans
import joblib

# Load your dataset
import pandas as pd
data = pd.read_csv('diabetes.csv')

# Separate features and target
X = data.drop(columns='Outcome')
y = data['Outcome']

# Scale the features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, stratify=y, random_state=42)

# Train Logistic Regression model
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train, y_train)
y_pred_cv_lnr = lr.predict(X_test)

# Train SVM model
from sklearn import svm
svm = svm.SVC()
svm.fit(X_train, y_train)
y_pred_cv_svm = svm .predict(X_test)

# Train Decision Tree model
from sklearn.tree import DecisionTreeClassifier
dtm = DecisionTreeClassifier()
dtm.fit(X_train, y_train)
y_pred_cv_dtm = dtm.predict(X_test)

# Train Random Forest model
from sklearn.ensemble import RandomForestClassifier
rfm = RandomForestClassifier()
rfm.fit(X_train, y_train)
y_pred_cv_rfm = rfm.predict(X_test)

# Train K-Nearest Neighbors model
from sklearn.neighbors import KNeighborsClassifier
knnm = KNeighborsClassifier()
knnm.fit(X_train, y_train)
y_pred_cv_knnm = knnm.predict(X_test)

# Train Naive Bayes model
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(X_train, y_train)
y_pred_cv_nb = nb.predict(X_test)

# Calculate accuracy scores for each model
from sklearn.metrics import accuracy_score
lnra = accuracy_score(y_test, y_pred_cv_lnr)
svma = accuracy_score(y_test, y_pred_cv_svm)
dtma = accuracy_score(y_test, y_pred_cv_dtm)
rfma = accuracy_score(y_test, y_pred_cv_rfm)
knnma = accuracy_score(y_test, y_pred_cv_knnm)
nba = accuracy_score(y_test, y_pred_cv_nb)

model_accuracies = {
    'Logistic Regression': lnra,
    'SVM': svma,
    'Decision Tree': dtma,
    'Random Forest': rfma,
    'KNN': knnma,
    'Naive Bayes': nba
}

best_model_name = max(model_accuracies, key=model_accuracies.get)
best_accuracy = model_accuracies[best_model_name]

if best_model_name == 'Logistic Regression':
    best_model = lr    # Use Logistic Regression model
elif best_model_name == 'SVM':
    best_model = svm   # Use SVM model
elif best_model_name == 'Decision Tree':
    best_model = dtm   # Use Decision Tree model
elif best_model_name == 'Random Forest':
    best_model = rfm   # Use Random Forest model
elif best_model_name == 'KNN':
    best_model = knnm  # Use KNN model
elif best_model_name == 'Naive Bayes':
    best_model = nb    # Use Naive Bayes model

# Save Logistic Regression model
joblib.dump(best_model, 'models/diabetes_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')

# Train KMeans for unsupervised learning
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_scaled)

# Save the KMeans model
joblib.dump(kmeans, 'models/kmeans_model.pkl')

print("Models trained and saved successfully.")
