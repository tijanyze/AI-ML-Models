# Import necessary libraries
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Sample text data
documents = ["I love machine learning", "Text classification is interesting", "Naive Bayes is a simple algorithm",
             "Python programming is fun", "Machine learning models are powerful"]

# Corresponding labels
labels = ["Machine Learning", "Machine Learning", "Machine Learning", "Python Programming", "Machine Learning"]

# Create a CountVectorizer to convert text to numerical features
vectorizer = CountVectorizer()

# Convert text data to numerical features
X = vectorizer.fit_transform(documents)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

# Create a Multinomial Naive Bayes classifier
clf = MultinomialNB()

# Train the classifier on the training data
clf.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = clf.predict(X_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
