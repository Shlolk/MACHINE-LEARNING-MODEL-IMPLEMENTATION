# 📌 1. Import Required Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
# 📌 2. Create or Load Dataset
data = {
    'text': [
        "Congratulations, you have won a lottery!",
        "Hi, how are you doing today?",
        "Click this link to claim your reward",
        "Reminder: your appointment is at 3 PM",
        "You've been selected for a $1000 Amazon gift card",
        "Let's catch up this weekend",
        "Get free access to Netflix for 1 year!",
        "Don't forget the meeting tomorrow",
        "Claim your FREE iPhone now!",
        "See you at the party tonight"
    ],
    'label': ['spam', 'ham', 'spam', 'ham', 'spam', 'ham', 'spam', 'ham', 'spam', 'ham']
}

df = pd.DataFrame(data)
df.head()# 📌 3. Preprocess Data
X = df['text']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)# 📌 4. Train the Model
model = MultinomialNB()
model.fit(X_train_vec, y_train)# 📌 5. Evaluate the Model
y_pred = model.predict(X_test_vec)

accuracy = accuracy_score(y_test, y_pred)
print("✅ Accuracy: {:.2f}%".format(accuracy * 100))

print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred))

print("📊 Confusion Matrix:")
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=['ham', 'spam'], yticklabels=['ham', 'spam'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()# 📌 6. Test with Custom Input
def classify_message(msg):
    vec = vectorizer.transform([msg])
    prediction = model.predict(vec)[0]
    return "🚫 SPAM" if prediction == "spam" else "✅ HAM"

# Test
test_msg = "You've won a free vacation. Click here now!"
print("Message:", test_msg)
print("Prediction:", classify_message(test_msg))