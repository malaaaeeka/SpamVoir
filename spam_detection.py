# installig libraries
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC 
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load data (example with CSV)
data = pd.read_csv('spam.csv', encoding='latin-1')
data = data[['v1', 'v2']]  # Typically label and text columns
data.columns = ['label', 'text']

# Convert labels to binary (0=ham/notspam, 1=spam)
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

# Split into features and labels
X = data['text']
y = data['label']

# Convert text to numerical features using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_features=100000)
X = vectorizer.fit_transform(X)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 

#train svm model
svm_model = SVC(kernel='linear')  # Linear SVM works best for text
svm_model.fit(X_train, y_train)  # Train the model

#accuracy 
y_pred = svm_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2%}")

#Classification Report (Precision, Recall, F1-Score)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Not Spam', 'Spam']))

#Confusion Matrix (Visualization)
plt.figure(figsize=(6, 4))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Not Spam', 'Spam'],
            yticklabels=['Not Spam', 'Spam'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

 #Test with New Emails
def predict_spam(email_text):
    email_vec = vectorizer.transform([email_text])
    prediction = svm_model.predict(email_vec)
    return "Spam" if prediction[0] == 1 else "Not Spam"

# Test examples
print("\nTesting Sample Emails:")
print("1. 'WINNER! Claim your prize now!' →", predict_spam("WINNER! Claim your prize now!"))
print("2. 'Meeting at 3 PM tomorrow' →", predict_spam("Meeting at 3 PM tomorrow"))
print("3. 'Your account has been locked' →", predict_spam("Your account has been locked"))
print("4. 'Free vacation for our best customer' →", predict_spam("Free vacation for our best customer"))
print("5. 'Project update: Deadline extended' →", predict_spam("Project update: Deadline extended"))
print("6. 'Your package will arrive tomorrow' →", predict_spam("Your package will arrive tomorrow"))
print("7. 'Click here to claim your $1000 reward' →", predict_spam("Click here to claim your $1000 reward"))
print("8. 'Congratulations!! you won a car' →", predict_spam("Congratulations!! you won a car"))
print("9. 'Congratulations! You won a $1000 Walmart gift card!' →", predict_spam("Congratulations! You won a $1000 Walmart gift card!"))
print("10. 'Your Amazon account has been suspended - verify now' →", predict_spam("Your Amazon account has been suspended - verify now"))
print("11. 'You've been selected for a free cruise vacation!' →", predict_spam("You've been selected for a free cruise vacation!"))
print("12. 'URGENT: Your NayaPay password expired' →", predict_spam("URGENT: Your NayaPay password expired"))
print("13. 'Your Netflix subscription has been canceled' →", predict_spam("Your Netflix subscription has been canceled"))
print("14. 'Your social security number needs verification' →", predict_spam("Your social security number needs verification"))
print("15. 'Your bank account is overdrawn - click to review' →", predict_spam("Your bank account is overdrawn - click to review"))
print("16. 'You qualify for a government grant!' →", predict_spam("You qualify for a government grant!"))
print("27. 'Your Microsoft Windows license is expired' →", predict_spam("Your Microsoft Windows license is expired"))
print("18. 'Your LinkedIn connection request' →", predict_spam("Your LinkedIn connection request"))
print("19. 'Your electricity bill is due next week' →", predict_spam("Your electricity bill is due next week"))
print("20. 'Zoom meeting link for tomorrow's presentation' →", predict_spam("Zoom meeting link for tomorrow's presentation"))
print("21. 'Your monthly bank statement is ready' →", predict_spam("Your monthly bank statement is ready"))
print("22. 'Your recent order has been shipped' →", predict_spam("Your recent order has been shipped"))



