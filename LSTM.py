import pandas as pd
import numpy as np
import re
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from scikeras.wrappers import KerasClassifier
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Ensure necessary NLTK resources are downloaded
nltk.download('stopwords')
nltk.download('punkt')

# Data preparation
df = pd.read_csv('/home/abin/IML lab/final.csv')

# Split the 'text;sentiment_label' column into 'text' and 'sentiment_label'
df[['text', 'sentiment_label']] = df['text;sentiment_label'].str.split(';', expand=True)
df['text'] = df['text'].str.strip()
df['sentiment_label'] = df['sentiment_label'].str.strip()

# Check for missing values
print(df.isnull().sum())

# Preprocessing: Remove stopwords
stop_words = set(stopwords.words('english'))

def remove_stopwords(text):
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

df['cleaned_text'] = df['text'].apply(lambda x: remove_stopwords(x))

# Label Encoding
label_encoder = LabelEncoder()
df['sentiment_label_encoded'] = label_encoder.fit_transform(df['sentiment_label'])

# Check the distribution of sentiment labels
print(df['sentiment_label_encoded'].value_counts())

# Remove classes with fewer than 2 samples
class_counts = df['sentiment_label_encoded'].value_counts()
classes_to_remove = class_counts[class_counts < 2].index
df = df[~df['sentiment_label_encoded'].isin(classes_to_remove)]

# Tokenization and Padding
MAX_NUM_WORDS = 10000
MAX_SEQUENCE_LENGTH = 100

tokenizer = Tokenizer(num_words=MAX_NUM_WORDS, oov_token="<OOV>")
tokenizer.fit_on_texts(df['cleaned_text'])

X = tokenizer.texts_to_sequences(df['cleaned_text'])
X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')

y = df['sentiment_label_encoded'].values

# Train-test split without stratification (if needed)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Function to create the model
def create_model(embedding_dim=128, lstm_units=64, dropout_rate=0.2, optimizer='adam'):
    model = Sequential([
        Embedding(input_dim=MAX_NUM_WORDS, output_dim=embedding_dim, input_length=MAX_SEQUENCE_LENGTH),
        Bidirectional(LSTM(lstm_units, return_sequences=True)),
        Dropout(dropout_rate),
        Bidirectional(LSTM(lstm_units // 2)),
        Dropout(dropout_rate),
        Dense(64, activation='relu'),
        Dropout(dropout_rate),
        Dense(3, activation='softmax')  # 3 output classes: negative, neutral, positive
    ])
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',  # Use sparse categorical crossentropy for integer labels
                  metrics=['accuracy'])
    return model

# Wrap Keras model for use with GridSearchCV
# KerasClassifier should pass its arguments as keyword arguments to the create_model function
model = KerasClassifier(build_fn=create_model, verbose=0)

# Hyperparameter grid for GridSearchCV
param_grid = {
    'embedding_dim': [128, 256],
    'lstm_units': [64, 128],
    'dropout_rate': [0.2, 0.3],
    'optimizer': ['adam', 'rmsprop'],
    'batch_size': [32, 64],
    'epochs': [10, 15]
}

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1, cv=3, verbose=2)

# Early stopping callback to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Perform GridSearchCV
grid_search.fit(X_train, y_train, callbacks=[early_stopping])

# Best hyperparameters from GridSearchCV
print(f"Best Hyperparameters: {grid_search.best_params_}")

# Evaluate the best model
best_model = grid_search.best_estimator_

# Evaluate on test data
loss, accuracy = best_model.score(X_test, y_test)
print(f"Test Accuracy: {accuracy*100:.2f}%")

# Predictions and classification report
y_pred = best_model.predict(X_test)
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))


