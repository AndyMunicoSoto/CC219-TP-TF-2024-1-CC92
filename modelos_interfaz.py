import tkinter as tk
from tkinter import messagebox
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.ensemble import VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
import pandas as pd
import tensorflow as tf

# Cargar datos y preprocesar si es necesario
train_data = pd.read_csv('train_preprocessed.csv')
test_data = pd.read_csv('test_preprocessed.csv')

# Eliminar filas con valores NaN en train_data
train_data = train_data.dropna(subset=['processed_text'])

# Preparar datos para SVM y KNN
vectorizer = TfidfVectorizer()
X_train_svm = vectorizer.fit_transform(train_data['processed_text'])
y_train_svm = train_data['Y']

# Eliminar filas con valores NaN en test_data
test_data = test_data.dropna(subset=['processed_text'])

# Preparar datos para SVM y KNN en el conjunto de prueba
X_test_svm = vectorizer.transform(test_data['processed_text'])
y_test_svm = test_data['Y']

# Preparar datos para CNN
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(train_data['processed_text'])
X_train_cnn = tokenizer.texts_to_sequences(train_data['processed_text'])
X_train_cnn = pad_sequences(X_train_cnn, maxlen=100)
y_train_cnn = train_data['Y']

# Definir modelo CNN como extractor de características
cnn_model = Sequential()
cnn_model.add(Embedding(input_dim=5000, output_dim=100, input_length=100))
cnn_model.add(Conv1D(filters=128, kernel_size=5, activation='relu'))
cnn_model.add(GlobalMaxPooling1D())
cnn_model.add(Dense(256, activation='relu'))
cnn_model.add(Dropout(0.5))
cnn_model.add(Dense(1, activation='sigmoid'))  # Capa de salida para clasificación binaria

# Compilar modelo CNN
cnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Entrenar modelo CNN
cnn_model.fit(X_train_cnn, y_train_cnn, epochs=5, batch_size=128, validation_split=0.2)

# Extraer características con el modelo CNN para ambos conjuntos de datos
X_train_cnn_output = cnn_model.predict(X_train_cnn)
X_test_cnn = tokenizer.texts_to_sequences(test_data['processed_text'])
X_test_cnn = pad_sequences(X_test_cnn, maxlen=100)
X_test_cnn_output = cnn_model.predict(X_test_cnn)

# Entrenar modelos SVM y KNN
svm_model = SVC(kernel='rbf', C=10, gamma='scale', probability=True)
svm_model.fit(X_train_svm, y_train_svm)
knn_model = KNeighborsClassifier(n_neighbors=5, weights='distance')
knn_model.fit(X_train_svm, y_train_svm)

# Crear ensamble por votación
voting_clf = VotingClassifier(estimators=[('svm', svm_model), ('knn', knn_model)], voting='soft', weights=[2, 1])
voting_clf.fit(X_train_cnn_output, y_train_svm)

# Evaluar el ensamble por votación
ensemble_preds = voting_clf.predict(X_test_cnn_output)
ensemble_accuracy = accuracy_score(y_test_svm, ensemble_preds)
print("Ensemble Accuracy:", ensemble_accuracy)

# Función para predecir sarcasmo
def predict_sarcasm(text):
    # Preprocesar texto
    text_seq = tokenizer.texts_to_sequences([text])
    text_pad = pad_sequences(text_seq, maxlen=100)
    cnn_features = cnn_model.predict(text_pad)
    prediction = voting_clf.predict(cnn_features)
    return prediction[0]

# Crear interfaz de usuario con Tkinter
def classify_text():
    input_text = text_entry.get("1.0",'end-1c')
    if input_text.strip():
        prediction = predict_sarcasm(input_text)
        result = "Sarcasmo" if prediction == 1 else "No es sarcasmo"
        messagebox.showinfo("Resultado", result)
    else:
        messagebox.showwarning("Advertencia", "Por favor, ingrese un texto.")

root = tk.Tk()
root.title("Detector de Sarcasmo")
root.geometry("600x400")  # Tamaño de la ventana

frame = tk.Frame(root, padx=20, pady=20)
frame.pack(expand=True)

tk.Label(frame, text="Ingrese un texto:", font=("Arial", 14)).pack(pady=10)

text_entry = tk.Text(frame, height=10, width=50, font=("Arial", 12))
text_entry.pack(pady=10)

classify_button = tk.Button(frame, text="Clasificar", command=classify_text, font=("Arial", 14), bg="blue", fg="white")
classify_button.pack(pady=20)

root.mainloop()