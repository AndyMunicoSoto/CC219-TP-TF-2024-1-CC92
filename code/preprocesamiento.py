import pandas as pd
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Cargar los datos
train_data = pd.read_csv('TF-Data/Sarcasm/train.csv')
test_data = pd.read_csv('TF-Data/Sarcasm/test.csv')

# Función de preprocesamiento de texto mejorada
def preprocess_text(text):
    # Convertir a minúsculas
    text = text.lower()
    # Eliminar caracteres especiales, puntuaciones y números
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Tokenización
    tokens = word_tokenize(text)
    # Eliminar stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    # Lematización
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    # Reconstruir el texto preprocesado
    processed_text = ' '.join(tokens)
    return processed_text

# Aplicar preprocesamiento mejorado a los datos de entrenamiento y prueba
train_data['processed_text'] = train_data['text'].apply(preprocess_text)
test_data['processed_text'] = test_data['text'].apply(preprocess_text)

# Guardar los datos preprocesados en nuevos archivos CSV
#train_data.to_csv('train_preprocessed.csv', index=False)
#test_data.to_csv('test_preprocessed.csv', index=False)
