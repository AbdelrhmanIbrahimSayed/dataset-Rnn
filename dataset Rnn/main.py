import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import socket
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.neighbors import NearestNeighbors
from keras.src.legacy.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)
CORS(app)

PORT = 8001

file_path = 's.xlsx'
df = pd.read_excel(file_path)
selected_columns = ['title', 'sub genres', 'New genre', 'description', 'rating']
df = df[selected_columns]

df1 = pd.read_excel(file_path)
selected_columns1 = ['title', 'sub genres', 'New genre', 'description', 'coverImg', 'pdfurl',
                     'language', 'pages',
                     'rating']
df1 = df1[selected_columns1]

df2 = pd.read_excel(file_path)
selected_columns2 = ['title', 'rating']
df2 = df2[selected_columns2]

df.dropna(inplace=True)
df['text'] = df['New genre'].astype(str) + ' ' + df['sub genres'].astype(str)

# Encode categorical labels
label_encoder = LabelEncoder()
df['encoded_genre'] = label_encoder.fit_transform(df['New genre'])

# Tokenize the text
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df['text'])
sequences = tokenizer.texts_to_sequences(df['text'])
X = pad_sequences(sequences, maxlen=100)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, df['encoded_genre'], test_size=0.2,
                                                    random_state=42)

# Build the RNN model
vocab_size = len(tokenizer.word_index) + 1
embedding_dim = 100

model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim))
model.add(LSTM(100))
model.add(Dense(len(label_encoder.classes_), activation='softmax'))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=5, batch_size=64, validation_split=0.2)

# Feature extraction for Nearest Neighbors model
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(df['title'])

# Model training for Nearest Neighbors
model_nn = NearestNeighbors(n_neighbors=5, algorithm='brute', metric='cosine')
model_nn.fit(tfidf_matrix)


# Function to recommend titles based on genre and sub-genre
@app.route('/', methods=['GET'])
def recommend_titles():
    user_input = request.args.get('genre')
    user2_input = request.args.get('sub_genre')
    if user_input is None:
        return jsonify({"error": "No title provided"}), 400
    if user2_input is None:
        user2_input = ''

    recommended_titles = model_recommended_titles(user_input, user2_input)

    recommendations = []
    for title in recommended_titles:
        if title in df['title'].values:
            genre = df.loc[df['title'] == title, 'New genre'].iloc[0]
            description = df.loc[df['title'] == title, 'description'].iloc[0]
            rating = df.loc[df['title'] == title, 'rating'].iloc[0]
            recommendation = {
                "genre": genre,
                "title": title,
                "description": description,
                "rating": rating
            }
            recommendations.append(recommendation)

    return jsonify(recommendations)


# @app.route('/random_story', methods=['GET'])
# def get_random_story():
#     user_input = request.args.get('genre')
#     user2_input = request.args.get('sub_genre')
#     if user_input is None:
#         return jsonify({"error": "No genre provided"}), 400
#     if user2_input is None:
#         user2_input = ''
#
#     recommended_titles = model_recommended_titles(user_input, user2_input)
#
#     if not recommended_titles:
#         return jsonify({"error": "No recommendations found for the given genre and sub-genre"}), 404
#
#     random_story = random.choice(recommended_titles)
#
#     if random_story in df['title'].values:
#         genre = df.loc[df['title'] == random_story, 'New genre'].iloc[0]
#         description = df.loc[df['title'] == random_story, 'description'].iloc[0]
#         rating = df.loc[df['title'] == random_story, 'rating'].iloc[0]
#         random_story_info = {
#             "genre": genre,
#             "title": random_story,
#             "description": description,
#             "rating": rating
#         }
#         return jsonify(random_story_info)
#
#     return jsonify({"error": "Random story not found in the database"}), 404


@app.route('/search', methods=['GET'])
def recommend():
    user_input = request.args.get('genre')
    if user_input is None:
        return jsonify({"error": "No title provided"}), 400
    try:
        user_tfidf = tfidf_vectorizer.transform([user_input])
        distances, indices = model_nn.kneighbors(user_tfidf)
        recommendations = []
        for idx in indices[0]:
            recommendation = {
                "genre": df.iloc[idx]['New genre'],
                "title": df.iloc[idx]['title'],
                "description": df.iloc[idx]['description'],
                "rating": df.iloc[idx]['rating']
            }
            recommendations.append(recommendation)
        return jsonify(recommendations)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Function to predict and recommend titles using RNN model
def model_recommended_titles(user_input, user2_input):
    if not isinstance(user_input, str) or not isinstance(user2_input, str):
        return []  # Return empty list if inputs are not strings

    input_data = {
        'New genre': user_input,
        'sub genres': user2_input,
    
    input_df = pd.DataFrame([input_data])
    input_sequences = tokenizer.texts_to_sequences(
        input_df['New genre'] + ' ' + input_df['sub genres'])
    input_padded = pad_sequences(input_sequences, maxlen=100)

    predicted_probabilities = model.predict(input_padded)
    predicted_class = predicted_probabilities.argmax(axis=1)[0]

    recommended_titles = \
        df[(df['encoded_genre'] == predicted_class) | (
                df['sub genres'] == input_data['sub genres'])][
            'title'].tolist()

    return recommended_titles[:10]


# Add this method to the code
@app.route('/get_data', methods=['GET'])
def get_cover_img_pdf_url():
    title = request.args.get('title')
    if title is None:
        return jsonify({"error": "No title provided"}), 400
    data = df1[df1['title'] == title]
    if data.empty:
        return jsonify({"error": "Title not found"}), 404
    if 'coverImg' not in data.columns or 'pdfurl' not in data.columns:
        return jsonify({"error": "Cover image or PDF URL not available"}), 404
    cover_img = data.iloc[0]['coverImg']
    pdf_url = data.iloc[0]['pdfurl']
    language = data.iloc[0]['language']
    pages = data.iloc[0]['pages']
    description = data.iloc[0]['description']
    rating = data.iloc[0]['rating']
    return jsonify({
        "title": title,
        "coverImg": cover_img,
        "pdfurl": pdf_url,
        "language": language,
        "pages": pages,
        "description": description,
        "rating": rating
    })


@app.route('/process_book_data', methods=['POST'])
def process_book_data():
    # Fetch parameters from JSON body
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400
    try:
        # Path to your Excel file
        excel_file_path = 's.xlsx'
        # Try to read existing data
        try:
            existing_data = pd.read_excel(excel_file_path)
        except FileNotFoundError:
            # Define DataFrame columns if the file does not exist
            columns = ['bookId', 'title', 'author', 'rating', 'description', 'language',
                       'subGenres', 'characters', 'pages', 'publisher', 'awards', 'recommendation']
            existing_data = pd.DataFrame(columns=columns)
        # Append new data
        new_data = pd.DataFrame([data])
        updated_data = pd.concat([existing_data, new_data], ignore_index=True)
        # Write updated data to Excel file
        with pd.ExcelWriter(excel_file_path, engine='openpyxl', mode='w') as writer:
            updated_data.to_excel(writer, index=False)
        return jsonify({"message": "Book data processed and saved"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Run the Flask app
if __name__ == '__main__':
    host_ip = socket.gethostbyname(socket.gethostname())
    app.run(debug=True, host=host_ip, port=8001)
