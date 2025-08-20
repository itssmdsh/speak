from flask import Flask, request, jsonify, send_file
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

app = Flask(__name__)

@app.route('/train', methods=['POST'])
def train_model():
    # Upload CSV file
    file = request.files['csv']
    if not file:
        return jsonify({"error": "No file provided"}), 400

    # Save file to a temporary location
    csv_path = 'temp/data.csv'
    file.save(csv_path)

    # Load and preprocess data
    df = pd.read_csv(csv_path)
    labels = df['label'].astype('category').cat.codes.values
    NUM_CLASSES = len(df['label'].unique())

    def wrist_scale(row, side):
        xs = [row[f'{side}_x_{i}'] for i in range(21)]
        ys = [row[f'{side}_y_{i}'] for i in range(21)]
        zs = [row[f'{side}_z_{i}'] for i in range(21)]
        wrist = np.array([xs[0], ys[0], zs[0]])
        mcp = np.array([xs[9], ys[9], zs[9]])
        scale = np.linalg.norm(mcp - wrist) + 1e-6
        pts = np.vstack([xs, ys, zs]).T
        pts = (pts - wrist) / scale
        return pts.flatten()

    X = []
    for _, r in df.iterrows():
        right_vec = wrist_scale(r, 'right')
        left_vec = wrist_scale(r, 'left')
        X.append(np.concatenate([left_vec, right_vec]))
    X = np.array(X, dtype=np.float32)
    y = labels

    # Train the model
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(126,)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=30, batch_size=32, verbose=0)

    # Convert to TensorFlow Lite model
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    # Save the model to a temporary file
    tflite_model_path = 'temp/hand_sign_model.tflite'
    with open(tflite_model_path, 'wb') as f:
        f.write(tflite_model)

    # Return the model as a file
    return send_file(tflite_model_path, as_attachment=True, attachment_filename='hand_sign_model.tflite')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)