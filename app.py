import logging
import os
import shutil
import tempfile
import uuid

import numpy as np
import pandas as pd
import tensorflow as tf
from flask import Flask, jsonify, request, send_file
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO)

app = Flask(__name__)

@app.route('/')
def index():
    return 'Hand Sign Model Training API. POST to /train with csv file.'

@app.route('/train', methods=['POST'])
def train_model():
    csv_file = request.files.get('csv')
    if not csv_file:
        return jsonify({'error': 'No CSV file provided'}), 400

    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, 'data.csv')
        csv_file.save(csv_path)

        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            return jsonify({'error': f'CSV read failed: {e}'}), 400

        required_cols = {f'{side}_{axis}_{i}' for side in ('right', 'left')
                         for axis in ('x', 'y', 'z') for i in range(21)}
        missing = required_cols - set(df.columns)
        if missing:
            return jsonify({'error': f'Missing columns: {missing}'}), 400

        if 'label' not in df.columns:
            return jsonify({'error': '"label" column missing'}), 400

        # encode labels
        labels = df['label'].astype('category').cat.codes.values
        unique_labels = df['label'].astype('category').cat.categories.tolist()

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

        n_classes = len(unique_labels)
        if n_classes < 2:
            return jsonify({'error': 'Need at least 2 classes'}), 400

        X_train, X_test, y_train, y_test = train_test_split(
            X, labels, test_size=0.2, stratify=labels, random_state=42)

        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(126,)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(n_classes, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        model.fit(X_train, y_train, validation_data=(X_test, y_test),
                  epochs=30, batch_size=32, verbose=1)

        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()

        out_path = os.path.join(tmpdir, 'hand_sign_model.tflite')
        with open(out_path, 'wb') as f:
            f.write(tflite_model)

        logging.info(f"Model size: {len(tflite_model)} bytes")
        return send_file(out_path, as_attachment=True, download_name='hand_sign_model.tflite')

if __name__ == '__main__':
    import os
    app.run(debug=os.getenv('DEBUG', 'False') == 'True', host='0.0.0.0', port=5000)
