import os
import numpy as np
from function import *
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import TensorBoard

# Expected frame shape
EXPECTED_FRAME_SHAPE = (63,)  # Adjust according to your actual data shape

# Label mapping
label_map = {label: num for num, label in enumerate(actions)}

sequences, labels = [], []

# Load data
for action in actions:
    for sequence in range(no_sequences):
        window = []
        for frame_num in range(sequence_length):
            try:
                # Attempt to load frame
                res = np.load(
                    os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)),
                    allow_pickle=True
                )

                # Ensure the frame is not empty
                if res.size == 0:
                    print(f"⚠️ Empty frame encountered for action: {action}, sequence: {sequence}, frame: {frame_num}. Skipping this frame.")
                    continue  # Skip empty frames

                # Ensure res is a NumPy array and has a consistent shape
                res = np.array(res, dtype=np.float32)

                # Fix shape if necessary
                if res.shape != EXPECTED_FRAME_SHAPE:
                    print(f"⚠️ Frame shape mismatch! Expected {EXPECTED_FRAME_SHAPE}, got {res.shape}. Fixing it.")
                    
                    # Flatten the array if dimensions are greater than 1
                    if res.ndim > 1:
                        res = res.flatten()

                    # Pad or truncate to expected shape
                    if res.shape[0] < EXPECTED_FRAME_SHAPE[0]:
                        res = np.pad(res, (0, EXPECTED_FRAME_SHAPE[0] - res.shape[0]), mode='constant')
                    elif res.shape[0] > EXPECTED_FRAME_SHAPE[0]:
                        res = res[:EXPECTED_FRAME_SHAPE[0]]

                # Ensure consistent frame shape
                if res.shape == EXPECTED_FRAME_SHAPE:
                    window.append(res)
                else:
                    print(f"⚠️ Skipping frame with unexpected shape: {res.shape}.")
                    continue  # Skip frames with inconsistent shape

            except Exception as e:
                print(f"⚠️ Error loading frame: {e}. Skipping frame.")
                continue  # Skip problematic frames

        # Only add to sequences if window has frames of consistent shape
        if len(window) == sequence_length:
            sequences.append(window)
            labels.append(label_map[action])
        else:
            print(f"⚠️ Inconsistent sequence length for action: {action}, sequence: {sequence}. Skipping this sequence.")

# Convert sequences to NumPy array
try:
    X = np.array(sequences, dtype=np.float32)  # No shape errors now!
except Exception as e:
    print(f"⚠️ Error in converting sequences to NumPy array: {e}.")
    exit()

# Convert labels to categorical format
y = to_categorical(labels).astype(int)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

# TensorBoard setup
log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)

# Define model
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(X.shape[1], X.shape[2])))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(len(actions), activation='softmax'))

# Compile model
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

# Train model
model.fit(X_train, y_train, epochs=200, callbacks=[tb_callback])

# Print model summary
model.summary()

# Save model
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
model.save('model.h5')
