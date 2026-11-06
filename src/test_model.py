import tensorflow as tf

MODEL_PATH = "models/ASL_final_model.keras"

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully!")
print()
print("Input shape:", model.input_shape)
print("Output shape:", model.output_shape)