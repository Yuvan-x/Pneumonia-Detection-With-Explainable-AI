import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D
from PIL import Image
import cv2

# ------------------ Config ------------------ #
st.set_page_config(page_title="Pneumonia Detection", layout="centered")
IMG_SIZE = 150

# ------------------ Load Model ------------------ #
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("pneumonia_model.h5")
    return model

model = load_model()

# ------------------ Prediction ------------------ #
def predict_image(img):
    img = img.convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)

    prob = model(img_tensor, training=False)[0][0]

    if prob < 0.5:
        return "Normal", 1 - prob, img_tensor
    else:
        return "Pneumonia", prob, img_tensor


# ------------------ Grad-CAM (WORKING VERSION) ------------------ #
def get_gradcam_heatmap(model, img_tensor):

    # Find last Conv layer
    last_conv_layer = None
    for layer in reversed(model.layers):
        if isinstance(layer, Conv2D):
            last_conv_layer = layer
            break

    if last_conv_layer is None:
        st.error("❌ No Conv2D layer found")
        return None

    # Create sub-model (SAFE METHOD)
    conv_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=last_conv_layer.output
    )

    classifier_input = tf.keras.Input(shape=last_conv_layer.output.shape[1:])
    x = classifier_input

    # Rebuild remaining layers manually
    for layer in model.layers[model.layers.index(last_conv_layer) + 1:]:
        x = layer(x)

    classifier_model = tf.keras.Model(classifier_input, x)

    # Gradient computation
    with tf.GradientTape() as tape:
        conv_output = conv_model(img_tensor)
        tape.watch(conv_output)

        preds = classifier_model(conv_output)
        loss = preds[:, 0]

    grads = tape.gradient(loss, conv_output)

    if grads is None:
        st.error("❌ Gradients not computed")
        return None

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_output = conv_output[0]
    heatmap = conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0)
    max_val = tf.reduce_max(heatmap)

    if max_val == 0:
        return None

    heatmap /= max_val

    return heatmap.numpy()


# ------------------ Overlay ------------------ #
def overlay_heatmap(original_img, heatmap):
    original_img = original_img.convert("RGB")
    original_img = np.array(original_img)

    heatmap = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
    heatmap = np.uint8(255 * heatmap)

    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    result = cv2.addWeighted(original_img, 0.6, heatmap, 0.4, 0)

    return result


# ------------------ UI ------------------ #
st.title("🩺 Pneumonia Detection with Explainable AI")

uploaded_file = st.file_uploader("Upload X-ray", type=["jpg", "jpeg", "png"])

if uploaded_file:

    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Image", use_container_width=True)

    label, confidence, img_tensor = predict_image(image)

    if label == "Pneumonia":
        st.error(f"🦠 Prediction: {label}")
    else:
        st.success(f"✅ Prediction: {label}")

    st.write(f"Confidence: {round(float(confidence) * 100, 2)}%")

    # Severity
    if label == "Pneumonia":
        if confidence > 0.80:
            st.error("🔴 High Severity")
        elif confidence > 0.60:
            st.warning("🟠 Moderate Severity")
        else:
            st.info("🟡 Low Severity")
    else:
        st.success("🟢 Healthy")

    # Grad-CAM
    st.subheader("🔥 Explainable AI")

    heatmap = get_gradcam_heatmap(model, img_tensor)

    if heatmap is not None:
        result = overlay_heatmap(image, heatmap)
        st.image(result, caption="Highlighted Region", use_container_width=True)
    else:
        st.warning("⚠️ Heatmap not generated")