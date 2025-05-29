# NVIDIA L4T ML base image (includes PyTorch, TF, TensorRT, OpenCV etc.)
# Adjust the tag to your JetPack version!
# For JetPack 4.6.x (L4T r32.7.x):
FROM nvcr.io/nvidia/l4t-ml:r32.7.1-py3
# For JetPack 5.x (L4T r35.x.x), e.g.:
# FROM nvcr.io/nvidia/l4t-ml:r35.2.1-py3

WORKDIR /app

# System dependencies for OpenCV display, GStreamer etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev \
    libgstreamer1.0-0 gstreamer1.0-plugins-base gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-libav \
    libgstreamer-plugins-base1.0-dev \
    && rm -rf /var/lib/apt/lists/*

# Python-Abhängigkeiten kopieren und installieren
COPY requirements_jetson.txt .
RUN pip3 install --no-cache-dir -r requirements_jetson.txt

# Inferenzskript und ggf. Utility-Skripte kopieren
COPY inference_jetson.py .

# --- Modell-Handling ---
# The TensorRT engine should be available here.
# Ensure 'face_detector.engine' (or your chosen name)
# is in the same directory as this Dockerfile when you build.
COPY face_detector.engine .

# Umgebungsvariable für GStreamer (kann nützlich sein)
ENV GST_PLUGIN_PATH=/usr/lib/aarch64-linux-gnu/gstreamer-1.0

# Standardbefehl zum Starten der Inferenz
CMD ["python3", "inference_jetson.py"]