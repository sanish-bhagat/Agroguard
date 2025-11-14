# ✅ Base image with TensorFlow 2.17.0 (includes tf.keras)
FROM tensorflow/tensorflow:2.17.0

# ✅ Set working directory
WORKDIR /app

# ✅ Copy requirements first (for caching)
COPY requirements.txt .

# ✅ Upgrade pip and fix Keras issue
RUN pip install --upgrade pip setuptools wheel && \
    pip install --ignore-installed blinker && \
    pip install tf-keras==2.17.0 && \
    pip install --no-cache-dir -r requirements.txt

# ✅ Copy project files
COPY . .

# ✅ Expose port
EXPOSE $PORT

# ✅ Run Flask app
CMD ["python", "app.py"]