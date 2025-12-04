FROM agrigorev/model-2025-hairstyle:v1

# Install dependencies
RUN pip install numpy onnxruntime pillow

# Copy script python kita
COPY lambda_function.py .

# Set CMD agar otomatis menjalankan handler kita
CMD [ "lambda_function.lambda_handler" ]