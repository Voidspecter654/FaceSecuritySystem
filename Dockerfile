FROM python:3.10-slim
WORKDIR /app
RUN apt-get update && apt-get install -y build-essential cmake g++ libopenblas-dev liblapack-dev libx11-dev libsm6 libxext6 libxrender-dev libglib2.0-0 libopencv-dev && rm -rf /var/lib/apt/lists/*
COPY . /app
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 5000
CMD ["python", "app.py"]
