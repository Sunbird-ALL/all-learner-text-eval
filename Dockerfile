# Use the official Python image as the base image
FROM python:3.10-slim-bookworm

# Install dependencies for building Python packages and FFmpeg
RUN apt-get update && \
    apt-get install -y \
    build-essential \
    gcc \
    libssl-dev \
    libffi-dev \
    python3-dev \
    autoconf \
    automake \
    cmake \
    git-core \
    libass-dev \
    libfreetype6-dev \
    libsdl2-dev \
    libtool \
    libva-dev \
    libvdpau-dev \
    libvorbis-dev \
    libxcb1-dev \
    libxcb-shm0-dev \
    libxcb-xfixes0-dev \
    pkg-config \
    texinfo \
    wget \
    zlib1g-dev \
    nasm \
    yasm && \
    rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Install FFmpeg
# Install dependencies and the latest FFmpeg
RUN apt-get update && \
    apt-get install -y wget xz-utils && \
    wget -O /tmp/ffmpeg-release-amd64-static.tar.xz https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz && \
    tar -xf /tmp/ffmpeg-release-amd64-static.tar.xz -C /tmp && \
    mv /tmp/ffmpeg-*-static/ffmpeg /usr/local/bin/ && \
    mv /tmp/ffmpeg-*-static/ffprobe /usr/local/bin/ && \
    rm -rf /tmp/ffmpeg-*-static /var/lib/apt/lists/*

# Copy the requirements file and install dependencies
COPY requirements.txt .

# Install app dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all the files to the container
COPY . .

# One worker process per CPU (see app.py); keep numpy/BLAS in each worker to
# one thread instead of one per node core.
ENV OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

# Expose the port that the app runs on
EXPOSE $PORT

# Command to run the Flask application
CMD ["python", "app.py", "--host", "0.0.0.0"]
