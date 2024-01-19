# Dockerfile

# Use the official Python image as the base image
FROM python:3.10-slim-buster

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .

# Install app dependencies
RUN pip install -r requirements.txt

# Copy all the files to the container
COPY . .

# Expose the port that the app runs on
EXPOSE 5000

# Command to run the Flask application
CMD ["python", "app.py","--host","0.0.0.0"]
