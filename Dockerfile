FROM python:3.11.10

# Update pip
RUN pip install --upgrade pip

# Setup directory structure
RUN mkdir -p /home/app
WORKDIR /home/app
RUN mkdir -p data/raw
RUN mkdir -p data/processed
RUN mkdir src

# Copy files to directory structure
COPY ./src src
COPY ./requirements.txt .
COPY ./.env .
COPY ./app.py .
COPY ./data/raw ./data/raw

# Create python environment
RUN pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu
RUN python -m pip install -r requirements.txt

# Install models
RUN python -m spacy download en_core_web_md

# Make port 5000 available for the app
EXPOSE 5000

# Run the Flask app
CMD ["flask", "run", "--host=0.0.0.0"]