FROM continuumio/miniconda3

WORKDIR /app

# Copy environment config
COPY environment.yml .


ENV HF_HOME=/root/.cache/huggingface

ENV PIP_DEFAULT_TIMEOUT=1000 \
    PIP_RETRIES=15 \
    PIP_NO_CACHE_DIR=no


RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    ffmpeg \
	curl \
    ca-certificates \
    libsm6 \
    libxext6 \
	libgl1 \
    libglib2.0-0 \
    libgl1-mesa-dev \
    && apt-get clean



# Create the Conda environment
RUN conda env create -f environment.yml

SHELL ["conda", "run", "-n", "Faithfulness", "/bin/bash", "-c"]


RUN python -m spacy download en_core_web_sm


COPY . ./

## if you want to use this dockerfile wihtout docker compose uncomment the command below
#RUN curl -L https://github.com/ollama/ollama/releases/download/v0.9.6/ollama-linux-amd64.tgz -o ollama-linux-amd64.tgz && \
#    tar -C /usr -xzf ollama-linux-amd64.tgz && \
#    chmod +x /usr/bin/ollama && \
#    /usr/bin/ollama serve &



# Expose ports
EXPOSE 8000
EXPOSE 7860
ENV GRADIO_SERVER_NAME="0.0.0.0"


CMD ["conda", "run", "--no-capture-output", "-n", "Faithfulness", "bash", "start.sh"]