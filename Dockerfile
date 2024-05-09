FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

RUN apt-get update -y && \
    apt-get install -y git build-essential

#only do this for captum stuff
#RUN DEBIAN_FRONTEND=noninteractive apt-get install -y wkhtmltopdf

COPY requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r /tmp/requirements.txt
