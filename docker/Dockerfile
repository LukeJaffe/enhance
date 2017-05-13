FROM ubuntu:16.04
MAINTAINER Luke Jaffe <jaffe5@llnl.gov>

# Install ubuntu packages
RUN apt update && apt install -y \
    git=1:2.7.4-0ubuntu1 \
    python3=3.5.1-3 \
    python3-dev=3.5.1-3 \
    python3-pip=8.1.1-2ubuntu0.4 \
    python3-numpy=1:1.11.0-1ubuntu1 \
    python3-scipy=0.17.0-1 \
    python3-matplotlib=1.5.1-1ubuntu1 \
    build-essential=12.1ubuntu2 \
    cmake=3.5.1-1ubuntu3 \
    wget=1.17.1-1ubuntu1.2 \
    pkg-config=0.29.1-0ubuntu1 \
    man-db=2.7.5-1 \
    vim=2:7.4.1689-3ubuntu1.2 \
    eog=3.18.2-1ubuntu2.1 \
    libopenblas-dev=0.2.18-1ubuntu1 \
    liblapack-dev=3.6.0-2ubuntu2 \
    gfortran=4:5.3.1-1ubuntu1

# Upgrade pip
RUN pip3 install --upgrade pip==9.0.1

# Install python libs
RUN pip3 install \
    ipython==6.0.0 \
    jupyter==1.0.0 \
    keras==2.0.4

# Install cuda
WORKDIR /opt
RUN wget https://developer.nvidia.com/compute/cuda/8.0/Prod2/local_installers/cuda_8.0.61_375.26_linux-run --no-check-certificate
RUN sh cuda_8.0.61_375.26_linux-run --toolkit -silent
ENV CUDA_HOME /usr/local/cuda
ENV PATH ${CUDA_HOME}/bin:$PATH
ENV LD_LIBRARY_PATH $LD_LIBRARY_PATH:/usr/local/cuda-8.0/lib64

# Install cudnn
RUN wget http://olin:8000/files/cudnn-8.0-linux-x64-v5.1.tgz
RUN tar zxf cudnn-8.0-linux-x64-v5.1.tgz
WORKDIR /opt/cuda
RUN cp -R * /usr/local/cuda
ENV CUDNN_PATH /usr/local/cuda/lib64/libcudnn.so

# Install tensorflow
RUN wget --no-check-certificate https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.1.0-cp35-cp35m-linux_x86_64.whl
RUN pip3 install tensorflow_gpu-1.1.0-cp35-cp35m-linux_x86_64.whl

# Install pytorch
RUN pip3 install http://download.pytorch.org/whl/cu80/torch-0.1.12.post2-cp35-cp35m-linux_x86_64.whl
RUN pip3 install torchvision

# Install scikit image
RUN pip3 install scikit-image

# Need libpython for treestats build
RUN apt-get install -y python-dev=2.7.11-1
RUN apt-get install -y python-pip
RUN apt-get install -y python-numpy
RUn apt-get install -y python-scipy
RUN pip2 install http://download.pytorch.org/whl/cu80/torch-0.1.12.post2-cp27-none-linux_x86_64.whl
RUN pip2 install torchvision
RUN pip2 install scikit-image
RUN pip2 install --upgrade pip
RUN pip2 install ipython

# Set user specific env
ENV HOST_UID=1001

# Create user 'username'
RUN useradd --create-home --shell /bin/bash --groups sudo username
RUN echo "username:password" | chpasswd

# Change id of username
RUN usermod -u $HOST_UID username
RUN groupmod -g $HOST_UID username
