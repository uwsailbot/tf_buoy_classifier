# Dockerfile with tensorflow cpu support on python3, opencv3.3
FROM tensorflow/tensorflow:1.4.0-py3

RUN apt-get update

ENV NUM_CORES 2

RUN apt-get -y update -qq && \
    apt-get -y install wget \
                       apt-utils \
                       unzip \

                       # Required
                       build-essential \
                       cmake \
                       git \
                       pkg-config \
                       libatlas-base-dev \
                       libgtk2.0-dev \
                       libavcodec-dev \
                       libavformat-dev \
                       libswscale-dev \

                       # Optional
                       libtbb2 libtbb-dev \
                       libjpeg-dev \
                       libpng-dev \
                       libtiff-dev \
                       libv4l-dev \
                       libdc1394-22-dev \

                       qt4-default \

                       # Missing libraries for GTK
                       libatk-adaptor \
                       libcanberra-gtk-module \

                       # Tools
                       imagemagick \

                       # For use matplotlib.pyplot in python
                       python3-tk

# Build and install OpenCV
WORKDIR /
RUN wget https://github.com/opencv/opencv/archive/3.3.0.zip \
	&& unzip 3.3.0.zip \
	&& mkdir /opencv-3.3.0/cmake_binary \
	&& cd /opencv-3.3.0/cmake_binary \
	&& cmake -D CMAKE_BUILD_TYPE=RELEASE \
        -D CMAKE_INSTALL_PREFIX=$(python3 -c "import sys; print(sys.prefix)") \
        -D INSTALL_C_EXAMPLES=OFF \
        -D INSTALL_PYTHON_EXAMPLES=OFF \
        -D WITH_IPP=OFF \
	-D WITH_TBB=ON \
        -D ENABLE_AVX=ON \
        -D WITH_CUDA=OFF \
     	-D PYTHON_EXECUTABLE=$(which python3) \
        -D PYTHON_DEFAULT_EXECUTABLE=$(which python3) \
        -D PYTHON_INCLUDE_DIR=$(python3 -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())") \
        -D PYTHON_LIBRARY=$(python3 -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR')+'/'+sysconfig.get_config_var('LDLIBRARY'))") \
        -D PYTHON3_EXECUTABLE=$(which python3) \
        -D PYTHON3_INCLUDE_DIR=$(python3 -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())") \
        -D PYTHON3_INCLUDE_PATH=$(python3 -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())") \
        -D PYTHON3_LIBRARIES=$(python3 -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR')+'/'+sysconfig.get_config_var('LDLIBRARY'))") \
        -D PYTHON3_PACKAGES_PATH=$(python3 -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())") \
        -D BUILD_EXAMPLES=OFF .. \
	&& make -j$NUM_CORES \
        && make install \
        && cd / \
        && rm /3.3.0.zip \
	&& rm -r /opencv-3.3.0


RUN apt-get -y install libgtk-3-dev \
                       libboost-all-dev

RUN pip3 --no-cache-dir install \
    h5py \
    numpy \
    hdf5storage \
    scipy \
    py3nvml \
    scikit-image \
    PyYAML \
    pillow \
    lxml \
    matplotlib

WORKDIR "/home/TF"

CMD ["/bin/bash"]