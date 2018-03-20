# Buoy Object Detection with TensorFlow's Mobilenet model
Object detection allows for the recognition, detection, and localization of multiple buoys within an image using a live video feed 

## Installation

### Download and Install Docker
```bash
sudo apt-get update
sudo apt-get install \
    apt-transport-https \
    ca-certificates \
    curl \
    software-properties-common
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
sudo add-apt-repository \
   "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
   $(lsb_release -cs) \
   stable"
sudo apt-get update
sudo apt-get install docker-ce
sudo groupadd docker
sudo usermod -aG docker $USER
```

### Build the Dockerfile to install all dependencies
>_**Note:** If the following docker commands do not work, run it with `sudo` (or log out and log back in)._

**Note:** This must be run in the root folder of this repository  
**Alternatively:** Replace `.` with `/path/to/Dockerfile`
```bash
docker build -t  tf-buoy-classifier .
```

Verify that the image has been successfully built using

```bash
$ docker images

REPOSITORY              TAG                 IMAGE ID            CREATED             SIZE
tf-buoy-classifier      latest              272ef1a20710        10 seconds ago      2.54GB
tensorflow/tensorflow   1.4.0-py3           7d680bfcec87        4 months ago        1.25GB
```

Make sure you are in the root directory of this repository and start the docker container with:

```bash
xhost +
docker run -it --rm --privileged -p 8888:8888 --env DISPLAY=$DISPLAY -v /dev/video0:/dev/video0 --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" -v $(pwd):/home/TF tf-buoy-classifier:latest
```

## Usage
>_The following must be run inside the docker container_

### Build and install python
If running for the first time, run:

```bash
python3 -B setup.py build
python3 -B setup.py install
```

```bash
python3 -B object_detection/object_detection_runner.py
```

## Based of

[MIT LICENSE](LICENSE)


[Realtime-Object-Detection](https://github.com/GustavZ/realtime_object_detection)