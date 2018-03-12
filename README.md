# Buoy Object Detection with TensorFlow's Mobilenet model
Object detection allows for the recognition, detection, and localization of multiple buoys within an image. 

## Installation

Download Docker and run the dockerfile to install all the dependencies. Folder structure should be like this:

```bash
dockerimagename/Dockerfile
```
Run inside the dockerimagename folder. 

```bash
sudo docker build -t  dockerimagename . 
```

Check if the docker image has been build successfully with 

```bash
sudo docker image ls
```

Start the docker with: (change to appropriate paths)

```bash
xhost +
```

```bash
sudo docker run -it --privileged -p 8888:8888 --env DISPLAY=$DISPLAY -v /dev/video0:/dev/video0 --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" -v /home/path/to/cloned/repo/:/home/TF dockerimagename:latest
```

Inside the docker, add `models` and `models/slim` to your `PYTHONPATH` using:

```bash
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
```

>_**Note:** This last command must be ran every time you open terminal, or added to your `~/.bashrc` file._


## Usage
Run:

```bash
python object_detection/object_detection_runner.py
```

## Source

[MIT LICENSE](LICENSE)
