### Traffic Light classifier

To detect and classify the traffic lights in our camera images we use the [Tensorflow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection). This is a collection of pre-trained models, and high level subroutines that facilitate the use and fine-tuning of these models. The models are compiled in [Tensorflow detection model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md), and mainly belong to two object detection methods: SSD (Single Shot Detector) and R-CNN (Regions with CNN).

The utilities to use the models include:

- An inference script in the form of a Jupyter Notebook, to detect objects on an image from a "frozen_inference_graph.pb" ([Object Detection Demo](https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb))
- Tools to create TFRecord files from original data ([dataset tools](https://github.com/tensorflow/models/tree/master/research/object_detection/dataset_tools))
- A training script to fine-tune a pre-trained model with our own dataset, locally or in Google Cloud ([model_main.py](https://github.com/tensorflow/models/blob/master/research/object_detection/model_main.py))
- A script to export a new "frozen_inference_graph.pb" from a fine-tuned model ([export_inference_graph.py](https://github.com/tensorflow/models/blob/master/research/object_detection/export_inference_graph.py))

The flow to get all this going includes:

- Clone and install tensorflow/models ([Installation](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md))
- Try out the tutorial notebook with an off-the-shelf model ([Quick Start: Jupyter notebook for off-the-shelf inference](https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb))
- Get our own data into TFRecord files ([Preparing inputs](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/preparing_inputs.md))
- Get the pre-trained model we want to use ([Tensorflow detection model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md))
- Modify the pipeline.config file to describe our setup ([Configuring an object detection pipeline](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/configuring_jobs.md))
- Run model_main.py locally: ([Running locally](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/running_locally.md))
	- Test with a few train steps just to quickly check everything is working
	- Export for inference
	- Try on the tutorial notebook
- Run model_main.py on Google Cloud Platform (GCP) ([Running on the cloud](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/running_on_cloud.md)):
	- Get a Google Cloud account, setup a project and a bucket ([Getting Started](https://cloud.google.com/ml-engine/docs/tensorflow/getting-started-training-prediction))
	- Install google-cloud-sdk locally ([SDK documentation](https://cloud.google.com/sdk/docs/))
	- Upload model and data to the bucket, modifying pipeline.config to GCP setup
	- Package our local tools to send and run on GCP
	- Create YAML file
	- Run script on GCP (~ 1.5 hours for train_steps=50000) 
	- Download new model
	- Export for inference
	- Try on the tutorial notebook









---

---


This is the project repo for the final project of the Udacity Self-Driving Car Nanodegree: Programming a Real Self-Driving Car. For more information about the project, see the project introduction [here](https://classroom.udacity.com/nanodegrees/nd013/parts/6047fe34-d93c-4f50-8336-b70ef10cb4b2/modules/e1a23b06-329a-4684-a717-ad476f0d8dff/lessons/462c933d-9f24-42d3-8bdc-a08a5fc866e4/concepts/5ab4b122-83e6-436d-850f-9f4d26627fd9).

Please use **one** of the two installation options, either native **or** docker installation.

### Native Installation

* Be sure that your workstation is running Ubuntu 16.04 Xenial Xerus or Ubuntu 14.04 Trusty Tahir. [Ubuntu downloads can be found here](https://www.ubuntu.com/download/desktop).
* If using a Virtual Machine to install Ubuntu, use the following configuration as minimum:
  * 2 CPU
  * 2 GB system memory
  * 25 GB of free hard drive space

  The Udacity provided virtual machine has ROS and Dataspeed DBW already installed, so you can skip the next two steps if you are using this.

* Follow these instructions to install ROS
  * [ROS Kinetic](http://wiki.ros.org/kinetic/Installation/Ubuntu) if you have Ubuntu 16.04.
  * [ROS Indigo](http://wiki.ros.org/indigo/Installation/Ubuntu) if you have Ubuntu 14.04.
* [Dataspeed DBW](https://bitbucket.org/DataspeedInc/dbw_mkz_ros)
  * Use this option to install the SDK on a workstation that already has ROS installed: [One Line SDK Install (binary)](https://bitbucket.org/DataspeedInc/dbw_mkz_ros/src/81e63fcc335d7b64139d7482017d6a97b405e250/ROS_SETUP.md?fileviewer=file-view-default)
* Download the [Udacity Simulator](https://github.com/udacity/CarND-Capstone/releases).

### Docker Installation
[Install Docker](https://docs.docker.com/engine/installation/)

Build the docker container
```bash
docker build . -t capstone
```

Run the docker file
```bash
docker run -p 4567:4567 -v $PWD:/capstone -v /tmp/log:/root/.ros/ --rm -it capstone
```

### Port Forwarding
To set up port forwarding, please refer to the "uWebSocketIO Starter Guide" found in the classroom (see Extended Kalman Filter Project lesson).

### Usage

1. Clone the project repository
```bash
git clone https://github.com/udacity/CarND-Capstone.git
```

2. Install python dependencies
```bash
cd CarND-Capstone
pip install -r requirements.txt
```
3. Make and run styx
```bash
cd ros
catkin_make
source devel/setup.sh
roslaunch launch/styx.launch
```
4. Run the simulator

### Real world testing
1. Download [training bag](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/traffic_light_bag_file.zip) that was recorded on the Udacity self-driving car.
2. Unzip the file
```bash
unzip traffic_light_bag_file.zip
```
3. Play the bag file
```bash
rosbag play -l traffic_light_bag_file/traffic_light_training.bag
```
4. Launch your project in site mode
```bash
cd CarND-Capstone/ros
roslaunch launch/site.launch
```
5. Confirm that traffic light detection works on real life images

### Other library/driver information
Outside of `requirements.txt`, here is information on other driver/library versions used in the simulator and Carla:

Specific to these libraries, the simulator grader and Carla use the following:

|        | Simulator | Carla  |
| :-----------: |:-------------:| :-----:|
| Nvidia driver | 384.130 | 384.130 |
| CUDA | 8.0.61 | 8.0.61 |
| cuDNN | 6.0.21 | 6.0.21 |
| TensorRT | N/A | N/A |
| OpenCV | 3.2.0-dev | 2.4.8 |
| OpenMP | N/A | N/A |

We are working on a fix to line up the OpenCV versions between the two.
