# FS-AI
2# FSAI Simulator Setup

## Computer Specs Required
- 8 core 2.3Ghz CPU
- 12 GB memory
- 30GB free SSD storage (120GB if building unreal from source)
- Recent NVidia card with Vulkan support and 3 GB of memory
- You can check the video card drivers by running `vulkaninfo`
- Different brand video cards might work but have not been tested
- (Working on VM/RW setup, for those who don't meet specs)

## Pre-requisites
- Install latest version of [Unreal Engine](https://www.unrealengine.com/en-US/download)
- Original setup documentation for [FSDS](https://fs-driverless.github.io/Formula-Student-Driverless-Simulator/v2.2.0/)

## 1. WSL 1

1. Install [Windows Terminal](https://www.microsoft.com/store/productId/9N0DX20HK701?ocid=pdpshare)
2. Open Windows PowerShell as administrator
3. Enable WSL and set to version 1
    ```powershell
    dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart        
    ```
    ```powershell
    wsl --set-default-version 1
    ```

## 2. Ubuntu 18.04.6

1. Install [Ubuntu 18.04.6](https://www.microsoft.com/store/productId/9PNKSF5ZN4SW?ocid=pdpshare)
2. Open Ubuntu 18.04.6

## 3. Required Packages

1. Update, upgrade, install local packages
    ```
    sudo apt update 
    ```
    ```
    sudo apt upgrade -y sudo apt install curl git python3-pip -y
    ```
    ```
    sudo apt install curl git python3-pip -y
    ```

## 4. FSDS Binaries

1. Install [fsds-v2.2.0-windows](https://github.com/FS-Driverless/Formula-Student-Driverless-Simulator/releases)
2. Extract all `fsds-v2.2.0-windows` files
3. Run the `FSDS.exe` file

## 5. ROS Melodic

1. Setup `sources.list`
    ```
    sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
    ```
2. Setup your keys
    ```
    sudo apt install curl
    ```
    ```
    curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add -
    ```
3. Install Melodic
    ```
    sudo apt update
    ```
    ```
    sudo apt install ros-melodic-desktop-full
    ```
4. Setup environment with `bash.rc`
    ```
    echo "source /opt/ros/melodic/setup.bash" >> ~/.bashrc
    ```
    ```
    source ~/.bashrc
    ```
5. Install dependencies
    ```
    sudo apt install python-rosdep python-rosinstall python-rosinstall-generator python-wstool build-essential
    ```
6. Initialise rosdep
    ```
    sudo apt install python-rosdep
    ```
    ```
    sudo rosdep init
    ```
    ```
    rosdep update
    ```
7. Add following lines to `~/.bashrc` file
    ```
    nano ~/.bashrc
    ```
    ```
    source /opt/ros/melodic/setup.bash
    source ~/Formula-Student-Driverless-Simulator/ros/devel/setup.bash
    ```

## 6. ROS Bridge

1. Install ROS Bridge dependencies
    ```
    sudo apt-get install ros-melodic-tf2-geometry-msgs python-catkin-tools ros-melodic-rqt-multiplot ros-melodic-joy ros-melodic-cv-bridge ros-melodic-image-transport libyaml-cpp-dev libcurl4-openssl-dev
    ```
2. Install `git-lfs`
    ```
    sudo apt install git-lfs
    ```
3. Clone the FSDS repository (Must be cloned into the home directory) (`$HOME/Formula-Student-Driverless-Simulator`)
    ```
    git clone https://github.com/FS-Driverless/Formula-Student-Driverless-Simulator.git --recurse-submodules
    ```
4. Checkout the required FSDS version
    ```
    cd ~/Formula-Student-Driverless-Simulator
    ```
    ```
    git checkout tags/v2.2.0
    ```

## 7. AirLib

1. Run the AirSim setup script
    ```
    cd ~/Formula-Student-Driverless-Simulator/AirSim
    ```
    ```
    ./setup.sh
    ```

## 8. ROS Catkin Workspace

1. Navigate, initialise and build workspace
    ```
    cd ~/Formula-Student-Driverless-Simulator/ros
    ```
    ```
    catkin init
    ```
    ```
    catkin build
    ```

## 9. Launching ROS Bridge

1. All ROS Bridge nodes can be launched using `fsds_ros_bridge.launch` (ROS Bridge reads settings from `~/Formula-Student-Driverless-Simulator/settings.json`) (Ensure the settings are the same as the FSDS settings)
    ```
    cd ~/Formula-Student-Driverless-Simulator/ros
    ```
    ```
    source devel/setup.bash
    ```
    ```
    roslaunch fsds_ros_bridge fsds_ros_bridge.launch
    ```

## 10. Required Python Libraries

1. Install Python libraries (Install libraries separately to avoid cross-dependency errors) (`opencv-contrib-python` install may take `1-2hrs` to execute)
    ```
    cd ~
    ```
    ```
    sudo apt install python3 python3-pip
    ```
    ```
    pip3 install scikit-build
    ```
    ```
    pip3 install matplotlib
    ```
    ```
    pip3 install numpy
    ```
    ```
    pip3 install opencv-contrib-python
    ```
2. Install `msgpack-rpc-python` library
    ```
    git clone https://github.com/msgpack-rpc/msgpack-rpc-python.git
    ```
    ```
    cd msgpack-rpc-python
    ```
    ```
    pip3 install .
    ```

## 11. Test Run Whole System

1. Ensure both `settings.json` files are the exact same
2. `Formula-Student-Driverless-Simulator` `settings.json` and `fsds-v2.2.0-windows` `settings.json` must be the following (for now)

    ```json
    {
    "SettingsVersion": 1.2,
    "Vehicles": {
        "FSCar": {
        "DefaultVehicleState": "",
        "EnableCollisionPassthrogh": false,
        "EnableCollisions": true,
        "AllowAPIAlways": true,
        "RC":{
            "RemoteControlID": -1
        },
        "Sensors": {
            "Gps" : {
            "SensorType": 3,
            "Enabled": true
            },
            "Lidar": {
            "SensorType": 6,
            "Enabled": true,
            "X": 1.3, "Y": 0, "Z": 0.1,
            "Roll": 0, "Pitch": 0, "Yaw" : 0,
            "NumberOfLasers": 1,
            "PointsPerScan": 500,
            "VerticalFOVUpper": 0,
            "VerticalFOVLower": 0,
            "HorizontalFOVStart": -90,
            "HorizontalFOVEnd": 90,
            "RotationsPerSecond": 10,
            "DrawDebugPoints": true
            }
        },
        "Cameras": {},
        "X": 0, "Y": 0, "Z": 0,
        "Pitch": 0, "Roll": 0, "Yaw": 0
        }
    }
    }
    ```
2. In order to change the settings.json in the cloned repo, navigate to the correct directory and open nano
   ```
   cd Formula-Student-Driverless-Simulator
   ```
   ```
   nano settings.json
   ```
4. Close all Ubuntu terminals
5. Launch `FSDS.exe`, then run with spectator server
6. Open `1st Ubuntu terminal`, to ROS and launch ROS Bridge 
    ```
    cd Formula-Student-Driverless-Simulator/ros
    ```
    ```
    source devel/setup.bash
    ```
    ```
    roslaunch fsds_ros_bridge fsds_ros_bridge.launch
    ```
7. Open `2nd Ubuntu terminal`, to python code and run script
    ```
    cd Formula-Student-Driverless-Simulator/python/examples
    ```
    ```
    python3 autonomous_example.py
    ```
8. Check if the model car is moving in the FSDS.exe

## Success!
1. Congrats! You’ve now done what has taken us FOREVER to sort out in a lot less time (Tyler spent 4 hours in the CS lab on one error).

2. If there are any errors or improvements for this setup guide, please let us know or put in a pull request for amends.

3. Great success! Now go test your own scripts on the simulator you just set up!

## Time Savers
### Adding settings.json to bash to make it easier to access

1. Open the text editor
   ```
   nano ~/.bashrc
   ```
2. Add an alias at the end of the file (call it whatever you want I've just called it editsettings)
    ```
    alias editsettings='cd ~ && cd Formula-Student-Driverless-Simulator && nano settings.json'
    ```
3. Run this command to save it
   ```
   source ~/.bashrc
   ```
4. Now whenever you want to edit he settings.json file you can just run:
   ```
   editsettings
   ```

### Adding the Formula-Student-Driverless-Simulator directory to bash to make it quicker to access

1. Open the text editor
   ```
   nano ~/.bashrc
   ```
2. Add an alias at the end of the file (call it whatever you want I've just called it fs)
    ```
    alias fs='cd Formula-Student-Driverless-Simulator'
    ```
3. Run this command to save it
   ```
   source ~/.bashrc
   ```
4. Now whenever you want to get to that directory you can just run:
   ```
   fs
   ```
### Finally you can do this for running the ROS Bridge (biggest time saver here)

1. Open the text editor
   ```
   nano ~/.bashrc
   ```
2. Add the following lines to create a function (call it whatever you want (are you seeing the pattern))
    ```
    runrosbridge() { 
    cd ~/Formula-Student-Driverless-Simulator/ros 
    source devel/setup.bash 
    roslaunch fsds_ros_bridge fsds_ros_bridge.launch
    }
    ```
3. Run this command to save it
   ```
   source ~/.bashrc
   ```
4. Now whenever you want to get to load the ROS Bridge yuo just run:
   ```
   runrosbridge
   ```
