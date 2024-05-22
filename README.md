Repository for code related to the project "Feasibility Testing of Real-Time Object Segmentation and Recognition Models in Small Embedded Systems ", 
a project engineered throughout the duration of a Masters in Electronic & Computer Engineering. 

## Project Abstract
With the rapid growth of computer vision and artificial intelligence technologies, there is a growing emphasis on augmenting small embedded systems with deep learning capabilities. The aim of this project is to analyse and demonstrate the feasibility of integrating object segmentation and recognition models with these systems via connection to a remote deep learning capable computation platform. Through the deployment of traditional algorithms and modern inference methods, the study aims to enhance the efficiency and capabilities of small-scale systems in resource-constrained environments. Results indicate that while on-device machine learning offers reliability and no need for a network connection, it entails substantial computational overheads, greatly limiting the performance of viable models. Conversely, network-based inference methods are capable of fast inference speeds, usage of large cutting-edge models, and real-time connection speeds on reliable networks. The findings underscore the importance of tailoring system architectures to specific application requirements and optimizing machine learning models for embedded deployment. Ultimately, this research contributes to the broader embedded system field, offering insights into the practical implementation challenges and potential avenues for future advancements.

## Installation
For the model, follow the README installation instructions in the FastSAM subfolder. 
For RasPi scripts, you'll need a Raspberry Pi, obviously. And a PiCam or webcam.
Clone the FastSAM repo from Ultralytics [directly](https://docs.ultralytics.com/models/fast-sam/#fastsam-official-usage).
This project was built using Python 3.9.


## Usage
Most of the Python scripts are self contained, able to run when the appropriate interpreter is selected and all libraries installed. 

*Caution: It's far too easy to get stuck in dependency hell when installing this project's packages, you've been warned!!!*
 

## Contributing
Not sure why you'd want to, but pull requests are welcome. For changes, please open an issue first
to discuss what you would like to change.


## License

[MIT](https://choosealicense.com/licenses/mit/)
