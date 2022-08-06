# End-to-end Example For Sedna Re-ID Feature

 ## Motivation

Sedna is a general collaborative framework for AI training and inference across cloud and edge. It supports joint inference, incremental learning, federated learning, lifelong learning and so on. Currently, Sedna has released a new feature -- `Multi-edge tracking Re-ID`. But it falls far short of examples for this feature. It is needed to extend more examples based on Sedna collaborative features, so that Sedna can be used more widely.

Now, the architecture of the Re-ID feature is as below (see in [Re-ID Tutorial](https://github.com/vcozzolino/sedna/blob/feature-reid-ct/examples/multiedgetracking/tutorial/tutorial.md)):

<img src=".\images\ReID-example-for-Sedna-ReIDarch.png" style="zoom:75%;" />

Based on the architecture, this proposal is to develop an end-to-end application based on the Sedna's Re-ID feature. Under the scenario of Re-ID inference, build a user-friendly end-to-end application by taking advantage of cloud-edge collaboration. This may provide some references for people who want to use the Re-ID feature in different scenarios.

### Goals

- Based on the Re-ID feature, develop an application for video human flow statistics and security authentication.
  - Video sources from specific opensource datasets can be PRID2011, etc.
  - Video sources from multiple angles/devices are selected as different edge Pods to give full play to the multi-edge collaboration capability of KubeEdge.
  - Count the human flow at the edge side, obtain the human flow statistics on the cloud (for demonstration, analysis or prediction). In addition, the cloud side also has authentication capabilities for security purposes using the Re-ID feature.
  - Provide a user-friendly results visualization website/App using `Re-ID Manager`'s API for video streaming and other functions. 
- Extend more examples and scenarios for Sedna and its new feature -- `Multi-edge tracking Re-ID`.
  - Provide some references for developers who are interested in Sedna.
  - Provide more examples for Sedna's official documentation. 

## Proposal

We propose building an application for video human flow statistics and security authentication which can be used in industrial park, apartment and other similar scenarios. 

Human flow statistics are calculated at the edge side, and the data of the human flow can be used for site management, reasonable allocation of public resources, anti-stampede warning and so on. The cloud side records the temporal changes of the human flow (can be used to predict the human flow in the future), and the Re-ID module is used for security authentication to identify suspicious characters.

### Use Cases

- Users can query human flow statistics from the cloud side.
- Users can use the history human flow data recorded on the cloud to predict human flow in the future.
- Users can query information about suspicious character through the records in the database.
- The system can help administrators identify the name/ID of people whose information has already been stored in the database.

## Design Details

### Architecture and Modules

<img src=".\images\ReID-example-for-Sedna-AppArch.png" style="zoom:70%;" />

- `Re-ID Manager`

  It exposes API so that users can invoke corresponding functions. It can interact between users and internal services, which belongs to the `Global Manager` in Sedna. The Pod requires only CPU support and is deployed in the cloud. In addition, it can output result video stream after being processed by other pods.

- `FE-Re-ID Pod`

  Feature extraction and Re-ID are conducted in this pod, which is based on the preprocessing results provided by edge node -- `Dectection/Tracking Pods`. At the same time, relevant functions are invoked by API in `Re-ID Manager Pod`. The Pod requires a GPU or CPU and is deployed in the cloud.

- `Database`

  It contains three kinds of data.

  - User Feature: It is used for identity comparison in Re-ID, which may be collected in advance.
  - Re-ID Records: It contains `time`, `cameraID`, `personID`, `direction`, etc. They are updated when people enter/exit the building, which can be used for security authentication.
  - Human Flow Statistics: It contains `time`, `gateID`, `inNum`, `outNum`, etc. They are updated after a specific interval, such as one hour, which can be used to display the total number of persons in the building or carry out data analysis for other purpose.

- `Traffic Prediction Pod`(optional)

  It is an optional pod, which can be added when human flow prediction is needed. It can use the human flow statistics to train a time-series prediction model and used to predict human flow statistics in the future.

- `Website / App`

  It receives results provided by `Re-ID Manager` API and display them visually on the website or APP. The content includes the video processed by the algorithm, the data of human flow and so on. This module is the demonstration module of the case, which can be customized by the user.

- `Detection/Tracking Pods`

  It take care of object detection and tracking. specifically, it receives video stream from edge devices and provides preprocessing for `FE-Re-ID Pod` in the cloud. Video stream can come from surveillance cameras and other sources. The Pod requires GPU or CPU support and is deployed at the edge.

## Road Map

### 2022.07

- Determine what opensource dataset to be used (used for video data source).
- Use video data to run through the entire KubeEdge architecture and Sedna Re-ID feature. See in [Re-ID Tutorial](https://github.com/vcozzolino/sedna/blob/feature-reid-ct/examples/multiedgetracking/tutorial/tutorial.md).

### 2022.08

- Use `Detection/Tracking Pods` to calculate human flow statistics. The AI models used in these pods may be modified.
- Based on Re-ID feature, develop the function for security authentication in `FE-Re-ID Pod`.
- Build the modules according to the architecture. Some of the modules may be modified to provide better experiences.
- Run different test cases to improve the robustness of the application.

### 2022.09

- Build a website to visualize all the result in this application(human flow statistics, security authentication\).
- Write a documentation to explain how this application works(can be used as an example for Sedna Re-ID feature and demonstrate in Sedna official documentation).