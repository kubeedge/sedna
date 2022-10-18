[toc]

# The proposal about building a high frequency Sedna based end to end use case in ModelBox format

## 1. Background

> Sedna is an edge cloud collaborative AI project at KubeEdge SIG AI. Benefiting from the edge-cloud collaboration capabilities provided by KubeEdge, Sedna can realize cross-edge-cloud collaborative training and collaborative reasoning capabilities, and Sedna supports a variety of popular AI frameworks. **Sedna can simply enable edge-cloud synergy for existing training and inference scripts**, bringing the benefits of lower costs, improved model performance, and data privacy. **The goal of ModelBox is to solve the programming complexity of AI developers when developing AI applications**, reduce the development difficulty of AI applications, and hand over complex data processing to ModelBox. Developers mainly focus on the business logic itself, not the software details. While improving the efficiency of AI inference development, it ensures the performance, reliability, security and other attributes of the software, **so as to better enable Sedna developers to manage and use the Sedna framework**.

## 2. Motivation

### Goals

- ModelBox provides visual layout function to help Sedna developers improve the programming complexity when developing edge cloud collaborative AI.
- ModelBox provides support for multiple hardware devices and software frameworks to help Sedna shield heterogeneous software and hardware problems.
- ModelBox provides a high-performance scheduling engine to help Sedna developers solve the commercial performance problems of AI applications.

## 3. Proposal

- Make use of the extensible function of ModelBox, use Python to develop functional units, and package Sedna library and related codes in ModelBox. Use the visual layout function of ModelBox container to call Sedna API to realize the collaborative application development of edge cloud.![ModelBox Development Application](sedna/docs/proposals/images/ModelBox Development Application.png)

  **use case**

- Carry out the collaborative development of edge clouds on ModelBox, and use helmet detection incremental learning training samples. Start the ModelBox reasoning service, and the helmet detection video stream data is reasoned on the side. When encountering difficulties, it is uploaded to the cloud side. After the incremental sample conditions are met, the training container starts the incremental training model, puts the model in the evaluation container for evaluation, and finally starts the side reasoning service.

- **View [official website document for details]( https://modelbox-ai.com/ )**

  

  ![Helmet_detection case](sedna/docs/proposals/images/Helmet_detection case.png)

## 4. Design Details

### Flow chart of ModelBox integrated Sedna edge-cloud collaboration function

Use **ModelBox** visual development function to integrate **Sedna** related functional units. Use the **k8s** service to schedule and manage end-to-end cloud clusters. First start the **ModelBox side ** inference container. The side **Sedna LC** controls the acquisition of data, and upload the cloud to enable the side **Sedna worker **to create inference services. After normal sample inference, output the results. After encountering difficult cases, the **LC trigger** monitors that the incremental samples meet the retraining requirements, and automatically triggers the training container to start and complete the model training. After training, the model is placed in the **Evaluation container** for evaluation. Finally, after the evaluation pod is finished, the reasoning container of the side **ModelBox** is started, and then the reasoning result is output.

![ModelBox intergrated Sedna flow chart](sedna/docs/proposals/images/ModelBox intergrated Sedna flow chart.png)

### 1. Specific development process of functional unit of ModelBox

#### 1. Function unit creation

 The directory structure of the created **C++** functional unit is as follows:

```
[flowunit-name]
     |---CMakeList.txt           # For cpack packaging
     |---[flowunit-name].toml    # Configuration file for webui display
     |---[flowunit-name].py      # Interface implementation file
```

#### 2. Function unit attribute configuration

#### 3. Logical realization of functional unit

#### 4. Function unit compilation and operation

The ModelBox framework Python project is compiled. The functional units created through the command line or visual UI contain file by default. The main functions are as follows:

- Set function unit name
- Header files required for linking functional units
- Libraries required for linking functional units
- Set compilation target to dynamic library
- Specify the function unit installation directory

**View [official website document for details]( https://modelbox-ai.com/ )**



### 2. Application case of encapsulated Sedna

#### 1. Create incremental learning

#### 2. Start incremental training

**View [use incremental learning in helmet detection scenario for specific details](https://github.com/kubeedge/sedna/blob/main/examples/incremental_learning/helmet_detection/README.md)**



## 5.Road Map

**2022.07.01 - 2022.08.15**

​	**1. Installation and construction of ModelBox environment**

​	**2. Understand the development tasks of ModelBox functional units**

**2022.08.16 - 2022.09.30**

​	**1. Complete Sedna function module**

​	**2. Complete helmet detection incremental training application case**

​	**3. Topic summary and document arrangement**