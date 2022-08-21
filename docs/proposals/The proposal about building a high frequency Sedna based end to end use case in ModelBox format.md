[toc]

# The proposal about building a high frequency Sedna based end to end use case in ModelBox format

## 1. Background

> Sedna is an edge cloud collaborative AI project at KubeEdge SIG AI. Benefiting from the edge-cloud collaboration capabilities provided by KubeEdge, Sedna can realize cross-edge-cloud collaborative training and collaborative reasoning capabilities, and Sedna supports a variety of popular AI frameworks. **Sedna can simply enable edge-cloud synergy for existing training and inference scripts**, bringing the benefits of lower costs, improved model performance, and data privacy. **The goal of ModelBox is to solve the programming complexity of AI developers when developing AI applications**, reduce the development difficulty of AI applications, and hand over complex data processing to ModelBox. Developers mainly focus on the business logic itself, not the software details. While improving the efficiency of AI inference development, it ensures the performance, reliability, security and other attributes of the software, **so as to better enable Sedna developers to manage and use the Sedna framework**.

## 2. Motivation

### Goals

**The ModelBox application encapsulates the Sedna edge cloud collaborative training function module and the fully functional edge cloud collaborative AI application case (incremental training function)**.

## 3. Proposal

- **Use the expandable development function of ModelBox, use C++ to develop functional units, package Sedna Library in ModelBox, and call Sedna API to realize relevant functions.**
- **Carry out flow chart unit development on ModelBox, call the encapsulated incremental learning API of Sedna library, upload data to the cloud side through the incremental learning training model, upload data to the cloud and train data respectively, call the cloud testing training API of Sedna Library in ModelBox, train in the cloud, send the training results back to the side, compare the accuracy with the side training data, and realize the incremental training helmet detection sample**



## 4. Design Details

### Flow chart of ModelBox integrated Sedna edge-cloud collaboration function

![ModelBox intergrated Sedna flow chart](images\ModelBox intergrated Sedna flow chart.png)

Use **ModelBox editor** visual development function to carry out unit development function. Use the **ModelBox** format to encapsulate the **k8s** service and call the interface to manage the **Sedna** edge cloud nodes. The incremental training model is realized. The data obtained by the side is trained by simple reasoning to obtain an inaccurate model. The data obtained by the side is transmitted to the cloud through **local controller**. The cloud performs incremental training through **global manager** to obtain a more accurate model through reasoning. The inference results of the side cloud model are output through **httpserver**. **K8s** library is encapsulated by ModelBox, and Sedna functions such as edge reasoning training and cloud incremental training are encapsulated by function unit development function.

### 1. Specific development process of functional unit of ModelBox

#### 1. Function unit creation

 The directory structure of the created **C++** functional unit is as follows:

```
[flowunit-name]
     |---CMakeList.txt           # Compile files,and C++ functional units are compiled with cmake
     |---[flowunit-name].cc      # Interface implementation file
     |---[flowunit-name].h       # Header file
     |---[flowunit-name].toml    # Configuration file for webui display
```

#### 2. Function unit attribute configuration

#### 3. Logical realization of functional unit

#### 4. Function unit compilation and operation

The ModelBox framework C++ project is compiled using **cmake**. The functional units created through the command line or visual UI contain`cmakelists.txt` file by default. The main functions are as follows:

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