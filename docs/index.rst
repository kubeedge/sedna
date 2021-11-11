===========================================
Sedna documentation
===========================================

.. image:: ./_static/logo.png
   :width: 200

Sedna is an edge-cloud synergy AI project incubated in KubeEdge SIG AI. Benefiting from the edge-cloud synergy capabilities provided by KubeEdge, Sedna can implement across edge-cloud collaborative training and collaborative inference capabilities, such as joint inference, incremental learning, federated learning, and lifelong learning. Sedna supports popular AI frameworks, such as TensorFlow, Pytorch, PaddlePaddle, MindSpore.

Sedna can simply enable edge-cloud synergy capabilities to existing training and inference scripts, bringing the benefits of reducing costs, improving model performance, and protecting data privacy.


.. toctree::
    :maxdepth: 1
    :caption: GUIDE

    index/guide
    index/quickstart


.. toctree::
    :maxdepth: 1
    :titlesonly:
    :glob:
    :caption: DEPLOY

    Cluster Installation (for production) <setup/install>
    AllinOne Installation (for development) <setup/all-in-one>
    Standalone Installation (for hello world) <setup/local-up>


.. toctree::
    :maxdepth: 1
    :caption: INTRODUCTION
    :hidden:

    proposals/architecture
    proposals/dataset-and-model
    proposals/federated-learning
    proposals/incremental-learning
    proposals/joint-inference
    proposals/lifelong-learning
    proposals/object-search
    proposals/object-tracking


.. toctree::
    :maxdepth: 1
    :glob:
    :caption: EXAMPLES

    examples/joint_inference/helmet_detection_inference/README
    examples/incremental_learning/helmet_detection/README
    examples/federated_learning/surface_defect_detection/README
    examples/federated_learning/yolov5_coco128_mistnet/README
    examples/lifelong_learning/atcii/README
    examples/storage/s3/*


.. toctree::
    :maxdepth: 1
    :caption: API REFERENCE
    :titlesonly:
    :glob:

    api/lib/*
    Python API <autoapi/lib/sedna/index>


.. toctree::
    :maxdepth: 1
    :caption: Contributing
    :titlesonly:
    :glob:

    Control Plane <contributing/prepare-environment>


.. toctree::
    :caption: ROADMAP
    :hidden:

    index/roadmap


RELATED LINKS
=============

.. mdinclude:: index/related_link.md

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
