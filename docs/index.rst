===========================================
Sedna documentation
===========================================

.. image:: ./_static/logo.png
   :width: 200

Sedna is an edge-cloud synergy AI project incubated in KubeEdge SIG AI. Benefiting from the edge-cloud synergy capabilities provided by KubeEdge, Sedna can implement across edge-cloud collaborative training and collaborative inference capabilities, such as joint inference, incremental learning, federated learning, and lifelong learning. Sedna supports popular AI frameworks, such as TensorFlow, Pytorch, PaddlePaddle, MindSpore.

Sedna can simply enable edge-cloud synergy capabilities to existing training and inference scripts, bringing the benefits of reducing costs, improving model performance, and protecting data privacy.

.. toctree::
    :maxdepth: 1
    :caption: QUICK START

    quickstart


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
    :titlesonly:
    :glob:
    :caption: DEPLOY

    Installtion <setup/install>
    Standalone <setup/local-up>

.. toctree::
    :maxdepth: 1
    :glob:
    :caption: EXAMPLES

    examples/federated_learning/surface_defect_detection/README
    examples/incremental_learning/helmet_detection/README
    examples/joint_inference/helmet_detection_inference/README
    examples/lifelong_learning/atcii/README
    examples/storage/s3/*


.. toctree::
    :maxdepth: 1
    :caption: API
    :titlesonly:
    :glob:

    api/crd/*
    api/lib/*

.. toctree::
    :caption: Contributing

    Prepare <contributing/prepare-environment>


.. toctree::
    :maxdepth: 1
    :caption: API REFERENCE
    :titlesonly:
    :glob:

    autoapi/lib/sedna/index


.. toctree::
    :caption: ROADMAP
    :hidden:

    roadmap


RELATED LINKS
=============

.. mdinclude:: related_link.md

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
