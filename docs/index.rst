===========================================
Sedna documentation
===========================================

.. image:: ./_static/logo.png

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

    setup/*


.. toctree::
    :maxdepth: 1
    :glob:
    :caption: EXAMPLES

    examples/federated_learning/surface_defect_detection/README
    examples/incremental_learning/helmet_detection/README
    examples/joint_inference/helmet_detection_inference/README
    examples/lifelong_learning/atcii/README
    examples/storage/s3/README


.. toctree::
    :maxdepth: 1
    :caption: API
    :titlesonly:
    :glob:

    api/crd/*
    api/lib/*

.. toctree::
    :maxdepth: 1
    :caption: Contributing
    :titlesonly:
    :glob:

    contributing/*

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


Community
=========

Sedna is an open source project and in the spirit of openness and freedom, we welcome new contributors to join us.
You can get in touch with the community according to the ways:
* [Github Issues](https://github.com/kubeedge/sedna/issues)
* [Regular Community Meeting](https://zoom.us/j/4167237304)
* [slack channel](https://app.slack.com/client/TDZ5TGXQW/C01EG84REVB/details)


RELATED LINKS
=============

.. mdinclude:: related_link.md

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
