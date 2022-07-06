# Development Guide

This document is intended to provide contributors with an introduction to developing a runnable algorithm module of the Sedna project. 

The Sedna framework components are decoupled and the registration mechanism is used to combine functional components to facilitate function and algorithm expansion. For details about the Sedna architecture and main mechanisms, see [Lib README](/lib/sedna/README.md).

During Sedna application development, the first problem encountered is how to import service data sets to Sedna. For details, see [Datasets Guide](./datasets.md).

For different algorithms, see [Algorithm Development Guide](./new_algorithm.md). You can add new algorithms to Sedna step by step based on the examples provided in this document.

Before develop a module, follow [lib API Reference](https://sedna.readthedocs.io/en/latest/autoapi/lib/sedna/index.html) to learn about the interface design of sedna.
