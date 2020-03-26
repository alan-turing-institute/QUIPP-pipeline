# Binder setup

This folder contains the configuration files to set up the QUIPP-pipeline repository on the public Binder service, [`mybinder.org`](https://mybinder.org/).

The QUIPP-pipeline project uses libraries in Python, R and C++.
We have created a Docker image, available on Docker Hub, which has all the dependencies pre-installed.
A guide on how this image was created can be found in [`QUIPP-pipeline/env-configuration/README.md`](../env-configuration/README.md).

When a user chooses to access this repository on `mybinder.org`, the Dockerfile in this folder is used by repo2docker to generate a new image (unless the image has recently been cached by another user who accessed the same commit).
The Dockerfile consists of three main steps:
- The pre-generated Docker image is used as a base. This image contains all the dependencies of this project, but no code.
- Some environment variables are set for JupyterHub.
- The files in this repository are copied into the image.

Once the image has been prepared, the user will be directed to a their own instance of a container based on that image, hosted in the cloud.
They can then explore and interact with the code in this repository, without having to install anything themselves.
For more details about Binder, see the Binder Project's ["Getting Started"](https://mybinder.readthedocs.io/en/latest/index-getting-started.html) guide.

Note that a user's Binder container is not persistant, and they will not be able to save their changes.
If they would like to do so, we recommend setting up a local copy of this repository.
