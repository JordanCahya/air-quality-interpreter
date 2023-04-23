# Air Quality Interpreter

## 1. Brief Documentation

The air quality interpreter was developed using **Visual Studio Code** as the code editor on a virtual machine that utilized **Window Subsystem for Linux version 2 (WSL 2)**. The necessary modules and packages to build the air quality interpreter are listed in the _"requirements.txt"_ file. 

The general workflow is as follows:

![image](https://user-images.githubusercontent.com/115296804/233823203-c2957192-bac2-4a59-a456-7ffb3a256957.png)

After data has been processed (visit https://medium.com/p/ddf7d6a9ed16/edit for more information), the FastAPI and Streamlit are then deployed and run within a container on the **Docker platform**. Supposedly, the FastAPI will provide a bridge between the backend and the frontend (Streamlit), however the frontend failed to read the gateway link provided by the FastAPI. Therefore, the frontend has been embedded with several codes from the FastAPI considering the time constraint. Due to unavailable resources, the deployment needs to be stopped only at Docker Containers, as most of the well-known deployment platform requires a subscription.

The generated response of prediction from the FastAPI is boolean either good air quality or bad air quality that will be discuss and display in the later section.

## 2. Description of Project















