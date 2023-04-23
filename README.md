# Air Quality Interpreter

## 1. Brief Documentation

The air quality interpreter was developed using **Visual Studio Code** as the code editor on a virtual machine that utilized **Window Subsystem for Linux version 2 (WSL 2)**. The necessary modules and packages to build the air quality interpreter are listed in the _"requirements.txt"_ file. 

The FastAPI and Streamlit are then deployed and run within a container on the **Docker platform**. Supposedly, the FastAPI will provide a bridge between the backend and the frontend (Streamlit), however the frontend failed to read the gateway link provided by the FastAPI. Therefore, the frontend has been embedded with several codes from the FastAPI.

The response of prediction from the FastAPI is boolean i.e., good air quality or bad air quality that will be display in the later section.











