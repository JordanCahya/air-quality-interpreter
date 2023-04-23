# Air Quality Interpreter

**Project Architecture:**

![image](https://user-images.githubusercontent.com/115296804/233829668-6224bc3f-58a8-4e88-af6a-c4cf69de6141.png)

## 1. Brief Documentation

The air quality interpreter was developed using **Visual Studio Code** as the code editor on a virtual machine that utilized **Window Subsystem for Linux version 2 (WSL 2)**. The necessary modules and packages to build the air quality interpreter are listed in the _"requirements.txt"_ file. 

The general workflow is as follows:

![image](https://user-images.githubusercontent.com/115296804/233823203-c2957192-bac2-4a59-a456-7ffb3a256957.png)

After data has been processed (visit https://medium.com/p/ddf7d6a9ed16/edit for more information), the FastAPI and Streamlit are then deployed and run within a container on the **Docker platform**. Supposedly, the FastAPI will provide a bridge between the backend and the frontend (Streamlit), however the frontend failed to read the gateway link provided by the FastAPI. Therefore, the frontend has been embedded with several codes from the FastAPI considering the time constraint. Due to unavailable resources, the deployment needs to be stopped only at Docker Containers, as most of the well-known deployment platform requires a subscription.

The generated response of prediction from the FastAPI is boolean either good air quality or bad air quality that will be discuss and display in the later section.

## 2. Description of Project

Air pollution is a serious environmental issue that has far-reaching consequences on human health. Exposure to polluted air can lead to a range of respiratory problems, such as bronchitis, asthma, and even lung cancer. Moreover, air pollution is not just limited to respiratory illnesses; it can also cause eye problems, such as glaucoma, as well as heart disease and stroke.

The goal of this project is to increase public awareness of air quality in Jakarta and encourage residents to take measures to protect their health by interpreting information on air quality on their own. Therefore, residents can informed decisions about where to engage in outdoor activities.

An air quality interpreter is created and trained using a certain dataset from https://data.jakarta.go.id/dataset/indeks-standar-pencemaran-udara-ispu-tahun-2021 to reach the goal of this project. The database contains a dataset of the Air Pollution Standard Index (ISPU) measured from the Air Quality Monitoring Station (SPKU). The interpreter is trained on a dataset ranging from June to December 2021, to give a more reliable interpretation of air quality, which has the following features:

- station = (str) the location where the measurement is taken
- pm10 = (int) Air particles that are smaller than 10 microns (micrometers) in size.
- pm25 = (int) Air particles that are smaller than 2.5 microns (micrometers) in size.
- so2 = (int) sulfur dioxide - a colorless gas or liquid with a strong, choking odor.
- co = (int) carbon monoxide - an odorless, colorless gas formed by the incomplete combustion of fuels.
- o3 = (int) ozone - a highly reactive gas.
- no2 = (int) nitrogen dioxide - highly reactive gases.
- category = (str) the category of the air pollution standard index calculation result.

## 3. How to Use the Air Quality Interpreter in Local Host

- Ensure laptop have been installed with docker desktop

- Ensure code editor (in my case VS Code) have been integrated with docker desktop

- Deploy the Streamlit and the FastAPI to Docker Platform using "docker build -t <name_of_project>". Each of the Streamlit and the FastAPI need to be deploy separately using different project name.

Example for Streamlit = docker build -t air-quality-interpreter-deployment
Example for FastAPI = docker build -t air-quality-interpreter

- Build the container for the Streamlit and the FastAPI that have been deployed using "docker run -p <8501:8501 (for Streamlit)/8000:8000 (for FastAPI)> <name_of_project>. The name of the project need to be consistent with the deployment.

Example for Streamlit = docker run -p 8501:8501 air-quality-interpreter-deployment
Example for FastAPI = docker run -p 8000:8000 air-quality-interpreter

- The container should be run automatically after building the container. If the container is already stop working, visit the docker container, press start below Actions, and click on the port.

![image](https://user-images.githubusercontent.com/115296804/233839926-71d03905-6d9a-467d-bfc8-c8cdb0246de9.png)

- For backend without User Interface, press start on the FastAPI or container with port 8000:8000.

- For frontend with User Interface, press start on Streamlit or container with port 8501:8501.

- Supposedly, the frontend with User Interface is started and the port 8501:8501 has been clicked. Then the computer will redirected to browser and open a website which display as follows: 

![image](https://user-images.githubusercontent.com/115296804/233840647-7234d24f-1da0-4489-81f3-2705fe111c67.png)

From the above images, it can be seen that the user will need to input several input feature such as stasiun, pm10, pm25, so2, co, o3, and no2. The interpreter will later generate a boolean output wheteher it is Good Air Quality or Bad Air Quality.

## 4. Limitation

- Due to time constraints, the streamlit has been embedded with several codes from the fastAPI file as the streamlit failed to read the gateway link provided by the fastAPI. Therefore, no bridge/connection between the fastAPI and frontend (Streamlit).

- Due to unavailable resources, the deployment needs to be stopped only at Docker Containers, as most of the commonly used deployment platform requires a subscription.

## 5. Reference

- Pacmann Class.
- FastAPI, Docker, and Streamlit youtube videos by AssemblyAI and Krish Naik.
