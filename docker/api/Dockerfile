FROM tiangolo/uvicorn-gunicorn-fastapi:python3.10
WORKDIR /app
COPY ./requirements.txt /app/requirements.txt
RUN \
apt-get update && \
apt-get upgrade -y && \
apt-get autoremove -y && \
apt-get clean -y && \
pip install --upgrade pip && \
pip install wheel && \
pip install -r requirements.txt
COPY ./app /app/app
EXPOSE 8000
CMD ["python", "app/src/api.py"]