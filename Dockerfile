FROM python:3.7.5-slim

RUN mkdir -p /app
WORKDIR /app
ADD requirements.txt /app/requirements.txt
RUN pip install -r requirements.txt
COPY . /app
# Python commands run inside the virtual environment
CMD streamlit run app.py --server.port $PORT
