FROM python:3.7-slim-buster
COPY . /app
WORKDIR /app
RUN pip install wheel
RUN pip install -r requirements.txt
#EXPOSE 8501
ENV STREAMLIT_SERVER_PORT=$PORT
CMD streamlit run main.py --server.port $PORT

