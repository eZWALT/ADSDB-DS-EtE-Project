FROM python:3.10-slim-bullseye

WORKDIR /app

COPY . .

#Upgrade packages and python dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    && \
    pip3 install --no-cache-dir -r requirements.txt && \
    apt-get remove -y \
    build-essential \
    && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

#Execute tests before accessing streamlit
RUN PYTHONPATH=. pytest tests/ && echo "All tests passed!"

#Deploy web
ENTRYPOINT ["streamlit", "run", "Home.py"]
