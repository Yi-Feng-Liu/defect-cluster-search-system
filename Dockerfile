FROM python:3.10.9
COPY apt.conf /etc/apt/apt.conf
COPY pip.conf /etc/pip.conf
COPY requirements.txt requirements.txt

RUN pip install -U pip && pip install -r requirements.txt

# Installing Oracle instant client
WORKDIR /opt/oracle
RUN set -eux && apt-get update
RUN apt-get update && apt-get install -y libaio1 wget unzip \
    && wget --no-check-certificate https://download.oracle.com/otn_software/linux/instantclient/instantclient-basiclite-linuxx64.zip \
    && unzip instantclient-basiclite-linuxx64.zip \
    && rm -f instantclient-basiclite-linuxx64.zip \
    && cd /opt/oracle/instantclient* \
    && rm -f *jdbc* *occi* *mysql* *README *jar uidrvci genezi adrci \
    && echo /opt/oracle/instantclient* > /etc/ld.so.conf.d/oracle-instantclient.conf \
    && ldconfig


WORKDIR /defect_cluster_search_engine

COPY . .


EXPOSE 8502

# CMD bash

ENTRYPOINT ["streamlit", "run", "interface.py", "--server.port=8502", "--server.fileWatcherType", "none"]
