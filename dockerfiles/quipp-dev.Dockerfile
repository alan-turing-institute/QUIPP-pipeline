FROM oscartgiles/quipp:latest

COPY . /quipp-pipeline

WORKDIR /quipp-pipeline

USER root

RUN chown -R 1000 /quipp-pipeline

USER jovyan