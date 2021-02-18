# Latest version of the quipp pipeline
FROM turinginst/quipp-env:base

COPY . /quipp-pipeline

WORKDIR /quipp-pipeline

USER root

RUN chown -R 1000 /quipp-pipeline

USER jovyan