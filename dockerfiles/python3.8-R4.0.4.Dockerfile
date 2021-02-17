# Docker image with Python3.8 and R 4.#.#

FROM python:3.8

RUN apt update
COPY . /env-configuration

# Install anything in apt.txt
RUN cat /env-configuration/apt.txt | xargs -I % apt install % -y

# Install R 4.0.4
RUN bash /env-configuration/R/R-4.0.4-DebianBuster.sh

# Install python dependencies
RUN pip install -r /env-configuration/requirements.txt

# Install R dependencies
RUN Rscript env-configuration/install.R

# Install SGF
RUN bash /env-configuration/postBuild
