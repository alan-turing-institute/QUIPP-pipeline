# Docker image with Python3.8 and R 4.#.#

FROM python:3.8

RUN apt update
COPY . /env-configuration

# Install anything in apt.txt and remove files no longer needed (make the image smaller)
RUN apt update && cat /env-configuration/apt.txt | xargs -I % apt install --no-install-recommends % -y && rm -rf /var/lib/apt/lists/*

# Install R 4.0.4
RUN bash /env-configuration/R/R-4.0.4-DebianBuster.sh

# Install python dependencies
RUN pip install -r /env-configuration/requirements.txt

# # Install R dependencies
RUN Rscript env-configuration/install.R

# Install SGF
RUN bash /env-configuration/postBuild && rm -R /env-configuration/

# Create a user
RUN groupadd -r jovyan && useradd --no-log-init -r --create-home -g jovyan -u 1000 jovyan
RUN chown -R 1000 /home/jovyan/
WORKDIR /home/jovyan
USER jovyan