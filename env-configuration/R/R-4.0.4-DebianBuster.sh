#!/bin/bash

# Install R 4.#.# on Debian Buster

apt update -y
apt install software-properties-common -y
apt install apt-transport-https -y
apt-key adv --keyserver keys.gnupg.net --recv-key 'E19F5F87128899B192B1A2C2AD5F960A256A04AF'
echo 'deb http://cloud.r-project.org/bin/linux/debian buster-cran40/' >> /etc/apt/sources.list
apt update
apt install -t buster-cran40 r-base -y