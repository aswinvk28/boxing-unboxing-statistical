FROM ubuntu:18.04

ENV HOME /c/Users/Aswin
RUN apt-get update && apt-get install python3 python3-pip -y
WORKDIR $HOME/Documents/Articles/Blogathon/boxing-unboxing
RUN pwd
COPY get-pip.py .
RUN python3 ./get-pip.py
RUN apt-get install openssh-server -y
RUN pip3 install cvxpy imageio numpy scipy scikit-learn matplotlib pandas
RUN pip3 install jupyter tensorflow
RUN pip3 install tensorflow_constrained_optimization

RUN mkdir /home/project
WORKDIR /home/project
COPY boxing-unboxing.ipynb .

ENTRYPOINT "/bin/bash"