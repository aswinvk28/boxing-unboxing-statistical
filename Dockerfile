FROM ubuntu:18.04

ENV HOME /c/Users/Aswin
RUN apt-get update && apt-get install python3 python3-pip -y
WORKDIR $HOME/Documents/Articles/Blogathon/boxing-unboxing
RUN pwd
COPY get-pip.py .
RUN python3 ./get-pip.py
RUN pip3 install cvxpy imageio numpy scipy scikit-learn matplotlib pandas
