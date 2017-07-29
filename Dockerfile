FROM ubuntu

RUN apt update && apt install python3 \
                              python3-pip \
                              vim \
                              nano \
                              openssh-server \
                              wget \
                              curl \
                              && \
                              apt-get clean && \
                              rm -rf /var/lib/apt/lists/*

RUN pip3 --no-cache-dir install \
        ipykernel \
        jupyter \
        matplotlib \
        numpy \
        scipy \
        sklearn \
        pandas \
        Pillow \
        torchvision \
        bash_kernel \
        && \
python3 -m bash_kernel.install \
        && \
python3 -m ipykernel.kernelspec

COPY jupyter_notebook_config.py /root/.jupyter/

EXPOSE 8888

RUN mkdir /notebooks
WORKDIR "/notebooks"

COPY start-jupyter.sh /
RUN mkdir /examples
COPY mnist.py /examples
RUN chmod +x /start-jupyter.sh

CMD ["/start-jupyter.sh", "--allow-root"]
