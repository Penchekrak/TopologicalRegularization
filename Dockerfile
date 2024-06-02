FROM cr.ai.cloud.ru/aicloud-jupyter/jupyter-server:0.0.94
RUN pip install -e git+https://github.com/c-hofer/torchph.git#egg=torchph