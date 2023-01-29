ARG NUCLIO_LABEL=1.1.18
ARG NUCLIO_ARCH=amd64
ARG NUCLIO_BASE_IMAGE=python:3.8
ARG NUCLIO_ONBUILD_IMAGE=quay.io/nuclio/handler-builder-python-onbuild:${NUCLIO_LABEL}-${NUCLIO_ARCH}

# Supplies processor uhttpc, used for healthcheck
FROM nuclio/uhttpc:0.0.1-amd64 as uhttpc

# Supplies processor binary, wrapper
FROM ${NUCLIO_ONBUILD_IMAGE} as processor

# From the base image
FROM ${NUCLIO_BASE_IMAGE}

# Copy required objects from the suppliers
COPY --from=processor /home/nuclio/bin/processor /usr/local/bin/processor
COPY --from=processor /home/nuclio/bin/py /opt/nuclio/
COPY --from=uhttpc /home/nuclio/bin/uhttpc /usr/local/bin/uhttpc

RUN pip install nuclio-sdk msgpack --no-index --find-links /opt/nuclio/whl

# Readiness probe
HEALTHCHECK --interval=1s --timeout=3s CMD /usr/local/bin/uhttpc --url http://127.0.0.1:8082/ready || exit 1

# USER CONTENT
ADD ./helloworld.py /opt/nuclio
ADD ./function.yaml /etc/nuclio/config/processor/processor.yaml
# END OF USER CONTENT

# Run processor with configuration and platform configuration
CMD [ "processor" ]

# 安装
# docker build -t docker-hello-func:latest . && docker run --name docker-hello-func -d -p 8090:8080 docker-hello-func:latest

# 卸载
# docker stop docker-hello-func && docker rm docker-hello-func && docker rmi docker-hello-func:latest