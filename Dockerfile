FROM python:3.12-slim AS execute

LABEL author="Frederik Labonte"

RUN --mount=target=/var/lib/apt/lists,type=cache,sharing=locked \
    --mount=target=/var/cache/apt,type=cache,sharing=locked \
    apt-get update && \
    apt-get install -y git && \
    apt-get upgrade -y
    
RUN python -m venv /venv
ENV PATH="/venv/bin:$PATH"

RUN --mount=type=bind,src=./,target=/project,readonly \
    --mount=type=cache,target=/root/.cache/ \
    pip install --upgrade pip && \
    pip install /project
    
VOLUME ["/vectore_store"]
EXPOSE 8000
CMD [ "/bin/simple_rag" ]
