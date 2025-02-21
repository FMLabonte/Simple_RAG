FROM python:3.13-slim AS execute

LABEL author="Frederik Labonte"

RUN --mount=target=/var/lib/apt/lists,type=cache,sharing=locked \
    --mount=target=/var/cache/apt,type=cache,sharing=locked \
    apt-get update && \
    apt-get upgrade -y

RUN --mount=type=bind,src=./,target=/project,readonly \
    --mount=type=cache,target=/root/.cache/ \
    pip install --upgrade pip && \
    pip install /project
    
VOLUME ["/vectore_store"]
EXPOSE port 8000
CMD simple_rag
