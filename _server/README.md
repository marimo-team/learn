---
title: Readme
marimo-version: 0.18.4
---

# marimo learn server

This folder contains server code for hosting marimo apps.

## Running the server

```bash
cd _server
uv run --no-project main.py
```

## Building a Docker image

From the root directory, run:

```bash
docker build -t marimo-learn .
```

## Running the Docker container

```bash
docker run -p 7860:7860 marimo-learn
```