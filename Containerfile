FROM ubuntu:26.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    swi-prolog \
    python3 \
    python3.14-venv \
    python3-dev \
    build-essential \
    curl \
    git \
    && mkdir -p /usr/lib/swi-prolog/lib/x86_64-linux \
    && ln -sf /usr/lib/x86_64-linux-gnu/libswipl.so \
              /usr/lib/swi-prolog/lib/x86_64-linux/libswipl.so \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv "$VIRTUAL_ENV"
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

ENV CFLAGS="-std=gnu17"
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["sh", "-c", "exec uvicorn \"${APP_MODULE:-src.main:app}\" --host 0.0.0.0 --port 8000"]
