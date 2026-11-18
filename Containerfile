# ==============================================================================
# ШАР 1: Встановлення детермінованого символьного ядра SWI-Prolog
# ==============================================================================
FROM ubuntu:22.04 AS swi-prolog-layer

# Запобігання інтерактивним запитам під час встановлення системних пакетів
ENV DEBIAN_FRONTEND=noninteractive

# Оновлення системних репозиторіїв та встановлення SWI-Prolog разом із базовими утилітами
RUN apt-get update && apt-get install -y --no-install-recommends \
    swi-prolog \
    curl \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Створення стандартизованих директорій для автономного середовища SWI-Prolog, правил та QLF-компіляції
RUN mkdir -p /opt/swipl_standalone/bin \
    && mkdir -p /opt/swipl_standalone/rules

# ==============================================================================
# ШАР 2: Встановлення середовища виконання Python та пакетів із requirements.txt
# ==============================================================================
FROM swi-prolog-layer AS python-environment-layer

# Інсталяція інтерпретатора Python та менеджерів пакетів на базі попереднього шару
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Визначення робочої директорії застосунку в контейнері
WORKDIR /app

# Копіювання файлу залежностей Python із локального хоста
COPY requirements.txt .

# Оновлення pip та встановлення всіх необхідних бібліотек (LangChain, FastAPI, клієнти БД тощо)
RUN pip3 install --no-cache-dir --upgrade pip \
    && pip3 install --no-cache-dir -r requirements.txt

# Копіювання решти вихідного коду проєкту до робочої директорії контейнера
COPY . /app

# Експорт портів для мікросервісної взаємодії (наприклад, для API Gateway)
EXPOSE 8000

# Визначення команди за замовчуванням для запуску головного модуля системи
CMD ["python3", "main.py"]
