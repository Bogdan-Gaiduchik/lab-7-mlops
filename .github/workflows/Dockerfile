FROM python:3.13-slim                       # Python 3.13 
WORKDIR /app                                # рабочая директория
COPY requirements.txt .                     # сначала только зависимости (кэш слоёв)
RUN pip install --no-cache-dir -r requirements.txt
COPY . .                                    # копируем весь проект (включая data.csv)
CMD ["pytest", "-v"]                        # по умолчанию — прогон тестов