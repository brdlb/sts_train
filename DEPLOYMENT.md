# Инструкции по деплою Perudo Web Application

## Варианты деплоя

### 1. Локальный деплой (для разработки)

#### Backend

```bash
# Установите зависимости
pip install -r requirements.txt

# Запустите сервер
python -m src.perudo.web.main

# Или с uvicorn напрямую
uvicorn src.perudo.web.main:app --reload --host 0.0.0.0 --port 8000
```

#### Frontend

```bash
cd frontend
npm install
npm run dev
```

**Доступ:**
- Backend: http://localhost:8000
- Frontend: http://localhost:5173
- API Docs: http://localhost:8000/docs

---

### 2. Production деплой на VPS/сервер

#### Требования
- Python 3.8+
- Node.js 18+
- Nginx (рекомендуется)
- Supervisor или systemd (для управления процессами)

#### Backend

1. **Установите зависимости:**
```bash
pip install -r requirements.txt
pip install gunicorn  # Для production WSGI сервера
```

2. **Создайте файл конфигурации Gunicorn** (`gunicorn_config.py`):
```python
bind = "127.0.0.1:8000"
workers = 4
worker_class = "uvicorn.workers.UvicornWorker"
timeout = 120
keepalive = 5
```

3. **Запустите через Gunicorn:**
```bash
gunicorn src.perudo.web.main:app -c gunicorn_config.py
```

4. **Или используйте systemd service** (`/etc/systemd/system/perudo-api.service`):
```ini
[Unit]
Description=Perudo API Service
After=network.target

[Service]
User=www-data
Group=www-data
WorkingDirectory=/path/to/sts_train
Environment="PATH=/path/to/venv/bin"
ExecStart=/path/to/venv/bin/gunicorn src.perudo.web.main:app -c gunicorn_config.py
Restart=always

[Install]
WantedBy=multi-user.target
```

Активация сервиса:
```bash
sudo systemctl enable perudo-api
sudo systemctl start perudo-api
sudo systemctl status perudo-api
```

#### Frontend

1. **Соберите production build:**
```bash
cd frontend
npm install
npm run build
```

2. **Результат будет в `frontend/dist/`**

3. **Настройте Nginx** (`/etc/nginx/sites-available/perudo`):
```nginx
server {
    listen 80;
    server_name your-domain.com;

    # Frontend
    location / {
        root /path/to/sts_train/frontend/dist;
        try_files $uri $uri/ /index.html;
    }

    # Backend API
    location /api {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    # API Docs
    location /docs {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    # Static files caching
    location ~* \.(js|css|png|jpg|jpeg|gif|ico|svg)$ {
        expires 1y;
        add_header Cache-Control "public, immutable";
    }
}
```

Активация конфигурации:
```bash
sudo ln -s /etc/nginx/sites-available/perudo /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

#### SSL сертификат (Let's Encrypt)

```bash
sudo apt install certbot python3-certbot-nginx
sudo certbot --nginx -d your-domain.com
```

---

### 3. Docker деплой

#### Создайте Dockerfile для Backend (`Dockerfile.backend`):

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Установите системные зависимости
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Копируйте requirements и установите зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install gunicorn

# Копируйте код приложения
COPY . .

# Создайте директории для данных
RUN mkdir -p data models

# Экспортируйте порт
EXPOSE 8000

# Запустите приложение
CMD ["gunicorn", "src.perudo.web.main:app", "--bind", "0.0.0.0:8000", "--workers", "4", "--worker-class", "uvicorn.workers.UvicornWorker"]
```

#### Создайте Dockerfile для Frontend (`frontend/Dockerfile`):

```dockerfile
# Build stage
FROM node:18-alpine as build

WORKDIR /app

COPY package*.json ./
RUN npm ci

COPY . .
RUN npm run build

# Production stage
FROM nginx:alpine

COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
```

#### Создайте nginx конфигурацию для Docker (`frontend/nginx.conf`):

```nginx
server {
    listen 80;
    server_name localhost;
    root /usr/share/nginx/html;
    index index.html;

    location / {
        try_files $uri $uri/ /index.html;
    }

    location /api {
        proxy_pass http://backend:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    location /docs {
        proxy_pass http://backend:8000;
        proxy_set_header Host $host;
    }
}
```

#### Создайте docker-compose.yml:

```yaml
version: '3.8'

services:
  backend:
    build:
      context: .
      dockerfile: Dockerfile.backend
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/models
      - ./data:/app/data
    environment:
      - WEB_HOST=0.0.0.0
      - WEB_PORT=8000
    restart: unless-stopped

  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile
    ports:
      - "80:80"
    depends_on:
      - backend
    restart: unless-stopped
```

#### Запуск:

```bash
# Соберите и запустите контейнеры
docker-compose up -d

# Просмотр логов
docker-compose logs -f

# Остановка
docker-compose down
```

---

### 4. Облачный деплой

#### Railway / Render (Backend)

1. **Создайте файл `Procfile`:**
```
web: gunicorn src.perudo.web.main:app --bind 0.0.0.0:$PORT --workers 4 --worker-class uvicorn.workers.UvicornWorker
```

2. **Создайте `runtime.txt`:**
```
python-3.10.0
```

3. **Подключите репозиторий к Railway/Render**
4. **Настройте переменные окружения:**
   - `WEB_HOST=0.0.0.0`
   - `WEB_PORT` (автоматически устанавливается платформой)

#### Vercel / Netlify (Frontend)

1. **Соберите проект:**
```bash
cd frontend
npm run build
```

2. **Создайте `vercel.json` для Vercel:**
```json
{
  "buildCommand": "cd frontend && npm install && npm run build",
  "outputDirectory": "frontend/dist",
  "rewrites": [
    { "source": "/api/(.*)", "destination": "https://your-backend-url.railway.app/api/$1" },
    { "source": "/(.*)", "destination": "/index.html" }
  ]
}
```

3. **Или создайте `netlify.toml` для Netlify:**
```toml
[build]
  base = "frontend"
  command = "npm install && npm run build"
  publish = "dist"

[[redirects]]
  from = "/api/*"
  to = "https://your-backend-url.railway.app/api/:splat"
  status = 200
  force = true

[[redirects]]
  from = "/*"
  to = "/index.html"
  status = 200
```

4. **Обновите API URL в `frontend/src/services/api.ts`:**
```typescript
const API_BASE_URL = import.meta.env.VITE_API_URL || 'https://your-backend-url.railway.app';
```

---

### 5. Переменные окружения

Создайте файл `.env` для backend:

```env
# Web server
WEB_HOST=0.0.0.0
WEB_PORT=8000
WEB_DEBUG=false

# Database (SQLite)
# Путь к базе данных будет: data/perudo_games.db

# Models
# Путь к моделям берется из config.py
```

Для frontend создайте `.env.production`:

```env
VITE_API_URL=https://your-backend-url.com
```

---

### 6. Проверка деплоя

1. **Проверьте backend:**
```bash
curl http://localhost:8000/health
curl http://localhost:8000/api/models/list
```

2. **Проверьте frontend:**
   - Откройте в браузере
   - Проверьте консоль браузера на ошибки
   - Попробуйте создать игру

3. **Проверьте логи:**
```bash
# Backend (systemd)
sudo journalctl -u perudo-api -f

# Docker
docker-compose logs -f backend
```

---

### 7. Резервное копирование

#### База данных SQLite:

```bash
# Создайте резервную копию
cp data/perudo_games.db data/perudo_games.db.backup

# Или с timestamp
cp data/perudo_games.db "data/perudo_games_$(date +%Y%m%d_%H%M%S).db"
```

#### Автоматическое резервное копирование (cron):

```bash
# Добавьте в crontab (crontab -e)
0 2 * * * cp /path/to/sts_train/data/perudo_games.db /path/to/backups/perudo_games_$(date +\%Y\%m\%d).db
```

---

### 8. Мониторинг и обслуживание

#### Логирование

Backend логирует в stdout/stderr. Для production используйте:
- **systemd journal** (если используется systemd)
- **Docker logs** (если используется Docker)
- **Cloud logging** (если используется облачный деплой)

#### Мониторинг производительности

Рекомендуется использовать:
- **Uptime monitoring**: UptimeRobot, Pingdom
- **Error tracking**: Sentry
- **Performance monitoring**: New Relic, DataDog

---

### 9. Обновление приложения

#### Обновление Backend:

```bash
# Остановите сервис
sudo systemctl stop perudo-api

# Обновите код
git pull

# Установите новые зависимости
pip install -r requirements.txt

# Запустите сервис
sudo systemctl start perudo-api
```

#### Обновление Frontend:

```bash
cd frontend
git pull
npm install
npm run build
# Перезапустите nginx или docker контейнер
```

---

### 10. Troubleshooting

#### Backend не запускается:
- Проверьте, что порт 8000 свободен: `lsof -i :8000`
- Проверьте логи: `sudo journalctl -u perudo-api -n 50`
- Проверьте права доступа к директориям `models/` и `data/`

#### Frontend не подключается к Backend:
- Проверьте CORS настройки в `config.py`
- Проверьте, что API URL правильный в `frontend/src/services/api.ts`
- Проверьте firewall правила

#### Модели не загружаются:
- Убедитесь, что модели находятся в правильной директории
- Проверьте права доступа к файлам моделей
- Проверьте логи backend на ошибки загрузки

---

### Полезные команды

```bash
# Проверка статуса сервисов
sudo systemctl status perudo-api
sudo systemctl status nginx

# Просмотр логов
sudo journalctl -u perudo-api -f
sudo tail -f /var/log/nginx/error.log

# Перезапуск сервисов
sudo systemctl restart perudo-api
sudo systemctl restart nginx

# Проверка портов
netstat -tulpn | grep :8000
netstat -tulpn | grep :80
```

