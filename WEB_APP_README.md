# Perudo Web Application

Веб-приложение для игры Perudo с поддержкой игры человека против нескольких ИИ-агентов.

## Архитектура

- **Backend**: FastAPI сервер на Python
- **Frontend**: React приложение с TypeScript
- **База данных**: SQLite для хранения истории игр

## Установка

### Backend

1. Установите зависимости:
```bash
pip install -r requirements.txt
```

2. Убедитесь, что у вас есть обученные модели в директории `models/` или `models/opponent_pool/`

### Frontend

1. Перейдите в директорию frontend:
```bash
cd frontend
```

2. Установите зависимости:
```bash
npm install
```

## Запуск

### Backend

Запустите FastAPI сервер:

```bash
python -m src.perudo.web.main
```

Или используя uvicorn напрямую:

```bash
uvicorn src.perudo.web.main:app --reload --host 0.0.0.0 --port 8000
```

Сервер будет доступен по адресу: `http://localhost:8000`

API документация: `http://localhost:8000/docs`

### Frontend

Запустите React приложение:

```bash
cd frontend
npm run dev
```

Приложение будет доступно по адресу: `http://localhost:5173`

## Использование

1. Откройте браузер и перейдите на `http://localhost:5173`
2. Выберите модели для 3 ИИ-агентов (или используйте одну модель для всех)
3. Нажмите "Start Game" для начала игры
4. Делайте ходы: ставки (bid), вызовы (challenge) или pacao (believe)
5. Просматривайте статистику в разделе "Statistics"

## API Endpoints

### Games
- `POST /api/games/create` - Создать новую игру
- `GET /api/games/{game_id}` - Получить состояние игры
- `POST /api/games/{game_id}/action` - Выполнить действие
- `GET /api/games/{game_id}/history` - Получить историю игры
- `GET /api/games` - Список игр

### Models
- `GET /api/models/list` - Список доступных моделей
- `GET /api/models/{model_id}/info` - Информация о модели
- `POST /api/models/validate` - Проверить модель

### Statistics
- `GET /api/statistics/games` - Статистика игр
- `GET /api/statistics/player` - Статистика игрока
- `GET /api/statistics/models` - Статистика по моделям

## Структура проекта

```
src/perudo/web/
├── api/              # API endpoints
├── database/         # База данных модели и операции
├── game_server.py    # Управление игровыми сессиями
├── config.py         # Конфигурация
└── main.py          # FastAPI приложение

frontend/
├── src/
│   ├── components/   # React компоненты
│   ├── services/     # API client
│   └── utils/        # Утилиты
└── package.json
```

## Примечания

- База данных SQLite создается автоматически в директории `data/perudo_games.db`
- Модели должны быть в формате `.zip` (Stable Baselines3)
- Игра поддерживает 4 игроков: 1 человек + 3 ИИ-агента

