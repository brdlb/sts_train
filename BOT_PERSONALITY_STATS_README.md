# Статистика личностей ботов

Система для отслеживания и анализа статистики различных личностей ботов во время обучения RL агента.

## Описание

Эта система работает аналогично системе opponent pool, но вместо отслеживания снапшотов моделей, она отслеживает статистику игр различных личностей ботов против обучаемого RL агента.

## Отслеживаемая статистика

Для каждой личности бота собирается следующая статистика:

- **Основные показатели:**
  - Количество игр
  - Количество побед/поражений
  - Процент побед (winrate)
  - ELO рейтинг

- **Детальная статистика:**
  - Общее количество раундов выжито
  - Среднее количество раундов выживания за игру
  - Количество потерянных/выигранных кубиков
  - Успешные/неуспешные челленджи
  - Успешные/неуспешные calza
  - Сколько раз бота челленджили
  - Сколько раз бота челленджили правильно

## Файл статистики

Статистика сохраняется в JSON файл: `models/bot_personality_stats.json`

Пример структуры:

```json
{
  "rl_agent_elo": 1523.5,
  "personalities": {
    "Standard Stan": {
      "personality_name": "Standard Stan",
      "games_played": 150,
      "wins": 45,
      "losses": 105,
      "total_rounds": 1200,
      "total_dice_lost": 450,
      "total_dice_won": 0,
      "successful_challenges": 0,
      "failed_challenges": 0,
      "successful_calzas": 0,
      "failed_calzas": 0,
      "times_challenged": 0,
      "times_challenged_correctly": 0,
      "avg_survival_rounds": 8.0,
      "elo": 1487.3
    }
  }
}
```

## Использование

### Активация сбора статистики

Статистика собирается автоматически при использовании `use_bot_opponents=True` в конфигурации обучения.

В `config.py`:

```python
@dataclass
class TrainingConfig:
    # Opponent configuration
    use_bot_opponents: bool = True  # Использовать ботов вместо RL оппонентов
    bot_personalities: Optional[List[str]] = None  # Список личностей (None = все личности)
```

### Выбор конкретных личностей

Чтобы использовать только определенные личности:

```python
from src.perudo.training.config import Config, TrainingConfig

config = Config()
config.training.use_bot_opponents = True
config.training.bot_personalities = [
    "STANDARD_STAN",
    "AGGRESSIVE",
    "CAUTIOUS",
    "CALCULATING"
]

trainer = SelfPlayTraining(config)
trainer.train()
```

### Просмотр статистики во время обучения

Статистика автоматически выводится в конце обучения. Пример вывода:

```
================================================================================
BOT PERSONALITY STATISTICS
================================================================================

RL Agent ELO: 1523.5
Total personalities tracked: 20
Total games played: 3000
Average bot ELO: 1498.2

Best win rate: Aggressive Andy
  Win rate: 35.2% (150 games)

Worst win rate: Cautious Carl
  Win rate: 22.8% (150 games)

Highest ELO: Calculating Clara
  ELO: 1542.3 (150 games)

--------------------------------------------------------------------------------
DETAILED STATISTICS BY PERSONALITY
--------------------------------------------------------------------------------

Calculating Clara:
  Games: 150 | Wins: 48 | Losses: 102
  Win rate: 32.0%
  ELO: 1542.3
  Avg survival rounds: 8.5
  Challenge accuracy: 68.5% (89/130)
  Calza accuracy: 45.2% (14/31)
  Times challenged: 95 | Correctly: 58 (61.1%)

[... остальные личности ...]

================================================================================
```

### Программный доступ к статистике

```python
from src.perudo.training import SelfPlayTraining, BotPersonalityTracker

# Создать трекер
tracker = BotPersonalityTracker(
    stats_file="models/bot_personality_stats.json",
    elo_k=32
)

# Получить статистику конкретной личности
stats = tracker.get_stats("STANDARD_STAN")
if stats:
    print(f"Win rate: {stats.winrate * 100:.1f}%")
    print(f"ELO: {stats.elo:.1f}")
    print(f"Games: {stats.games_played}")

# Получить статистику всех личностей
all_stats = tracker.get_all_stats()
for name, stats in all_stats.items():
    print(f"{name}: {stats.winrate * 100:.1f}% winrate, {stats.elo:.1f} ELO")

# Получить сводку
summary = tracker.get_summary()
print(f"Best performer: {summary['best_winrate']['name']}")
print(f"Highest ELO: {summary['highest_elo']['name']}")

# Вывести полную сводку в консоль
tracker.print_summary()
```

## Интеграция с обучением

Система автоматически интегрирована в процесс обучения:

1. При создании `SelfPlayTraining` с `use_bot_opponents=True` создается `BotPersonalityTracker`
2. После каждой игры статистика автоматически обновляется для всех участвовавших ботов
3. ELO рейтинги обновляются после каждой игры
4. Статистика сохраняется в JSON файл после каждого обновления
5. В конце обучения выводится полная сводка

## ELO система

Каждая личность бота имеет свой ELO рейтинг, который обновляется после каждой игры:

- Начальный ELO: 1500
- K-фактор: 32 (настраивается)
- RL агент также имеет свой ELO рейтинг
- Рейтинги обновляются по стандартной формуле ELO

Это позволяет:
- Сравнивать силу различных личностей
- Отслеживать прогресс RL агента относительно ботов
- Понимать, какие типы игры наиболее сложны для RL агента

## Расширение статистики

В будущем можно добавить отслеживание:
- Детальной статистики по челленджам и calza для каждого бота
- Истории изменения ELO со временем
- Статистики по различным стадиям игры (early/mid/late game)
- Взаимодействия между конкретными личностями

## Пример запуска тренировки с ботами

```bash
# Запуск с настройками по умолчанию (все боты)
python -m src.perudo.training.train

# Статистика будет сохраняться в models/bot_personality_stats.json
# В конце обучения будет выведена полная сводка
```

## Примечания

- Статистика накапливается между запусками обучения
- Чтобы сбросить статистику, удалите файл `models/bot_personality_stats.json`
- Программно сбросить статистику: `tracker.reset_stats()` или `tracker.reset_stats("PERSONALITY_NAME")`
- Файл создается автоматически при первом запуске обучения с ботами

