"""
Простой тест для проверки системы статистики личностей ботов.
"""

import os
import json
import tempfile
from src.perudo.training.bot_personality_tracker import BotPersonalityTracker, BotPersonalityStats


def test_bot_personality_tracker():
    """Тест основной функциональности BotPersonalityTracker."""
    print("="*80)
    print("ТЕСТ СИСТЕМЫ СТАТИСТИКИ ЛИЧНОСТЕЙ БОТОВ")
    print("="*80)
    
    # Создаем временный файл для тестирования
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
        test_file = f.name
    
    try:
        # Создаем трекер
        print("\n1. Создание трекера...")
        tracker = BotPersonalityTracker(stats_file=test_file, elo_k=32)
        print(f"   ✓ Трекер создан с файлом: {test_file}")
        
        # Добавляем несколько тестовых результатов
        print("\n2. Добавление тестовых результатов...")
        
        # Standard Stan выигрывает 3 игры из 10
        for i in range(10):
            won = (i < 3)
            tracker.update_game_result(
                personality_name="STANDARD_STAN",
                won=won,
                rounds_survived=8,
                dice_lost=5 if not won else 0,
                dice_won=0 if not won else 5,
            )
        print(f"   ✓ Добавлено 10 игр для Standard Stan (3 победы)")
        
        # Aggressive Andy выигрывает 5 игр из 10
        for i in range(10):
            won = (i < 5)
            tracker.update_game_result(
                personality_name="AGGRESSIVE",
                won=won,
                rounds_survived=7,
                dice_lost=5 if not won else 0,
                dice_won=0 if not won else 5,
            )
        print(f"   ✓ Добавлено 10 игр для Aggressive Andy (5 побед)")
        
        # Cautious Carl выигрывает 2 игры из 10
        for i in range(10):
            won = (i < 2)
            tracker.update_game_result(
                personality_name="CAUTIOUS",
                won=won,
                rounds_survived=9,
                dice_lost=5 if not won else 0,
                dice_won=0 if not won else 5,
            )
        print(f"   ✓ Добавлено 10 игр для Cautious Carl (2 победы)")
        
        # Проверяем статистику
        print("\n3. Проверка статистики...")
        
        stan_stats = tracker.get_stats("STANDARD_STAN")
        assert stan_stats is not None, "Standard Stan статистика не найдена"
        assert stan_stats.games_played == 10, f"Ожидалось 10 игр, получено {stan_stats.games_played}"
        assert stan_stats.wins == 3, f"Ожидалось 3 победы, получено {stan_stats.wins}"
        assert abs(stan_stats.winrate - 0.3) < 0.01, f"Ожидалось 30% winrate, получено {stan_stats.winrate*100:.1f}%"
        print(f"   ✓ Standard Stan: {stan_stats.games_played} игр, {stan_stats.wins} побед, {stan_stats.winrate*100:.1f}% winrate, ELO {stan_stats.elo:.1f}")
        
        aggressive_stats = tracker.get_stats("AGGRESSIVE")
        assert aggressive_stats is not None, "Aggressive статистика не найдена"
        assert aggressive_stats.games_played == 10, f"Ожидалось 10 игр, получено {aggressive_stats.games_played}"
        assert aggressive_stats.wins == 5, f"Ожидалось 5 побед, получено {aggressive_stats.wins}"
        print(f"   ✓ Aggressive: {aggressive_stats.games_played} игр, {aggressive_stats.wins} побед, {aggressive_stats.winrate*100:.1f}% winrate, ELO {aggressive_stats.elo:.1f}")
        
        cautious_stats = tracker.get_stats("CAUTIOUS")
        assert cautious_stats is not None, "Cautious статистика не найдена"
        assert cautious_stats.games_played == 10, f"Ожидалось 10 игр, получено {cautious_stats.games_played}"
        assert cautious_stats.wins == 2, f"Ожидалось 2 победы, получено {cautious_stats.wins}"
        print(f"   ✓ Cautious: {cautious_stats.games_played} игр, {cautious_stats.wins} побед, {cautious_stats.winrate*100:.1f}% winrate, ELO {cautious_stats.elo:.1f}")
        
        # Проверяем сводку
        print("\n4. Проверка сводки...")
        summary = tracker.get_summary()
        assert summary['total_personalities'] == 3, f"Ожидалось 3 личности, получено {summary['total_personalities']}"
        assert summary['total_games'] == 30, f"Ожидалось 30 игр, получено {summary['total_games']}"
        print(f"   ✓ Всего личностей: {summary['total_personalities']}")
        print(f"   ✓ Всего игр: {summary['total_games']}")
        print(f"   ✓ Лучший winrate: {summary['best_winrate']['name']} ({summary['best_winrate']['winrate']*100:.1f}%)")
        print(f"   ✓ Худший winrate: {summary['worst_winrate']['name']} ({summary['worst_winrate']['winrate']*100:.1f}%)")
        print(f"   ✓ Самый высокий ELO: {summary['highest_elo']['name']} ({summary['highest_elo']['elo']:.1f})")
        
        # Проверяем сохранение в файл
        print("\n5. Проверка сохранения в файл...")
        assert os.path.exists(test_file), f"Файл {test_file} не создан"
        with open(test_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        assert 'rl_agent_elo' in data, "rl_agent_elo отсутствует в файле"
        assert 'personalities' in data, "personalities отсутствует в файле"
        assert len(data['personalities']) == 3, f"Ожидалось 3 личности в файле, получено {len(data['personalities'])}"
        print(f"   ✓ Файл успешно сохранен: {test_file}")
        print(f"   ✓ RL Agent ELO: {data['rl_agent_elo']:.1f}")
        
        # Проверяем загрузку из файла
        print("\n6. Проверка загрузки из файла...")
        new_tracker = BotPersonalityTracker(stats_file=test_file, elo_k=32)
        reloaded_stan = new_tracker.get_stats("STANDARD_STAN")
        assert reloaded_stan is not None, "Standard Stan не загружен из файла"
        assert reloaded_stan.games_played == 10, f"После перезагрузки: ожидалось 10 игр, получено {reloaded_stan.games_played}"
        assert reloaded_stan.wins == 3, f"После перезагрузки: ожидалось 3 победы, получено {reloaded_stan.wins}"
        print(f"   ✓ Статистика успешно загружена из файла")
        
        # Выводим полную сводку
        print("\n7. Полная сводка:")
        print("-"*80)
        tracker.print_summary()
        
        print("\n" + "="*80)
        print("ВСЕ ТЕСТЫ ПРОЙДЕНЫ УСПЕШНО! ✓")
        print("="*80)
        
    finally:
        # Удаляем временный файл
        if os.path.exists(test_file):
            os.remove(test_file)
            print(f"\nВременный файл {test_file} удален")


if __name__ == "__main__":
    test_bot_personality_tracker()

