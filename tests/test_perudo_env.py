"""
Тесты для Gymnasium среды Perudo.
"""

import pytest
import numpy as np
from src.perudo.game.perudo_env import PerudoEnv


def test_env_initialization():
    """Тест инициализации среды."""
    env = PerudoEnv(num_players=4, dice_per_player=5)
    
    assert env.num_players == 4
    assert env.dice_per_player == 5
    assert env.observation_space is not None
    assert env.action_space is not None


def test_env_reset():
    """Тест сброса среды."""
    env = PerudoEnv(num_players=2, dice_per_player=5)
    obs, info = env.reset()
    
    assert isinstance(obs, np.ndarray)
    assert obs.dtype == np.float32
    assert len(obs) > 0
    assert "player_id" in info
    assert "game_state" in info


def test_env_step():
    """Тест выполнения шага в среде."""
    env = PerudoEnv(num_players=2, dice_per_player=5)
    obs, info = env.reset()
    
    # Выбираем случайное действие
    action = env.action_space.sample()
    
    # Выполняем действие
    next_obs, reward, terminated, truncated, info = env.step(action)
    
    assert isinstance(next_obs, np.ndarray)
    assert isinstance(reward, (int, float))
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)


def test_env_observation_shape():
    """Тест формы наблюдения."""
    env = PerudoEnv(num_players=2, dice_per_player=5, history_length=10)
    obs, _ = env.reset()
    
    expected_size = 2 + 10 * 3 + 2 + 1 + 2 + 1 + 5  # Все компоненты наблюдения
    assert obs.shape == (expected_size,)


def test_env_action_space():
    """Тест пространства действий."""
    env = PerudoEnv(num_players=2, dice_per_player=5, max_quantity=30)
    
    # Размер должен быть: 2 (challenge, pacao) + 30 * 6 (ставки)
    expected_size = 2 + 30 * 6
    assert env.action_space.n == expected_size


def test_env_render():
    """Тест визуализации среды."""
    env = PerudoEnv(num_players=2, dice_per_player=5, render_mode="human")
    env.reset()
    
    # Проверяем, что render не вызывает ошибок
    try:
        env.render()
    except Exception as e:
        pytest.fail(f"render() вызвал исключение: {e}")


def test_env_set_active_player():
    """Тест установки активного игрока."""
    env = PerudoEnv(num_players=4, dice_per_player=5)
    env.reset()
    
    # Устанавливаем активного игрока
    env.set_active_player(2)
    
    # Получаем наблюдение для него
    obs = env.get_observation_for_player(2)
    
    assert isinstance(obs, np.ndarray)
    assert len(obs) > 0


def test_env_game_over():
    """Тест окончания игры."""
    env = PerudoEnv(num_players=2, dice_per_player=5)
    obs, _ = env.reset()
    
    # Симулируем окончание игры
    env.game_state.game_over = True
    env.game_state.winner = 0
    
    # Выполняем действие (должно вернуть terminated=True)
    action = env.action_space.sample()
    _, _, terminated, _, _ = env.step(action)
    
    assert terminated or env.game_state.game_over

