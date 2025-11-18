# Решения проблемы MaskableCategorical Simplex Constraint Error

## Описание проблемы

Ошибка возникает когда вероятности в `MaskableCategorical` не удовлетворяют ограничению `Simplex()` (т.е. не суммируются до 1.0). Это может происходить из-за:

1. **Численных ошибок** при нормализации вероятностей после применения масок действий
2. **Проблем с маскированием** - все действия замаскированы или очень мало валидных действий
3. **Проблем с logits** - экстремальные значения logits приводят к численной нестабильности при softmax

## Найденные решения

### 1. Проверка масок действий (уже реализовано)

В коде уже есть обработка проблемы с масками в `perudo_vec_env.py`:
- Проверка, что хотя бы одно действие валидно
- Fallback на challenge/believe если все действия замаскированы
- Валидация типов и размеров масок

**Расположение:** `src/perudo/game/perudo_vec_env.py:1492-1634`

### 2. Рекомендуемые решения

#### Решение A: Добавить температурное масштабирование в policy_kwargs

Добавить параметр `temperature` для стабилизации вычислений вероятностей:

```python
# В src/perudo/training/train.py, при создании policy_kwargs
policy_kwargs = dict(
    features_extractor_class=TransformerFeaturesExtractor,
    features_extractor_kwargs=dict(...),
    net_arch=[dict(pi=[96, 64], vf=[128, 64])],
    # Добавить для стабилизации вероятностей
    action_distribution_kwargs=dict(
        temperature=1.0,  # Можно попробовать значения 0.8-1.2
    ),
)
```

**Примечание:** Это может не работать напрямую, так как `sb3_contrib` может не поддерживать этот параметр.

#### Решение B: Добавить clamp для logits перед softmax

Модифицировать политику для ограничения значений logits:

```python
# Создать кастомную политику, которая ограничивает logits
# Это требует изменения исходного кода sb3_contrib или создания wrapper
```

#### Решение C: Увеличить entropy_coef для стабилизации

Увеличить коэффициент энтропии для более равномерного распределения вероятностей:

```python
# В config.py
ent_coef: float = 0.02  # Увеличить с 0.01 до 0.02-0.05
```

Это поможет избежать экстремальных вероятностей.

#### Решение D: Добавить проверку и нормализацию вероятностей

Создать callback, который проверяет и исправляет проблемные батчи:

```python
# В src/perudo/training/callbacks.py добавить новый callback
class ProbabilityStabilityCallback(BaseCallback):
    """Callback для проверки и исправления проблем с вероятностями."""
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.problem_count = 0
    
    def on_step(self) -> bool:
        # Проверка вероятностей в политике
        if hasattr(self.model.policy, 'action_dist'):
            # Логирование проблемных случаев
            pass
        return True
```

#### Решение E: Обновить sb3_contrib до последней версии

Убедиться, что используется последняя версия `sb3_contrib` (2.7.0), которая может содержать исправления:

```bash
pip install --upgrade sb3-contrib
```

### 3. Немедленные действия для диагностики

1. **Проверить логи** - найти конкретные случаи, когда возникает ошибка:
   - Какие маски действий были в этот момент?
   - Сколько валидных действий было?
   - Какие значения logits?

2. **Добавить дополнительное логирование** в `action_masks()`:
   ```python
   # В perudo_vec_env.py, метод action_masks()
   logger.debug(f"Env {i}: {num_valid} valid actions out of {self.action_space.n}")
   logger.debug(f"Mask sum: {action_mask.sum()}, dtype: {action_mask.dtype}")
   ```

3. **Проверить версии библиотек**:
   ```bash
   pip show sb3-contrib stable-baselines3 torch
   ```

### 4. Долгосрочные решения

1. **Создать issue на GitHub** в репозитории `stable-baselines3-contrib`:
   - Описать проблему
   - Приложить минимальный воспроизводимый пример
   - Указать версии библиотек

2. **Рассмотреть альтернативные подходы**:
   - Использовать обычный PPO с ручной обработкой невалидных действий
   - Реализовать кастомную политику с более стабильной нормализацией

3. **Мониторинг во время обучения**:
   - Добавить проверки вероятностей в callback
   - Автоматически пропускать проблемные батчи
   - Сохранять чекпоинты перед проблемными шагами

## Рекомендуемый порядок действий

1. ✅ Убедиться, что `sb3_contrib` обновлен до 2.7.0
2. ✅ Проверить, что маски действий всегда содержат хотя бы одно валидное действие
3. ⚠️ Попробовать увеличить `ent_coef` до 0.02-0.05
4. ⚠️ Добавить дополнительное логирование для диагностики
5. ⚠️ Если проблема сохраняется, создать issue на GitHub

## Ссылки

- [Stable-Baselines3 Contrib GitHub](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib)
- [MaskablePPO Documentation](https://sb3-contrib.readthedocs.io/en/latest/modules/ppo_mask.html)
- [PyTorch Categorical Distribution](https://pytorch.org/docs/stable/distributions.html#categorical)


