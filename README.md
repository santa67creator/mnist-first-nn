# MNIST: моя первая нейронная сеть

Полносвязная сеть (784 → 500 → 250 → 10) на TensorFlow для распознавания
рукописных цифр MNIST. Написана по главе 2 книги про TensorFlow, адаптирована
под современный TensorFlow через `tf.compat.v1`.

**Результат:** ~98.3% точности на тесте после 20 эпох (CPU).

## Запуск
1. Скачай MNIST (4 файла `.gz`) в `~/AI/datasets/mnist`
2. `python -m venv venv && source venv/bin/activate`
3. `pip install -r requirements.txt`
4. `python main.py`
