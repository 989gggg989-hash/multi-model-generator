# 🎨 Multi-Model Image Generator

Этот репозиторий содержит:

1. **API** для генерации фото/видео через облачные модели:
   - Flux Schnell (быстрые фото)
   - SDXL (качественные фото)
   - AnimateDiff + SDXL (анимации)

2. **Локальный сайт** для тестирования.

---

## 🚀 Как использовать

### 1. Развертывание API (облако)

1. Создай аккаунт на [Hugging Face](https://huggingface.co/).
2. Создай новый Space:
   - Название: `multi-model-api`
   - SDK: `FastAPI`
   - Hardware: `GPU`
3. Загрузи папку `api/` в репозиторий Space.
4. Дождись запуска (5-10 минут).

### 2. Локальное тестирование

1. Открой `local-site/index.html` в браузере.
2. Введи текст → нажми кнопку.
3. Получи фото/видео с облака!

---

## 📦 Модели

- [Flux Schnell](https://huggingface.co/black-forest-labs/FLUX.1-schnell)
- [SDXL](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)
- [AnimateDiff](https://github.com/guoyww/animatediff)

---

## ⚠️ Важно

- Для Flux Schnell нужно принять условия на Hugging Face.
- Бесплатные GPU на HF ограничены (~400 часов/месяц).