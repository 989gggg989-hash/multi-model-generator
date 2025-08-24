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

API уже развернут и доступен по адресу:
🔗 [https://Vlad789-multi-model-api.hf.space](https://Vlad789-multi-model-api.hf.space)

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