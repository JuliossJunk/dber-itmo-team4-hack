# BootCamp TEAM 4 Sber Track
## Appendix
В рамках участия в буткемпе мы работаем над задачей с МАС. Наше решение по [задаче](https://drive.google.com/drive/folders/1VSLjJX7O18SzAPg_ICCAa-WU2JrUKjw5)

## LLM Architecture
Система использует два LLM для оптимального распределения задач:

### Qwen 3 Next 80B (Основная модель)
- **Router Agent** - классификация запросов
- **Simple Agent** - быстрые ответы на простые вопросы
- **Retriever Agent** - поиск и извлечение данных

### DeepSeek (Модель для глубокого рассуждения)
- **Analyzer Agent** - разбивка запроса на подзапросы (multi-hop reasoning)
- **Checker Agent** - верификация фактов через кросс-проверку
- **Counter-Argument Agent** - поиск противоположных точек зрения и критического анализа
- **Synthesizer Agent** - финальный синтез сбалансированного ответа

Такое распределение позволяет использовать сильные стороны каждой модели: Qwen для быстрого поиска и обработки данных, DeepSeek для аналитических задач и рассуждений.

## Architecture
![5237917530821692669](https://github.com/user-attachments/assets/44ecb2b8-ce8e-4137-b16b-6ee8b557b1df)


## How to start
1. Склонируйте проект с гита.
    ``` bash
       git clone https://github.com/JuliossJunk/dber-itmo-team4-hack
    ```
2. Создайте файл `.env` на основе `.env.example` и добавьте ваши API ключи:
    ```bash
    cp .env.example .env
    ```
    Отредактируйте `.env`:
    ```
    MAIN_LLM_KEY=your_main_llm_api_key_here
    DEEPSEEK_API_KEY=your_deepseek_api_key_here
    ```
3. Запустите установку зависимостей:
    ```bash
    poetry install
    ```
4. Запустите сервер чата.
    ```bash
    streamlit run app/app.py
    ```

## FAQ
Профессионально стреляем по мухе из ружья
## Authors
* [Я](https://github.com/JuliossJunk)
* [Demosthen42](https://github.com/Demosthen42)
* [StasGC](https://github.com/StasGC)
* Товарищ
* Коллега
