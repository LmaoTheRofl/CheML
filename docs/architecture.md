# Архитектура

`chemx parse` создаёт новый `runs/<id>` и записывает manifest. Run output inside `/tmp` запрещён: результаты парсинга должны оставаться в persistent workspace. PyMuPDF извлекает metadata, постраничный текст, layout blocks, таблицы, изображения и page renders. Если доступен `marker_single`, Marker Markdown добавляется в bundle; сбой Marker не блокирует fallback.

После bundle-building в workspace копируются только router skill и выбранный domain skill. Gold-маркеры проверяются до и после backend. `CodexBackend` и `OllamaBackend` различаются только аргументами запуска и реализуют один `Prediction` contract. После успешного inference записываются `prediction.json` и `prediction.csv`.

Evaluator запускается отдельной командой после `inference_complete`. Default gold source — скачанные локально parquet-файлы в `datasets`: evaluator находит parquet рядом с исходным PDF, проверяет имена/порядок/Arrow-типы полей относительно domain contract, извлекает DOI из bundle/PDF и сравнивает строки этой статьи по DOI; если DOI не сопоставился, fallback идёт по `pdf`, затем по нормализованному `title`. Метрика сравнивает мультимножества нормализованных значений независимо для каждого столбца, вычисляет среднее по полям `macro_f1` и записывает `evaluation.json` + `evaluation_metrics.csv`. `evaluate-batch` агрегирует новейший завершённый run каждой статьи в `article_macro_f1.csv`.

Тяжёлые компоненты разделены на extras: Marker, RDKit, Streamlit и gold loader. Из-за несовместимых ограничений Torch MolScribe запускается subprocess-adapter из отдельного окружения через `CHEMX_MOLSCRIBE_COMMAND`. Базовый проект остаётся пригодным для построения bundle и тестирования без GPU.
