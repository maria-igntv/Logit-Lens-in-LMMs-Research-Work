from tqdm.notebook import tqdm
from matplotlib.patches import Rectangle
import torch
import torch.nn.functional as F
import gc
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import os


def run_logit_lens_analysis(model, processor, inputs, num_new_tokens=30, save_prefix="results", answer=None):
    '''
    Этот скрипт осуществляет послойную декодировку выходов модели для заданного входа 
    (текст + изображение) и визуализирует:
    - энтропию предсказаний на каждом слое для каждого сгенерированного токена,
    - тепловую карту энтропии по слоям и токенам с аннотацией в виде самих токенов.

    Функция `run_logit_lens_analysis`:
    - принимает модель, процессор и входные данные (inputs),
    - генерирует `num_new_tokens` токенов по одному, записывая:
        - топ-1 предсказания модели на каждом слое (логит линза),
        - энтропию распределения на каждом слое,
    - сохраняет результаты в CSV-файлы (`*_tokens.csv`, `*_entropy.csv` и `*_entropy_num.csv`),
    - сохраняет и отображает график энтропии и PNG тепловую карту.

    Аргументы:
        model (PreTrainedModel): модель типа Qwen2VLForConditionalGeneration
        processor (AutoProcessor): препроцессор от Huggingface (AutoProcessor)
        inputs (dict): входные тензоры, включая изображение и токенизированный текст
        num_new_tokens (int): количество токенов, которые нужно сгенерировать
        save_prefix (str): префикс имени файла для сохранения визуализаций и CSV

    Возвращает:
        df_tokens (pd.DataFrame): предсказанные топ-1 токены по слоям
        df_entropy (pd.DataFrame): энтропия логитов по слоям
'''

    # Создание директории для сохранения результатов 
    output_dir = os.path.join("generated_data", save_prefix)
    os.makedirs(output_dir, exist_ok=True)

    device = next(model.parameters()).device
    generated = inputs["input_ids"].to(device)
    past_key_values = None

    all_layer_token_predictions = []
    all_layer_token_entropies = []
    tokens_per_layer_str = []

    for step in tqdm(range(num_new_tokens), desc="Генерация токенов"):
        with torch.no_grad():
            outputs = model(
                input_ids=generated,
                pixel_values=inputs["pixel_values"].to(device),
                attention_mask=inputs["attention_mask"].to(device),
                image_grid_thw=inputs["image_grid_thw"],
                past_key_values=past_key_values,
                output_hidden_states=True,
                use_cache=True,
                return_dict=True
            )

            hidden_states = outputs.hidden_states
            token_predictions = []
            token_entropies = []

            # Для каждого слоя вычисляем top1 токен и энтропию
            for h in hidden_states:
                logits = model.lm_head(h[:, -1, :])
                top1_id = torch.argmax(logits, dim=-1).item()
                token_predictions.append(top1_id)
                
                probs = F.softmax(logits, dim=-1)
                entropy = -(probs * probs.log()).sum().item()
                token_entropies.append(entropy)
                del logits, probs # очищаем большие тензоры сразу

            all_layer_token_predictions.append(token_predictions)
            all_layer_token_entropies.append(token_entropies)

            # Для DataFrame декодируем токены для читаемости
            row = [f"'{processor.tokenizer.decode([tok_id]).strip()}'" for tok_id in token_predictions]
            tokens_per_layer_str.append(row)

            # Обновляем входы
            next_token_id = torch.tensor([[token_predictions[-1]]], device=device)
            generated = torch.cat([generated, next_token_id], dim=-1)
            past_key_values = outputs.past_key_values

            # Очищаем память
            del outputs, hidden_states
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    tokens_per_layer = list(zip(*all_layer_token_predictions))
    entropy_per_layer = list(zip(*all_layer_token_entropies))
    entropy_per_layer_str = []

    for layer in entropy_per_layer:
        row = [f"'{entropy:.3f}'" for entropy in layer]
        entropy_per_layer_str.append(row)
    
    df_tokens = pd.DataFrame(data=list(zip(*tokens_per_layer_str))[::-1],  
        columns=[f"gen_token_{i}" for i in range(len(tokens_per_layer_str))],  
        index=[f"layer_{i}" for i in range(len(tokens_per_layer_str[0]) - 1, -1, -1)]  
)

    df_entropy = pd.DataFrame(entropy_per_layer[::-1],
                              columns=[f"gen_token_{i}" for i in range(len(entropy_per_layer[0]))],
                              index=[f"layer_{i}" for i in range(len(entropy_per_layer)-1, -1, -1)])


    # Визуализация энтропии
    df_entropy_float = df_entropy.applymap(
        lambda x: float(x.strip("'")) if isinstance(x, str) else float(x)
        )
    # Считаем среднюю энтропию по каждому слою
    avg_entropy_per_layer = df_entropy_float.mean(axis=1)

    plt.figure(figsize=(12, 6))
    plt.plot(avg_entropy_per_layer.index[::-1], avg_entropy_per_layer[::-1], marker='.')
    
    print("Визуализация энтропии")
    plt.xticks(rotation=45)
    plt.xlabel("Слой")
    plt.ylabel("Энтропия")
    plt.title("Средняя энтропия по слоям")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "entropy.png"))
    plt.show()
    
    # Визуализация тепловой карты
    print("Визуализация тепловой карты")
    
    df_entropy_num = pd.DataFrame(entropy_per_layer[::-1],
                              columns=[f"gen_token_{i}" for i in range(len(entropy_per_layer[0]))],
                              index=[f"layer_{i}" for i in range(len(entropy_per_layer)-1, -1, -1)])
    
    warnings.filterwarnings("ignore", category=UserWarning)

    plt.figure(figsize=(40, 20))
    ax = sns.heatmap(df_entropy_num, annot=df_tokens, fmt="", cmap="viridis", cbar=True)
    
    if answer:
        # Токенизируем answer -> финальные токены
        final_token_ids = processor.tokenizer.encode(answer, add_special_tokens=False)
        final_tokens = [processor.tokenizer.decode([tid]).strip() for tid in final_token_ids]
        final_tokens = final_tokens[:df_tokens.shape[1]]

        # Добавим сгенерированный моделью ответ как заголовки
        for idx, token in enumerate(final_tokens):
            ax.text(
                idx + 0.5, -0.5, 
                token,
                ha='center', va='bottom',
                fontsize=10, rotation=45
            )

        # Обводим совпадающие токены
        for col_idx, target_token in enumerate(final_tokens):
            for row_idx in range(df_tokens.shape[0]):
                cell_token = df_tokens.iloc[row_idx, col_idx].strip("'")
                if cell_token == target_token:
                    rect = Rectangle(
                        (col_idx, row_idx), 1, 1,
                        fill=False, edgecolor='red', linewidth=2
                    )
                    ax.add_patch(rect)

        plt.title("Тепловая карта энтропии по слоям для инференса")
    else:
        plt.title("Тепловая карта энтропии по слоям и токенам")

    plt.xlabel("Токены")
    plt.ylabel("Слои")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "heatmap.png"))
    plt.show()

    # Сохраняем в эту папку generated_data
    df_tokens.to_csv(os.path.join(output_dir, "tokens.csv"), index=True)
    df_entropy.to_csv(os.path.join(output_dir, "entropy.csv"), index=True)

    return df_tokens, df_entropy
