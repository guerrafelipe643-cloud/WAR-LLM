import torch
import tiktoken
from datasets import load_dataset

enc = tiktoken.get_encoding("gpt2")

print("Baixando TinyStories...")

train_dataset = load_dataset(
    "roneneldan/TinyStories",
    split="train[:1%]"
)

val_dataset = load_dataset(
    "roneneldan/TinyStories",
    split="validation[:5%]"
)


def tokenize_split(dataset, name):

    print(f"Tokenizando {name}...")

    tokens = []

    for i, example in enumerate(dataset):

        ids = enc.encode(example["text"])

        tokens.extend(ids)

        # Mostra progresso
        if i % 1000 == 0:
            print(
                f"{i}/{len(dataset)} "
                f"| {len(tokens):,} tokens"
            )

    tokens = torch.tensor(
        tokens,
        dtype=torch.int32
    )

    print(
        f"{name}: {tokens.shape[0]:,} tokens"
    )

    return tokens


train_tokens = tokenize_split(
    train_dataset,
    "train"
)

val_tokens = tokenize_split(
    val_dataset,
    "validation"
)


print()
print("Train:", train_tokens.shape)
print("Validation:", val_tokens.shape)

torch.save(
    train_tokens,
    "tinystories_train.pt"
)

torch.save(
    val_tokens,
    "tinystories_val.pt"
)

print("Tudo salvo!")