# RNN project: news classification, explained line by line

[Study index](README.md) · [Original notebook](RNN_NewsClassifier.ipynb) · [RNN foundations](RNN/README.md)

## What the project does

The notebook downloads AG News data, combines each article's title and description, converts the text into token IDs, and trains a vanilla recurrent neural network to predict a category. The output categories are World, Sports, Business, and Sci/Tech.

The model is a classifier. Given text about an event, it assigns a topic label; it does not verify whether the event happened or generate an article. The short Messi examples near the end probe the classifier's response to wording, including the misspelling `retured`.

## Before running

The code requires kagglehub, pandas, scikit-learn, and PyTorch. The download must contain `train.csv` and `test.csv` with `Title`, `Description`, and `Class Index` columns. The source assumes class indices 1–4 and converts them to zero-based indices. Inspect the CSV columns and values before training against another export.

The DataLoaders use two worker processes. For local Jupyter/Windows, zero workers is a useful fallback. The standalone `import torch` appears in cell 28, after the Dataset class definition. This works when cells run top-to-bottom because `torch.tensor` is called later, when the loader is iterated; fetching a Dataset item before cell 28 would fail. Moving the import to the initial import cell makes that dependency clearer.

## Text becomes a sequence of learned vectors

```text
"Microsoft announces NEW AI products!"
→ ["microsoft", "announces", "new", "ai", "products"]
→ five vocabulary IDs
→ append PAD IDs until length 80
→ remember real length = 5
→ look up a 128-value embedding for each ID
→ pack the real tokens, excluding trailing padding
→ RNN final hidden vector with 128 values
→ four raw class scores
→ predicted category
```

The numeric IDs depend on frequencies in the loaded training CSV. They are not hard-coded word meanings. `<PAD>` is zero; `<UNK>` is one. The vocabulary includes at most 30,000 entries including those two reserved entries.

## Architecture and shape reference

`B` is the batch size, usually 128; `T=80` is the padded sequence length; `V` is the actual vocabulary size.

| Stage | Shape | Meaning |
|---|---|---|
| Token IDs | `[B,80]`, long | One integer per token/pad |
| Real lengths | `[B]`, long on CPU for packing | Number of retained real tokens |
| Embedding | `[B,80,128]`, float | Learned vector per token |
| Packed representation data | `[sum(lengths),128]` | Real steps plus batch-order metadata |
| Packed RNN output data | `[sum(lengths),128]` | Hidden responses for real steps |
| Final hidden | `[1,B,128]` | One layer and one direction |
| `hidden[-1]` | `[B,128]` | One article representation |
| Linear head | `[B,4]` | Four class logits |
| Argmax | `[B]` | One predicted category ID |

Packing matters because an ordinary RNN updates its hidden state even when its input embedding is zero. Without packing, reading many padding steps can change the final state. `padding_idx=0` controls the embedding row; packing controls which timesteps the RNN processes. They solve different problems.

## What the RNN learns

At each real token, `h_t = tanh(W_ih e_t + b_ih + W_hh h_(t-1) + b_hh)`. The same weights are reused across token positions. No initial state is supplied, so each forward call begins from zero state. The model does not carry article memory across batches.

The [RNN API](https://docs.pytorch.org/docs/stable/generated/torch.nn.RNN.html) describes its sequence output and final-state conventions. Here `batch_first=True` does not move the layer axis of the final hidden tensor. `hidden[-1]` selects the last recurrent layer, which is also the only layer.

Parameter count is **`128V + 33,540`**: `128V` embedding values, `33,024` recurrent values, and `516` output-head values. At V=30,000 that is 3,873,540 parameters. The padding row is present in the embedding tensor but receives no normal embedding gradient. Most of the model's storage is in the vocabulary embedding.

Adam trains the embedding, RNN, and output head with cross-entropy at learning rate 0.001 for five epochs. Four equal logits would imply an illustrative cross-entropy of `log(4) ≈ 1.386`; this is not a measured notebook result.

## Notebook walkthrough

Cell numbers below count all notebook cells from one, including Markdown cells. Line numbers restart in each code block and count its blank lines. Every nonblank line has an explanation; blank lines only provide spacing. The original code is reproduced unchanged, so notebook limitations described later also apply to these blocks.

### Cell 1: Install kagglehub

```python
!pip -q install kagglehub
```

| Line | Explanation |
|---|---|
| L1 | Runs a notebook shell command to install kagglehub; `-q` reduces installation output. This is notebook syntax, not ordinary Python-script syntax. |

### Cell 2: Download the news dataset

```python
import kagglehub

path = kagglehub.dataset_download(
    "amananandrai/ag-news-classification-dataset"
)

print(path)
```

| Line | Explanation |
|---|---|
| L1 | Imports the client used by the next cell to download the project dataset. |
| L3 | Downloads or locates a cached dataset and assigns its returned local directory to `path`. |
| L4 | Selects the AG News dataset used by this project; kagglehub returns its local cache directory. |
| L5 | Closes the multiline call, collection, or grouped expression above; it adds no separate numerical operation. |
| L7 | Displays the returned dataset directory so later file paths can be checked. |

### Cell 3: List downloaded files

```python
import os

for root, dirs, files in os.walk(path):
    for file in files:
        print(os.path.join(root, file))
```

| Line | Explanation |
|---|---|
| L1 | Imports filesystem utilities used to inspect the downloaded directory. |
| L3 | Traverses the dataset directory recursively. Each iteration gives the current directory, its subdirectories, and its filenames. |
| L4 | Iterates filenames in each visited directory, helping identify the CSV locations. |
| L5 | Joins the current directory and filename into a complete path and prints it. |

### Cell 4: Read training and test CSV files

```python
import pandas as pd

train_df = pd.read_csv(os.path.join(path, "train.csv"))
test_df = pd.read_csv(os.path.join(path, "test.csv"))

print(train_df.shape)
print(test_df.shape)

train_df.head()
```

| Line | Explanation |
|---|---|
| L1 | Imports pandas as `pd` for reading and manipulating tabular data. |
| L3 | Reads the training CSV from the downloaded directory into a DataFrame. |
| L4 | Reads the separate test CSV. The source creates a Dataset for it later but never evaluates a test loader. |
| L6 | Displays training row and column counts. |
| L7 | Displays test row and column counts. |
| L9 | Shows initial training rows so text and label columns can be inspected. |

### Cell 5: Check column names

```python
print(train_df.columns)
```

| Line | Explanation |
|---|---|
| L1 | Displays exact CSV column names; later expressions require `Title`, `Description`, and `Class Index`. |

### Cell 7: Combine title and description

`astype(str)` is not missing-value cleaning: a missing entry can become the literal string `nan`. If missing text is present, fill it deliberately before concatenation.

```python
train_df["text"] = (
    train_df["Title"].astype(str)
    + " "
    + train_df["Description"].astype(str)
)

test_df["text"] = (
    test_df["Title"].astype(str)
    + " "
    + test_df["Description"].astype(str)
)
```

| Line | Explanation |
|---|---|
| L1 | Begins assigning a combined text column in the training DataFrame. |
| L2 | Converts each title to a string so concatenation has a consistent type. |
| L3 | Inserts a space between title and description so boundary words do not merge. |
| L4 | Converts descriptions to strings and appends them to the corresponding titles. |
| L5 | Closes the multiline call, collection, or grouped expression above; it adds no separate numerical operation. |
| L7 | Builds the same combined text representation in the test DataFrame. |
| L8 | Reads and string-converts test titles. |
| L9 | Inserts the same separating space used for training text. |
| L10 | Appends string-converted test descriptions. |
| L11 | Closes the multiline call, collection, or grouped expression above; it adds no separate numerical operation. |

### Cell 9: Define the tokenizer

```python
import re

def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())
```

| Line | Explanation |
|---|---|
| L1 | Imports Python regular expressions. |
| L3 | Defines one text-to-token-list function reused by vocabulary building and prediction. |
| L4 | Lowercases the text and extracts runs of word characters bounded as words. Punctuation separators disappear; digits, underscores, and Unicode word characters can remain. |

### Cell 10: Try tokenization on one sentence

```python
tokenize("Microsoft announces NEW AI products!")
```

| Line | Explanation |
|---|---|
| L1 | Demonstrates case normalization and punctuation removal; expected tokens are microsoft, announces, new, ai, products. |

### Cell 12: Count word frequencies

```python
from collections import Counter

counter = Counter()

for text in train_df["text"]:
    counter.update(tokenize(text))
```

| Line | Explanation |
|---|---|
| L1 | Imports a frequency-counting dictionary specialized for repeated items. |
| L3 | Creates an initially empty token-frequency counter. |
| L5 | Visits every combined article in the original training CSV, before its later validation split. |
| L6 | Tokenizes the article and increments each token's count. This counter determines vocabulary membership/order. |

### Cell 13: Inspect frequent words

```python
counter.most_common(20)
```

| Line | Explanation |
|---|---|
| L1 | Displays the twenty most frequent tokens and their counts; it does not change the vocabulary or model. |

### Cell 14: Build a capped vocabulary

```python
MAX_VOCAB = 30000

word2idx = {
    "<PAD>": 0,
    "<UNK>": 1
}

for word, count in counter.most_common(MAX_VOCAB - 2):
    word2idx[word] = len(word2idx)

VOCAB_SIZE = len(word2idx)

print("Vocabulary:", VOCAB_SIZE)
```

| Line | Explanation |
|---|---|
| L1 | Limits the total vocabulary to at most 30,000 entries, including reserved tokens. |
| L3 | Begins a token-to-integer dictionary. |
| L4 | Reserves ID zero for padding positions used to make examples equally long. |
| L5 | Reserves ID one for words absent from the retained vocabulary. |
| L6 | Closes the multiline call, collection, or grouped expression above; it adds no separate numerical operation. |
| L8 | Takes the most frequent 29,998 ordinary tokens, leaving two slots for PAD and UNK. `count` is unpacked but not otherwise used. |
| L9 | Assigns each new word the next consecutive ID, based on the dictionary's current size. |
| L11 | Records the actual vocabulary length, which can be below the cap for a small dataset. |
| L13 | Displays the embedding/output vocabulary size used when constructing the model. |

### Cell 17: Encode and pad an article

For a five-token article, this returns five real IDs followed by 75 zeros and length 5. For 100 tokens, it returns the first 80 IDs and length 80. For empty/punctuation-only input, length becomes zero, which the packing operation later rejects.

```python
MAX_LEN = 80

def encode_text(text):
    tokens = tokenize(text)

    ids = [
        word2idx.get(token, word2idx["<UNK>"])
        for token in tokens
    ]

    ids = ids[:MAX_LEN]

    length = len(ids)

    ids += [word2idx["<PAD>"]] * (MAX_LEN - len(ids))

    return ids, length
```

| Line | Explanation |
|---|---|
| L1 | Sets the maximum retained article length to 80 tokens. |
| L3 | Defines an encoder that returns both padded IDs and the real retained length. |
| L4 | Uses the shared tokenizer so train and inference text follow the same rule. |
| L6 | Begins a list comprehension converting tokens into IDs. |
| L7 | Retrieves a token's ID or substitutes UNK when it is absent from the vocabulary. |
| L8 | Repeats that lookup for every token in order. |
| L9 | Closes the multiline call, collection, or grouped expression above; it adds no separate numerical operation. |
| L11 | Keeps only the first 80 token IDs; information later in a long article is discarded. |
| L13 | Measures the truncated real sequence length before adding padding. |
| L15 | Appends enough PAD zeros on the right to make exactly 80 IDs. |
| L17 | Returns the fixed-width sequence and its real token count. |

### Cell 19: Define the Dataset interface

```python
from torch.utils.data import Dataset

class NewsDataset(Dataset):

    def __init__(self, dataframe):
        self.df = dataframe.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        text = self.df.loc[idx, "text"]

        ids, length = encode_text(text)

        label = int(self.df.loc[idx, "Class Index"]) - 1

        return (
            torch.tensor(ids, dtype=torch.long),
            torch.tensor(length, dtype=torch.long),
            torch.tensor(label, dtype=torch.long)
        )
```

| Line | Explanation |
|---|---|
| L1 | Imports the base interface implemented by the custom Dataset class below. |
| L3 | Declares a Dataset whose examples contain IDs, real length, and a class label. |
| L5 | Defines construction from a pandas DataFrame containing text and class columns. |
| L6 | Resets row labels to consecutive integers so `.loc[idx]` agrees with Dataset indexing after the split; `drop=True` avoids keeping the old index as a column. |
| L8 | Defines the length queried by DataLoader and sampling utilities. |
| L9 | Returns the number of article rows. |
| L11 | Defines how to fetch one example by its integer index. |
| L13 | Retrieves the combined article text for the requested row. |
| L15 | Encodes it as 80 padded IDs and a real length. |
| L17 | Converts the source labels 1–4 into model target IDs 0–3, matching the four output columns. |
| L19 | Begins returning a three-part example tuple. |
| L20 | Creates long token IDs with shape `[80]`, suitable for embedding lookup. |
| L21 | Creates a scalar long real-length tensor; batching turns these scalars into `[B]`. |
| L22 | Creates a scalar long target ID for cross-entropy. |
| L23 | Closes the multiline call, collection, or grouped expression above; it adds no separate numerical operation. |

### Cell 21: Split training rows into train and validation

Vocabulary counting already used all rows of this DataFrame. Thus the validation text influenced vocabulary selection, even though its labels are not used by the optimizer. For a strict evaluation, split first and build the vocabulary only from `train_part`.

```python

from sklearn.model_selection import train_test_split

train_part, val_part = train_test_split(
    train_df,
    test_size=0.1,
    random_state=42,
    stratify=train_df["Class Index"]
)
```

| Line | Explanation |
|---|---|
| L2 | Imports the helper that splits examples into separate training and validation subsets. |
| L4 | Begins randomly partitioning the original training DataFrame. |
| L5 | Supplies the original training rows; the external test CSV remains separate. |
| L6 | Reserves 10% of these rows for validation, leaving 90% for fitting model weights. |
| L7 | Makes this split reproducible with a fixed scikit-learn random seed. |
| L8 | Approximately preserves each class's proportion in both subsets. |
| L9 | Closes the multiline call, collection, or grouped expression above; it adds no separate numerical operation. |

### Cell 23: Create Datasets and loaders

```python
from torch.utils.data import DataLoader

train_dataset = NewsDataset(train_part)
val_dataset = NewsDataset(val_part)
test_dataset = NewsDataset(test_df)

train_loader = DataLoader(
    train_dataset,
    batch_size=128,
    shuffle=True,
    num_workers=2,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=128,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)
```

| Line | Explanation |
|---|---|
| L1 | Imports the loader that combines Dataset examples into mini-batches. |
| L3 | Wraps the training subset in the custom text Dataset. |
| L4 | Wraps the validation subset with the same encoding pipeline. |
| L5 | Wraps the external test CSV; no corresponding test_loader is constructed in the original notebook. |
| L7 | Begins building the training mini-batch loader. |
| L8 | Selects training examples as its source. |
| L9 | Combines up to 128 examples per batch; the last batch may contain fewer. |
| L10 | Randomizes the order of examples each time the training loader is traversed; the order within each example is preserved. |
| L11 | Uses two worker processes to prepare examples. If local notebook multiprocessing fails, begin with zero workers. |
| L12 | Requests pinned host memory where supported, which can help transfers to CUDA. This does not place the batch on the GPU. |
| L13 | Closes the multiline call, collection, or grouped expression above; it adds no separate numerical operation. |
| L15 | Begins building the validation loader. |
| L16 | Selects validation examples as its source. |
| L17 | Combines up to 128 examples per batch; the last batch may contain fewer. |
| L18 | Preserves Dataset order, useful for repeatable evaluation and aligned plots. |
| L19 | Uses two worker processes to prepare examples. If local notebook multiprocessing fails, begin with zero workers. |
| L20 | Requests pinned host memory where supported, which can help transfers to CUDA. This does not place the batch on the GPU. |
| L21 | Closes the multiline call, collection, or grouped expression above; it adds no separate numerical operation. |

### Cell 25: Define the embedding, packed RNN, and classifier

```python
import torch.nn as nn

class NewsRNN(nn.Module):

    def __init__(
        self,
        vocab_size,
        embedding_dim=128,
        hidden_size=128,
        num_classes=4
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=0
        )

        self.rnn = nn.RNN(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            batch_first=True
        )

        self.fc = nn.Linear(
            hidden_size,
            num_classes
        )

    def forward(self, x, lengths):

        embedded = self.embedding(x)

        packed = nn.utils.rnn.pack_padded_sequence(
            embedded,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )

        output, hidden = self.rnn(packed)

        final_hidden = hidden[-1]

        logits = self.fc(final_hidden)

        return logits
```

| Line | Explanation |
|---|---|
| L1 | Imports neural-network layers and losses under the short name `nn`. |
| L3 | Declares a neural-network module for four-class news prediction. |
| L5 | Starts the constructor signature so dimensions can be configured. |
| L6 | Refers to the model instance receiving the layers. |
| L7 | Requires the vocabulary size used for valid embedding IDs. |
| L8 | Defaults each token embedding to 128 learned values. |
| L9 | Defaults the recurrent hidden vector to 128 values. |
| L10 | Defaults the classifier to four categories. |
| L11 | Closes the multiline definition header and begins its indented body. |
| L12 | Initializes nn.Module bookkeeping before assigning child layers and their registered parameters. |
| L14 | Creates the learned lookup table for tokens. |
| L15 | Allocates one embedding row per vocabulary entry. |
| L16 | Allocates 128 embedding features per row by default. |
| L17 | Marks row zero as padding; its normal embedding gradient is suppressed. This alone does not skip RNN timesteps. |
| L18 | Closes the multiline call, collection, or grouped expression above; it adds no separate numerical operation. |
| L20 | Creates a vanilla RNN; default settings give one layer, one direction, and tanh activation. |
| L21 | Makes each timestep consume an embedding vector of the selected width. |
| L22 | Sets the size of the recurrent state. |
| L23 | Uses `[batch, time, features]` for sequence input/output; hidden and cell state layouts retain their layer axis first. |
| L24 | Closes the multiline call, collection, or grouped expression above; it adds no separate numerical operation. |
| L26 | Creates the affine classification head. |
| L27 | Takes the hidden vector's 128 features by default. |
| L28 | Produces one logit per news category. |
| L29 | Closes the multiline call, collection, or grouped expression above; it adds no separate numerical operation. |
| L31 | Defines a forward pass that requires both token IDs and their real sequence lengths. |
| L33 | Looks up every token/pad embedding, producing `[B,80,128]` under default dimensions. |
| L35 | Packs the embedded right-padded sequences so trailing pad steps are not processed as article content. |
| L36 | Supplies the padded floating embedding tensor to packing. |
| L37 | Moves the length tensor to CPU as required when lengths are supplied as a tensor. |
| L38 | Declares `[batch, time, features]` layout for this sequence operation. |
| L39 | Allows examples in arbitrary length order; the utility handles sorting internally. |
| L40 | Closes the multiline call, collection, or grouped expression above; it adds no separate numerical operation. |
| L42 | Runs the RNN on the packed real tokens; `output` remains packed and `hidden` is the final state `[1,B,128]`. |
| L44 | Selects the last layer's final state, giving one `[128]` representation per article; it is not the last article in the batch. |
| L46 | Maps article representations to four raw logits each. |
| L48 | Returns logits for loss calculation or category prediction. |

### Cell 28: Instantiate the model on the selected device

```python
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = NewsRNN(
    vocab_size=VOCAB_SIZE
).to(device)

print(model)
```

| Line | Explanation |
|---|---|
| L1 | Imports PyTorch; tensors, device selection, and gradient-control functions use this name. |
| L3 | Creates a CUDA device when available, otherwise CPU. Actual tensor/model movement happens in `.to(device)` calls. |
| L5 | Begins constructing a fresh NewsRNN with default embedding, hidden, and output dimensions. |
| L6 | Uses the actual size of the vocabulary, including reserved tokens. |
| L7 | Finishes construction and moves all registered model state to the selected device. |
| L9 | Prints the registered architecture, useful for checking layer widths and configuration. |

### Cell 29: Display the selected device

```python
print(device)
```

| Line | Explanation |
|---|---|
| L1 | Displays which device was selected; this does not move or train the model. |

### Cell 31: Create cross-entropy and Adam

```python
import torch.optim as optim

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)
```

| Line | Explanation |
|---|---|
| L1 | Imports optimizer constructors under the `optim` alias. |
| L3 | Creates multiclass cross-entropy: raw logits `[B,C]` are compared with long class IDs `[B]`. |
| L5 | Constructs Adam using the `optim` namespace imported in this cell. |
| L6 | Supplies the model's registered parameters, including learned embeddings or feature layers where present, to the optimizer. |
| L7 | Sets the initial learning rate to 0.001. |
| L8 | Closes the multiline call, collection, or grouped expression above; it adds no separate numerical operation. |

### Cell 33: Train for five epochs

Lengths remain on CPU in the training loop, which is appropriate for packing. The current one-layer NewsRNN has no Dropout or BatchNorm, but explicit training mode keeps the loop correct if such layers are added later.

```python
EPOCHS = 5

for epoch in range(EPOCHS):

    model.train()

    total_loss = 0
    correct = 0
    total = 0

    for x, lengths, y in train_loader:

        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        logits = model(x, lengths)

        loss = criterion(logits, y)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

        predicted = logits.argmax(dim=1)

        correct += (predicted == y).sum().item()
        total += y.size(0)

    train_accuracy = correct / total

    print(
        f"Epoch {epoch+1}/{EPOCHS} | "
        f"Loss: {total_loss / len(train_loader):.4f} | "
        f"Accuracy: {train_accuracy:.4f}"
    )
```

| Line | Explanation |
|---|---|
| L1 | Requests five complete traversals of the training loader. |
| L3 | Repeats training with epoch indices starting at zero and ending at `EPOCHS - 1`. |
| L5 | Selects training behavior. Dropout and BatchNorm, if present, use their training rules; this call does not itself compute gradients. |
| L7 | Resets the accumulated sum of batch-mean losses for this epoch. |
| L8 | Resets the count of examples whose predicted class matches the target. |
| L9 | Resets the count of evaluated examples. |
| L11 | Unpacks each batch into padded token IDs, real lengths, and class targets. |
| L13 | Moves token IDs to the device containing the embedding and recurrent model. |
| L14 | Moves class targets to the model device for loss and comparisons. |
| L16 | Clears old parameter gradients so this batch does not accidentally accumulate gradients from the previous batch. |
| L18 | Forwards the IDs and lengths; packing ignores trailing pad positions. |
| L20 | Computes mean cross-entropy between logits `[B,4]` and targets `[B]`. |
| L22 | Backpropagates from the scalar loss and fills parameter gradients. It does not change parameter values yet. |
| L24 | Uses the newly computed gradients and Adam's state to update trainable parameter values. |
| L26 | Adds this batch's mean loss to the epoch sum, without retaining its computation graph. |
| L28 | Selects the highest-scoring class ID separately for each example. |
| L30 | Compares predictions and labels elementwise, counts matches, and adds the Python count. |
| L31 | Adds the actual batch size, so a short final batch is counted correctly. |
| L33 | Computes the fraction of articles classified correctly during this training traversal. |
| L35 | Begins printing a readable progress or metric message assembled by the following lines. |
| L36 | Formats the current one-based epoch and the total number of epochs. |
| L37 | Displays the equal average of batch losses to four decimal places. |
| L38 | Displays training accuracy as a fraction, not a percentage; 0.85 means 85%. |
| L39 | Closes the multiline call, collection, or grouped expression above; it adds no separate numerical operation. |

### Cell 36: Define a reusable accuracy evaluator

```python
def evaluate(model, loader):

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for x, lengths, y in loader:

            x = x.to(device)
            y = y.to(device)

            logits = model(x, lengths)

            predictions = logits.argmax(dim=1)

            correct += (predictions == y).sum().item()
            total += y.size(0)

    return correct / total
```

| Line | Explanation |
|---|---|
| L1 | Defines a function that evaluates a supplied model on a supplied loader using the global selected device. |
| L3 | Selects evaluation behavior for mode-sensitive layers. Gradient tracking is controlled separately. |
| L5 | Resets the count of examples whose predicted class matches the target. |
| L6 | Resets the count of evaluated examples. |
| L8 | Disables graph recording for the indented inference block, reducing work and memory. |
| L10 | Iterates whichever labeled news loader the caller provides. |
| L12 | Moves token IDs to the device containing the embedding and recurrent model. |
| L13 | Moves class targets to the model device for loss and comparisons. |
| L15 | Computes category logits using each article's real length. |
| L17 | Selects the highest-scoring class ID separately for each example. |
| L19 | Counts matching class IDs and adds this batch's correct predictions. |
| L20 | Adds the actual number of target labels in the current batch. |
| L22 | Returns total correct divided by total evaluated examples; an empty loader would require a separate guard. |

### Cell 37: Measure validation accuracy

```python
val_accuracy = evaluate(model, val_loader)

print(f"Validation accuracy: {val_accuracy:.4f}")
```

| Line | Explanation |
|---|---|
| L1 | Evaluates the final trained weights on the held-out validation subset. |
| L3 | Prints the returned fraction to four decimal places; it is not a test-set score. |

### Cell 38: Define category names and predict a new headline

```python
idx2label = {
    0: "World",
    1: "Sports",
    2: "Business",
    3: "Sci/Tech"
}

def predict_news(text):
    model.eval()
    with torch.no_grad():
        ids, length = encode_text(text)
        input_tensor = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)
        length_tensor = torch.tensor([length], dtype=torch.long).to(device)

        logits = model(input_tensor, length_tensor)
        prediction = logits.argmax(dim=1).item()

    return idx2label[prediction]

predict_news("Apple launches a new AI-powered chip")
```

| Line | Explanation |
|---|---|
| L1 | Begins the reverse mapping from zero-based model IDs to readable category names. |
| L2 | Maps ID zero to World, corresponding to original CSV class index one. |
| L3 | Maps ID one to Sports, corresponding to original CSV class index two. |
| L4 | Maps ID two to Business, corresponding to original CSV class index three. |
| L5 | Maps ID three to Sci/Tech, corresponding to original CSV class index four. |
| L6 | Closes the multiline call, collection, or grouped expression above; it adds no separate numerical operation. |
| L8 | Defines a helper that classifies a supplied string with the global model and encoder. |
| L9 | Selects evaluation behavior for mode-sensitive layers. Gradient tracking is controlled separately. |
| L10 | Disables graph recording for the indented inference block, reducing work and memory. |
| L11 | Applies the same tokenization, vocabulary lookup, truncation, and padding used for training. |
| L12 | Creates long IDs, adds a batch axis to obtain `[1,80]`, and moves them to the model device. |
| L13 | Creates a one-element length tensor and moves it to the model device; forward immediately copies it back to CPU for packing, so that move is unnecessary. |
| L15 | Produces one row of four logits for the new text. |
| L16 | Selects its maximum-score class and converts the one-element result into a Python integer. |
| L18 | Returns the readable category name corresponding to that integer. |
| L20 | Runs a sample technology-related headline through the helper. Its actual output depends on learned weights. |

### Cell 39: Probe a football headline

```python
predict_news("Messi retured from football")
```

| Line | Explanation |
|---|---|
| L1 | Classifies this exact short string. The misspelling `retured` is preserved and may map to UNK; the output tests learned associations, not factual truth. |

### Cell 40: Change the context word to acting

```python
predict_news("Messi retured from acting")
```

| Line | Explanation |
|---|---|
| L1 | Changes one part of the previous sentence while retaining Messi and the misspelling. Compare outputs to investigate which cues dominate. |

### Cell 41: Change the context word to study

```python
predict_news("Messi retured from study")
```

| Line | Explanation |
|---|---|
| L1 | Tests another wording variation; the four-category classifier must still choose one of its known classes. |

### Cell 42: Change the context word to market

```python
predict_news("Messi retured from market")
```

| Line | Explanation |
|---|---|
| L1 | Tests whether business-related wording changes the category despite the same named person. No particular predicted label is guaranteed. |

### Cell 43: Empty final cell

This cell is empty. It performs no operation.

## Limitations that affect the reported result

**There is no test accuracy in the source workflow.** It reads `test.csv` and constructs `test_dataset`, but creates only training and validation loaders. Cell 37 reports validation accuracy. The following is an additional completion step after decisions about the model are finished:

```python
# Additional example; uses the existing test_dataset and evaluate function.
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=0)
test_accuracy = evaluate(model, test_loader)
print(f"Test accuracy: {test_accuracy:.4f}")
```

The first line batches the separate test articles in order, using the main process for local notebook compatibility. The second computes accuracy using the existing evaluator. The third displays that measured fraction. This snippet has not been run here and is not part of the original notebook.

**Vocabulary construction sees validation text.** Build the split before counting frequencies if you want validation preprocessing to be independent. Then count only `train_part['text']`; use the resulting vocabulary unchanged for validation, test, and inference. Merely rebuilding the vocabulary after training changes embedding IDs and invalidates the learned lookup table.

**Empty input breaks packing.** Blank or punctuation-only text returns length zero. A deliberate policy could reject such input with a helpful message, or substitute one UNK token with real length one. Apply any chosen rule consistently before training and inference. Do not simply clamp length to one while leaving an all-PAD input and pretend a real token was present.

**Length and tokenization discard information.** Articles are cut after 80 tokens; missing words map to UNK; punctuation and case are removed. A headline typed at prediction time is also shorter and may have a different style from the title-plus-description training examples. Inspect mistakes under these conditions before interpreting confidence.

## Reading the training numbers

The printed training accuracy is based on predictions made throughout changing weights in an epoch. It is a fraction, whereas the CNN project prints percentages. The printed loss is an average of batch means; sample weighting is needed for an exact mean when the final batch is short.

Only the split has a fixed random seed. The notebook does not seed PyTorch initialization or DataLoader shuffling. It validates only after all five epochs, rather than tracking validation loss/accuracy each epoch. It does not implement gradient clipping or save the best checkpoint. These are potential extensions, not behaviors already present.

For a vanishing/exploding-gradient investigation, first inspect losses and gradient magnitudes. Clipping, if used, belongs after `loss.backward()` and before `optimizer.step()`. LSTM or GRU cells change the recurrent model and should be compared using the same split and evaluation procedure.

## Reusing this model later

Save the learned weights together with `word2idx`, `MAX_LEN`, the category mapping, tokenizer rules, and architecture sizes. An embedding matrix without the matching word-ID dictionary cannot reproduce the intended predictions. The current notebook does not save these artifacts.

## Troubleshooting and check-your-understanding

| Question or symptom | Answer |
|---|---|
| Why are labels decremented by one? | Four output columns are indexed 0–3, while the source CSV labels are 1–4 |
| Why does packing require lengths measured before padding? | Otherwise the RNN would treat trailing pads as real content |
| Does `hidden[-1]` mean the last article? | No; it selects the last recurrent layer along the first state axis |
| Why is `output` unused? | This many-to-one classifier only needs each sequence's final hidden state |
| Why does punctuation-only input fail? | Its token count is zero, which packed sequences do not accept |
| Why might all Messi examples get the same label? | A strong named-entity association may outweigh the changed word; inspect predictions without assuming robust sentence understanding |
| Can validation accuracy be called test accuracy? | No; the supplied loader and role of the split determine which metric it is |
