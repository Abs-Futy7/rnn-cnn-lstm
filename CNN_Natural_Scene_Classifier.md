# CNN project: natural-scene classification, explained line by line

[Study index](README.md) · [Original notebook](CNN_Natural_Scene_Classifier.ipynb) · [CNN foundations](CNN/README.md)

## What the project does

The notebook trains a convolutional network to assign an RGB image to one of six natural-scene categories. It downloads an image dataset, creates training and test loaders, learns from labeled images, evaluates accuracy and a confusion matrix, and predicts a class for an uploaded image.

The six folder labels are expected to be buildings, forest, glacier, mountain, sea, and street. `ImageFolder` determines the actual names and numeric order from directory names. Always inspect `train_data.classes` and verify train/test `class_to_idx` agreement rather than assuming an order from memory.

## Before running

The notebook uses kagglehub, PyTorch, torchvision, Pillow, scikit-learn, and matplotlib. Dataset download requires network access and enough disk space. It assumes the downloaded directory contains `seg_train/seg_train` and `seg_test/seg_test`. Cell 3 helps inspect that layout.

The last inference cells use Google Colab's upload dialog. In local Jupyter, replace the upload cell and the expression based on `uploaded` with a local image path. The existing [mountian.jpg](mountian.jpg) is a possible local input when the working directory is `Pytorch`; its filename spelling is preserved. No new training run or accuracy measurement was performed for this guide.

## The complete tensor journey

`B` is the batch size, normally 64. One RGB pixel has three channel values; a channel is not a class.

| Stage | Shape | What changes |
|---|---|---|
| Loaded image | Height × width × 3 conceptually | PIL RGB representation |
| Resize, tensor conversion, normalization | `[3,128,128]` | Fixed size, CHW, floating values |
| Batch | `[B,3,128,128]` | Adds example axis |
| Conv 3→32, BN, ReLU | `[B,32,128,128]` | Learns 32 maps |
| Pool | `[B,32,64,64]` | Halves spatial size |
| Conv 32→64, BN, ReLU, pool | `[B,64,32,32]` | More channels, smaller maps |
| Conv 64→128, BN, ReLU, pool | `[B,128,16,16]` | 128 learned feature maps |
| Adaptive average pool | `[B,128,1,1]` | One average response per channel |
| Flatten and Dropout | `[B,128]` | One feature vector per image |
| Linear 128→6 | `[B,6]` | Six raw class scores |
| Argmax over classes | `[B]` | One predicted class ID per image |

## Why this architecture works

A 3×3 convolution with stride one and padding one preserves height and width. Each filter combines a local neighborhood across all input channels. The same filter weights are reused at all image positions. Three pooling operations reduce 128 to 64, 32, and 16.

BatchNorm learns channel-wise scale/shift and maintains statistics used at evaluation. ReLU discards negative activations. Adaptive average pooling summarizes each final channel over its 16×16 map, leaving 128 numbers. It discards fine spatial layout in favor of aggregate feature evidence. Dropout regularizes those numbers during training.

The model has **94,470 trainable values**: convolution parameters `896 + 18,496 + 73,856`, BatchNorm parameters `64 + 128 + 256`, and the final Linear layer `774`. Running BatchNorm statistics are additional buffers, not trainable values. These counts are derived from the source architecture, not from a training run.

## Loss and learning

For logits `z` and true class k, cross-entropy is `-log(exp(z_k)/sum_j exp(z_j))`. The model returns logits directly; it does not put softmax before the loss. The [cross-entropy API](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html) documents this contract. If all six logits were equal, the illustrative loss would be `log(6) ≈ 1.792`.

Adam updates the parameters using gradients. `weight_decay=1e-4` regularizes the parameters included in its group. Ten epochs means ten traversals of the training loader. This source has no validation split or checkpoint selection; test accuracy is evaluated after the final epoch.

## Notebook walkthrough

Cell numbers below count all notebook cells from one, including Markdown cells. Line numbers restart in each code block and count its blank lines. Every nonblank line has an explanation; blank lines only provide spacing. The original code is reproduced unchanged, so notebook limitations described later also apply to these blocks.

### Cell 1: Install the dataset client

```python
!pip -q install kagglehub
```

| Line | Explanation |
|---|---|
| L1 | Runs a notebook shell command to install kagglehub; `-q` reduces installation output. This is notebook syntax, not ordinary Python-script syntax. |

### Cell 2: Download the image dataset

```python
import kagglehub

path = kagglehub.dataset_download(
    "puneet6060/intel-image-classification"
)

print(path)
```

| Line | Explanation |
|---|---|
| L1 | Imports the client used by the next cell to download the project dataset. |
| L3 | Downloads or locates a cached dataset and assigns its returned local directory to `path`. |
| L4 | Identifies the Kaggle dataset requested by this project; the return value is its local cache path. |
| L5 | Closes the multiline call, collection, or grouped expression above; it adds no separate numerical operation. |
| L7 | Displays the returned dataset directory so later file paths can be checked. |

### Cell 3: Inspect the directory tree

```python
import os

for root, dirs, files in os.walk(path):

    level = root.replace(path, "").count(os.sep)

    if level <= 2:
        print(root)
```

| Line | Explanation |
|---|---|
| L1 | Imports filesystem utilities used to inspect the downloaded directory. |
| L3 | Traverses the dataset directory recursively. Each iteration gives the current directory, its subdirectories, and its filenames. |
| L5 | Estimates nesting depth by removing the base path text and counting path separators. |
| L7 | Limits printed paths to shallow levels so the output is readable; traversal itself still visits deeper directories. |
| L8 | Prints each selected directory path to help locate the train and test folders. |

### Cell 4: Construct training and test paths

```python
from pathlib import Path

train_dir = Path(path) / "seg_train" / "seg_train"
test_dir = Path(path) / "seg_test" / "seg_test"

print(train_dir)
print(test_dir)
```

| Line | Explanation |
|---|---|
| L1 | Imports the cross-platform Path class, whose `/` operator joins path components. |
| L3 | Builds the expected labeled training-image directory under the downloaded path. |
| L4 | Builds the expected labeled test-image directory; it is separate from training. |
| L6 | Displays the resolved training path for inspection. |
| L7 | Displays the resolved test path for inspection. |

### Cell 6: Define image preprocessing and augmentation

For a pixel value 0, normalization gives -1; for 0.5, it gives 0; for 1, it gives 1. Only the training pipeline contains a random flip.

```python
from torchvision import transforms

train_transform = transforms.Compose([

    transforms.Resize((128, 128)),

    transforms.RandomHorizontalFlip(),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])

test_transform = transforms.Compose([

    transforms.Resize((128, 128)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])
```

| Line | Explanation |
|---|---|
| L1 | Imports torchvision's image transforms. |
| L3 | Creates an ordered training transformation pipeline. |
| L5 | Resizes every training image to height 128 and width 128, allowing equal-sized batches. |
| L7 | Randomly flips images left/right with the transform's default probability of 0.5, adding training variation. |
| L9 | Converts ordinary uint8 PIL images into floating CHW tensors and scales values from 0–255 to 0–1. |
| L11 | Starts channel-wise normalization after tensor conversion. |
| L12 | Subtracts 0.5 separately from red, green, and blue channels. |
| L13 | Divides each channel by 0.5, so the preceding 0–1 range maps to approximately -1–1. |
| L14 | Closes the multiline call, collection, or grouped expression above; it adds no separate numerical operation. |
| L15 | Closes the multiline call, collection, or grouped expression above; it adds no separate numerical operation. |
| L17 | Creates a separate deterministic transformation pipeline for evaluation and prediction. |
| L19 | Resizes evaluation images to the same 128×128 size used for training. |
| L21 | Converts evaluation images to CHW float tensors with values in 0–1 before normalization. |
| L23 | Applies the same normalization used during training. |
| L24 | Uses the identical three channel means of 0.5. |
| L25 | Uses the identical three channel standard deviations of 0.5. |
| L26 | Closes the multiline call, collection, or grouped expression above; it adds no separate numerical operation. |
| L27 | Closes the multiline call, collection, or grouped expression above; it adds no separate numerical operation. |

### Cell 8: Create labeled image datasets

```python
from torchvision.datasets import ImageFolder

train_data = ImageFolder(
    train_dir,
    transform=train_transform
)

test_data = ImageFolder(
    test_dir,
    transform=test_transform
)

print(train_data.classes)
```

| Line | Explanation |
|---|---|
| L1 | Imports ImageFolder, which derives labels from class subdirectories. |
| L3 | Begins constructing the training Dataset; images are loaded when requested. |
| L4 | Points ImageFolder at the training folder whose immediate subdirectories represent classes. |
| L5 | Applies the training transform whenever a training image is retrieved. |
| L6 | Closes the multiline call, collection, or grouped expression above; it adds no separate numerical operation. |
| L8 | Constructs the held-out image Dataset using the same folder-label convention. |
| L9 | Points it to the separate test-image directory. |
| L10 | Applies deterministic evaluation preprocessing to each test image. |
| L11 | Closes the multiline call, collection, or grouped expression above; it adds no separate numerical operation. |
| L13 | Displays class names in the numeric order used by this training Dataset. |

### Cell 10: Create mini-batch loaders

```python
from torch.utils.data import DataLoader

train_loader = DataLoader(
    train_data,
    batch_size=64,
    shuffle=True,
    num_workers=2,
    pin_memory=True
)

test_loader = DataLoader(
    test_data,
    batch_size=64,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)
```

| Line | Explanation |
|---|---|
| L1 | Imports the loader that combines Dataset examples into mini-batches. |
| L3 | Creates the loader that will iterate training examples in batches. |
| L4 | Supplies the training Dataset, including its augmentation. |
| L5 | Combines up to 64 examples per batch; the final batch can be smaller. |
| L6 | Randomizes the order of examples each time the training loader is traversed; the order within each example is preserved. |
| L7 | Uses two worker processes to prepare examples. If local notebook multiprocessing fails, begin with zero workers. |
| L8 | Requests pinned host memory where supported, which can help transfers to CUDA. This does not place the batch on the GPU. |
| L9 | Closes the multiline call, collection, or grouped expression above; it adds no separate numerical operation. |
| L11 | Creates the loader used for held-out evaluation. |
| L12 | Supplies the test Dataset, whose preprocessing has no random flip. |
| L13 | Combines up to 64 examples per batch; the final batch can be smaller. |
| L14 | Preserves Dataset order, useful for repeatable evaluation and aligned plots. |
| L15 | Uses two worker processes to prepare examples. If local notebook multiprocessing fails, begin with zero workers. |
| L16 | Requests pinned host memory where supported, which can help transfers to CUDA. This does not place the batch on the GPU. |
| L17 | Closes the multiline call, collection, or grouped expression above; it adds no separate numerical operation. |

### Cell 12: Inspect one batch

```python
images, labels = next(iter(train_loader))

print(images.shape)
print(labels.shape)
```

| Line | Explanation |
|---|---|
| L1 | Creates an iterator and retrieves its first batch, unpacking image tensors and labels. This is only an inspection; it does not train the model. |
| L3 | Displays image shape, expected `[64,3,128,128]` for a full first batch. |
| L4 | Displays label shape, expected `[64]` for that batch. |

### Cell 14: Define SceneCNN

```python
import torch
import torch.nn as nn

class SceneCNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(

            nn.Conv2d(
                3,
                32,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(
                64,
                128,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Dropout(0.3),

            nn.Linear(
                128,
                6
            )
        )

    def forward(self, x):

        x = self.features(x)

        x = self.classifier(x)

        return x
```

| Line | Explanation |
|---|---|
| L1 | Imports PyTorch; tensors, device selection, and gradient-control functions use this name. |
| L2 | Imports neural-network layers and losses under the short name `nn`. |
| L4 | Declares a model class inheriting PyTorch's parameter registration and model utilities. |
| L6 | Defines initialization for a new SceneCNN instance. |
| L7 | Initializes nn.Module bookkeeping before assigning child layers and their registered parameters. |
| L9 | Begins the ordered convolutional feature extractor. |
| L11 | Creates the first two-dimensional convolution. |
| L12 | Sets three input channels: red, green, and blue. |
| L13 | Learns 32 output feature maps. |
| L14 | Uses a 3×3 spatial filter in each output channel. |
| L15 | Pads by one pixel on every spatial side, preserving 128×128 with the default stride one. |
| L16 | Closes the multiline call, collection, or grouped expression above; it adds no separate numerical operation. |
| L18 | Normalizes 32 channels and learns their scale/shift; evaluation uses running statistics. |
| L19 | Applies ReLU to zero negative responses while preserving shape. |
| L21 | Takes the maximum in 2×2 regions with default stride two: 128×128 becomes 64×64. |
| L23 | Creates the second convolution, which combines the first block's feature maps. |
| L24 | Receives 32 input channels from the first block. |
| L25 | Produces 64 learned output channels. |
| L26 | Uses 3×3 filters across all 32 input channels. |
| L27 | Preserves the 64×64 spatial size before pooling. |
| L28 | Closes the multiline call, collection, or grouped expression above; it adds no separate numerical operation. |
| L30 | Normalizes the second convolution's 64 output channels. |
| L31 | Applies the second block's elementwise nonlinearity. |
| L33 | Downsamples spatial size from 64×64 to 32×32. |
| L35 | Creates the third convolution for more abstract learned image features. |
| L36 | Receives 64 input channels. |
| L37 | Produces 128 output channels. |
| L38 | Uses 3×3 spatial kernels. |
| L39 | Preserves the 32×32 size before the third pooling operation. |
| L40 | Closes the multiline call, collection, or grouped expression above; it adds no separate numerical operation. |
| L42 | Normalizes the 128 output channels. |
| L43 | Zeros negative activations in the third block. |
| L45 | Downsamples 32×32 to 16×16. |
| L47 | Averages each channel's entire map into one value, producing `[B,128,1,1]`. |
| L48 | Closes the multiline call, collection, or grouped expression above; it adds no separate numerical operation. |
| L50 | Begins the classifier operating on the pooled feature representation. |
| L52 | Flattens all non-batch axes, converting `[B,128,1,1]` to `[B,128]`. |
| L54 | Randomly zeros activations with probability 0.3 during training; becomes an identity operation at evaluation. |
| L56 | Creates the final learned affine transformation. |
| L57 | Consumes the 128 pooled features per image. |
| L58 | Produces six raw scores, matching this project's six class folders. |
| L59 | Closes the multiline call, collection, or grouped expression above; it adds no separate numerical operation. |
| L60 | Closes the multiline call, collection, or grouped expression above; it adds no separate numerical operation. |
| L62 | Defines how an input batch flows through the model when `model(x)` is called. |
| L64 | Runs all convolution, normalization, activation, and pooling layers on the input. |
| L66 | Runs flattening, dropout, and the final classifier to obtain `[B,6]` logits. |
| L68 | Returns raw scores to the training loss or inference caller. |

### Cell 15: Select a device and instantiate the network

```python
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = SceneCNN().to(device)

print(model)
```

| Line | Explanation |
|---|---|
| L1 | Begins selecting the device on which model parameters and input tensors will be stored. |
| L2 | Chooses CUDA when PyTorch can use it; otherwise selects CPU. |
| L3 | Closes the multiline call, collection, or grouped expression above; it adds no separate numerical operation. |
| L5 | Constructs a fresh SceneCNN and moves its registered parameters and buffers to the selected device. |
| L7 | Prints the registered architecture, useful for checking layer widths and configuration. |

### Cell 16: Configure loss and optimizer

```python
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001,
    weight_decay=1e-4
)
```

| Line | Explanation |
|---|---|
| L1 | Creates multiclass cross-entropy: raw logits `[B,C]` are compared with long class IDs `[B]`. |
| L3 | Constructs Adam, an optimizer that maintains running gradient statistics for parameter updates. |
| L4 | Supplies the model's registered parameters, including learned embeddings or feature layers where present, to the optimizer. |
| L5 | Sets the initial learning rate to 0.001; this controls the overall scale of Adam updates. |
| L6 | Adds weight decay of 0.0001 to this Adam parameter group; this is separate from activation Dropout. |
| L7 | Closes the multiline call, collection, or grouped expression above; it adds no separate numerical operation. |

### Cell 17: Train the network for ten epochs

The accuracy collected here comes from changing model weights and training-mode behavior during the epoch. It is not a separate final evaluation of the fixed model on all training images.

```python
EPOCHS = 10

for epoch in range(EPOCHS):

    model.train()

    running_loss = 0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(
            device,
            non_blocking=True
        )

        labels = labels.to(
            device,
            non_blocking=True
        )

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        predictions = outputs.argmax(dim=1)

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

    accuracy = correct / total

    print(
        f"Epoch {epoch+1}/{EPOCHS} | "
        f"Loss: {running_loss/len(train_loader):.4f} | "
        f"Train Accuracy: {accuracy*100:.2f}%"
    )
```

| Line | Explanation |
|---|---|
| L1 | Requests ten complete traversals of the training loader. |
| L3 | Repeats training with epoch indices starting at zero and ending at `EPOCHS - 1`. |
| L5 | Selects training behavior. Dropout and BatchNorm, if present, use their training rules; this call does not itself compute gradients. |
| L7 | Resets the accumulated sum of batch-mean losses for this epoch. |
| L8 | Resets the count of examples whose predicted class matches the target. |
| L9 | Resets the count of evaluated examples. |
| L11 | Retrieves each training mini-batch, including its newly applied random image augmentation. |
| L13 | Begins transferring the image batch to the computation device. |
| L14 | Uses the same device selected for model parameters. |
| L15 | Requests a nonblocking transfer where supported; pinned host memory can help CUDA transfers. |
| L16 | Closes the multiline call, collection, or grouped expression above; it adds no separate numerical operation. |
| L18 | Begins transferring class IDs to the computation device. |
| L19 | Uses the model's selected device for labels as well. |
| L20 | Requests a nonblocking label transfer where supported. |
| L21 | Closes the multiline call, collection, or grouped expression above; it adds no separate numerical operation. |
| L23 | Clears old parameter gradients so this batch does not accidentally accumulate gradients from the previous batch. |
| L25 | Produces logits `[B,6]` using current parameters, before this batch's update. |
| L27 | Starts calculating one mean cross-entropy value for this batch. |
| L28 | Supplies the raw six-class logits. |
| L29 | Supplies matching long class IDs `[B]`. |
| L30 | Closes the multiline call, collection, or grouped expression above; it adds no separate numerical operation. |
| L32 | Backpropagates from the scalar loss and fills parameter gradients. It does not change parameter values yet. |
| L34 | Uses the newly computed gradients and Adam's state to update trainable parameter values. |
| L36 | Adds this batch's mean loss as a Python number. It weights batches equally, including a potentially smaller final batch. |
| L38 | Selects the most highly scored class in each image's output row. |
| L40 | Begins adding the number of correctly classified images in the batch. |
| L41 | Produces a Boolean match per image by comparing predicted IDs with labels. |
| L42 | Sums True values and extracts a Python count. |
| L44 | Adds the actual number of images in this batch, including a short final batch. |
| L46 | Computes accuracy over individual examples as a fraction between zero and one. |
| L48 | Begins printing a readable progress or metric message assembled by the following lines. |
| L49 | Formats a one-based epoch number together with the requested epoch count. |
| L50 | Reports an equal average of batch-mean losses to four decimal places. |
| L51 | Converts the training accuracy fraction to a percentage with two decimal places. |
| L52 | Closes the multiline call, collection, or grouped expression above; it adds no separate numerical operation. |

### Cell 19: Evaluate held-out images

```python
model.eval()

correct = 0
total = 0

all_predictions = []
all_labels = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        predictions = outputs.argmax(dim=1)

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_labels.extend(
            labels.cpu().numpy()
        )

accuracy = correct / total

print(
    f"Test Accuracy: {accuracy*100:.2f}%"
)
```

| Line | Explanation |
|---|---|
| L1 | Selects evaluation behavior for mode-sensitive layers. Gradient tracking is controlled separately. |
| L3 | Resets the count of examples whose predicted class matches the target. |
| L4 | Resets the count of evaluated examples. |
| L6 | Creates a list to retain every predicted class for the confusion matrix. |
| L7 | Creates a parallel list of actual class labels. |
| L9 | Disables graph recording for the indented inference block, reducing work and memory. |
| L11 | Iterates the ordered test batches without training augmentation. |
| L13 | Moves the image tensor to the same device as the model. |
| L14 | Moves target IDs to the model device for loss or comparison operations. |
| L16 | Runs the fixed trained model in evaluation mode and obtains six logits per image. |
| L18 | Chooses the index of the highest class score in each row, giving a long tensor `[B]`. |
| L20 | Begins accumulating this batch's number of correct predictions. |
| L21 | Compares predicted and actual class IDs elementwise. |
| L22 | Converts the sum of matches to a Python integer. |
| L24 | Adds the actual number of images in this batch, including a short final batch. |
| L26 | Appends each predicted ID from this batch to the accumulated list. |
| L27 | Moves predictions to CPU and converts them to NumPy values for later sklearn use. |
| L28 | Closes the multiline call, collection, or grouped expression above; it adds no separate numerical operation. |
| L30 | Appends corresponding true labels in the same order. |
| L31 | Moves true labels to CPU and converts them to NumPy values. |
| L32 | Closes the multiline call, collection, or grouped expression above; it adds no separate numerical operation. |
| L34 | Computes accuracy over individual examples as a fraction between zero and one. |
| L36 | Begins printing a readable progress or metric message assembled by the following lines. |
| L37 | Prints the final test accuracy as a percentage with two decimal places. |
| L38 | Closes the multiline call, collection, or grouped expression above; it adds no separate numerical operation. |

### Cell 21: Plot a confusion matrix

For example, the glacier row and mountain column count true glaciers predicted as mountains. These are raw counts, so classes with more images contribute more. To force a 6×6 matrix even when a class is absent, explicitly supply `labels=range(6)`.

```python
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

cm = confusion_matrix(
    all_labels,
    all_predictions
)

plt.figure(figsize=(7, 7))
plt.imshow(cm)

plt.xticks(
    range(6),
    train_data.classes,
    rotation=45
)

plt.yticks(
    range(6),
    train_data.classes
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

plt.colorbar()

plt.show()
```

| Line | Explanation |
|---|---|
| L1 | Imports the sklearn function that counts true/predicted class pairs. |
| L2 | Imports plotting functions under the name `plt`. |
| L4 | Builds a confusion matrix from all held-out predictions. |
| L5 | Supplies true class IDs first; these determine matrix rows. |
| L6 | Supplies predicted class IDs second; these determine matrix columns. |
| L7 | Closes the multiline call, collection, or grouped expression above; it adds no separate numerical operation. |
| L9 | Creates a square plotting canvas measured in inches. |
| L10 | Displays confusion counts as a colored image; diagonal entries are correct predictions. |
| L12 | Begins labeling the predicted-class axis. |
| L13 | Places six tick positions at integer class indices zero through five. |
| L14 | Uses the training Dataset's class names as readable labels. |
| L15 | Rotates x-axis labels 45 degrees to reduce overlap. |
| L16 | Closes the multiline call, collection, or grouped expression above; it adds no separate numerical operation. |
| L18 | Begins labeling the actual-class axis. |
| L19 | Uses the six class-index positions for the vertical axis. |
| L20 | Uses the matching class names for actual labels. |
| L21 | Closes the multiline call, collection, or grouped expression above; it adds no separate numerical operation. |
| L23 | Labels the horizontal axis as predicted class. |
| L24 | Labels the vertical axis as actual class. |
| L25 | Sets the figure title. |
| L27 | Adds a color scale explaining the magnitude of cell counts. |
| L29 | Renders the completed plot in the notebook. |

### Cell 23: Upload an image in Colab

```python
from google.colab import files

uploaded = files.upload()
```

| Line | Explanation |
|---|---|
| L1 | Imports the Colab-specific upload helper; it is unavailable in ordinary local Jupyter installations. |
| L3 | Opens an interactive upload dialog and stores a filename-to-file-bytes dictionary. |

### Cell 24: Predict the uploaded image

```python
from PIL import Image
import torch

def predict_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image = test_transform(image).unsqueeze(0)  # Add batch dimension
    image = image.to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs.data, 1)

    print(f"Predicted class: {train_data.classes[predicted.item()]}")

image_path = list(uploaded.keys())[0]
predict_image(image_path)
```

| Line | Explanation |
|---|---|
| L1 | Imports Pillow image loading and conversion utilities. |
| L2 | Imports PyTorch; tensors, device selection, and gradient-control functions use this name. |
| L4 | Defines a helper that accepts an image path and prints its predicted scene class. |
| L5 | Loads the image and forces three-channel RGB, avoiding grayscale/RGBA channel mismatches. |
| L6 | Applies deterministic evaluation preprocessing, then adds a batch axis: `[3,128,128]` becomes `[1,3,128,128]`. |
| L7 | Moves that one-image batch to the model device. |
| L9 | Selects evaluation behavior for BatchNorm and Dropout before prediction. |
| L10 | Disables graph recording for the indented inference block, reducing work and memory. |
| L11 | Computes the single image's six logits inside the no-grad block. |
| L12 | Takes the maximum score across the six columns and keeps its index. `.data` is unnecessary here; `outputs.argmax(dim=1)` expresses the intent directly. |
| L14 | Converts the one-element predicted ID tensor to a Python integer and indexes the matching class name. |
| L16 | Uses the first uploaded filename; this requires at least one successful upload and ignores additional uploaded images. |
| L17 | Calls the prediction helper on that file. |

### Cell 25: Display the image being classified

```python
import matplotlib.pyplot as plt
from PIL import Image

# Load the image using PIL
img = Image.open(image_path)

# Display the image using matplotlib
plt.imshow(img)
plt.axis('off') # Hide axes ticks
plt.title(f'Displayed Image: {image_path}')
plt.show()
```

| Line | Explanation |
|---|---|
| L1 | Imports plotting functions under the name `plt`. |
| L2 | Imports Pillow image loading and conversion utilities. |
| L4 | A comment describing the next operation; Python does not execute this line. |
| L5 | Reopens the selected file for display; this image is separate from the normalized tensor passed to the model. |
| L7 | A comment describing the next operation; Python does not execute this line. |
| L8 | Displays the original loaded image on the current plot. |
| L9 | Hides axis ticks so the image is easier to view; the inline comment is descriptive only. |
| L10 | Includes the selected filename in the plot title. |
| L11 | Renders the completed plot in the notebook. |

### Cell 26: Empty final cell

This cell is empty. It performs no operation.

## What the current notebook establishes, and what remains missing

The code trains a model and calculates test accuracy after the last epoch. There is no validation loader, early stopping, checkpoint saving, or learning-curve comparison on held-out validation data. Avoid choosing later architecture changes by repeatedly checking the test score; reserve training images for a validation split when extending the experiment.

ImageFolder independently derives labels for train and test. Check `train_data.class_to_idx == test_data.class_to_idx` before trusting comparisons. The final Linear layer hard-codes six outputs; a changed dataset requires checking that this still matches its classes.

The training loss is an average of batch means. To obtain an exact sample mean, accumulate `loss.item() * labels.size(0)` and divide by the sample count. Training accuracy uses predictions made before each batch update, with random flips and dropout active, whereas test accuracy uses fixed final weights and deterministic preprocessing.

The notebook has no random seed setup, so initialization, shuffling, and augmentation can change results. It also does not save model weights or class metadata. For reusable inference, preserve the state dictionary, architecture settings, class mapping, and transform definition together.

## Troubleshooting

| Symptom | Explanation and useful check |
|---|---|
| ImageFolder cannot find classes | Inspect the downloaded directory with cell 3 and verify the double `seg_train/seg_train` nesting |
| DataLoader workers fail locally | Try `num_workers=0` for both loaders |
| Expected three channels | Confirm RGB conversion and `[B,3,H,W]` tensor order |
| Model/input device mismatch | Move both the model and each input to the selected device |
| Colab import fails locally | Supply a local path directly instead of using the upload dialog |
| Upload indexing fails | No file was selected; the first-key expression requires a nonempty dictionary |
| Weak performance on a personal image | Inspect domain differences, image content, and preprocessing; the classifier must still choose one of its six known classes |

## Check your understanding

1. Why does `Conv2d(3,32,3,padding=1)` retain 128×128? **Stride one and one-pixel padding offset the 3×3 kernel's border reduction.**
2. Why is the final Linear input 128 rather than `128×16×16`? **Adaptive average pooling has already reduced each channel to one value.**
3. Does `model.eval()` prevent gradients by itself? **No; no-grad handles that separately.**
4. What does a confusion matrix row mean? **The actual class; its columns show where those images were predicted.**
5. Why use the test transform for a personal image? **Inference should use the same deterministic size/scaling convention as evaluation.**
