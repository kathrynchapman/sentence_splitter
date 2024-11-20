# Sentence Splitter

A Python package for sentence splitting using a pre-trained transformer model.

## Description

**Sentence Splitter** is a Python package that provides accurate sentence segmentation using a transformer-based token classification model. The model is bundled with the package, eliminating the need for additional downloads or configurations. It's designed to handle long texts efficiently and supports GPU acceleration if available.

## Features

- **Transformer-Based Model**: Leverages a pre-trained transformer model for high-accuracy sentence splitting.
- **Bundled Model**: The model and tokenizer are included with the package—no extra downloads required.
- **Easy to Use**: Simple API for quick integration into your projects.
- **Handles Long Texts**: Efficiently processes long texts by splitting them into manageable chunks.
- **GPU Acceleration**: Automatically utilizes CUDA if available for faster processing.

## Installation

Install the package via pip:

```bash
pip install sentence_splitter
```

## Requirements

- Python 3.6 or higher
- `torch`
- `transformers`

> **Note**: These dependencies will be installed automatically when you install the package via pip.

## Usage

### Basic Example

```python
from sentence_splitter import SentenceSplitter

# Initialize the splitter
splitter = SentenceSplitter()

# Input text
text = "This is a test. Here is another sentence. And yet another one!"

# Get sentences
sentences = splitter.predict(text)

print(sentences)
```

**Output:**

```
['This is a test.', 'Here is another sentence.', 'And yet another one!']
```

### Processing Long Texts

The `predict` method can handle long texts by splitting them into chunks. You can adjust the parameters as needed:

```python
sentences = splitter.predict(
    text,
    max_seq_len=512,   # Maximum sequence length for each chunk
    stride=100,        # Overlap between chunks to preserve context
    batch_size=4       # Number of chunks to process at once
)
```

## API Reference

### `SentenceSplitter`

A class for splitting text into sentences using a pre-trained transformer model.

#### Initialization

```python
splitter = SentenceSplitter(device=None)
```

- **Parameters**:
  - `device` (_str_, optional): The device to run the model on (`'cuda'` or `'cpu'`). Defaults to `'cuda'` if available, otherwise `'cpu'`.

#### Methods

- `predict(text, max_seq_len=512, stride=100, batch_size=4)`

  Splits the input text into sentences.

  - **Parameters**:
    - `text` (_str_): The text to split.
    - `max_seq_len` (_int_, optional): Maximum sequence length for the model. Defaults to `512`.
    - `stride` (_int_, optional): Number of tokens to overlap between chunks. Defaults to `100`.
    - `batch_size` (_int_, optional): Number of chunks to process simultaneously. Defaults to `4`.
  - **Returns**:
    - _List[str]_: A list of sentences.

## How It Works

The package uses a token classification model that labels each token as:

- **B**: Beginning of a sentence.
- **E**: End of a sentence.
- **I**: Inside a sentence.

By processing the tokens and their predicted labels, the splitter reconstructs the sentences accurately, even in complex texts.

## Example: Splitting Complex Text

```python
text = """
Despite the rain, the match continued. Players were determined; fans were cheering. 
"Unbelievable!" shouted the commentator. It's a night to remember.
"""

sentences = splitter.predict(text)

for i, sentence in enumerate(sentences, 1):
    print(f"Sentence {i}: {sentence}")
```

**Output:**

```
Sentence 1: Despite the rain, the match continued.
Sentence 2: Players were determined; fans were cheering.
Sentence 3: "Unbelievable!" shouted the commentator.
Sentence 4: It's a night to remember.
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Author

- **Kathryn Chapman**
- Email: kathryn.chapman@iges.com


## Acknowledgments

- [Hugging Face Transformers](https://github.com/huggingface/transformers) for the transformer models.
- [PyTorch](https://pytorch.org/) for the deep learning framework.

## Contact

For any questions or suggestions, feel free to reach out via email.

---