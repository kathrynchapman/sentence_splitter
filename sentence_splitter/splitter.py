from transformers import AutoTokenizer, AutoModelForTokenClassification, BitsAndBytesConfig
import torch

class SentenceSplitter:
    def __init__(self, device=None, efficient_mode=False):
        """
        Initialize the SentenceSplitter with the model from Hugging Face Hub.

        The model will be automatically downloaded and cached on first use.
        """
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')

        # Hugging Face Hub model name
        self.model_name = "kathryn-chapman/iges-sentence-splitter"

        print(f"Loading model from Hugging Face Hub: {self.model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if efficient_mode:
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            self.model = AutoModelForTokenClassification.from_pretrained(
                self.model_name,
                quantization_config=quantization_config
            )
        else:
            self.model = AutoModelForTokenClassification.from_pretrained(self.model_name).to(self.device)
        self.labels = ['B', 'E', 'I']

    def reconstruct_labels(self, tokens, labels):
        reconstructed_words = []
        reconstructed_labels = []

        # Temporary variables to collect tokens for each word
        current_word = ""
        current_labels = []

        # Process tokens and labels together
        for token, label in zip(tokens, labels):
            # Check if the token starts a new word (by the presence of '▁' at the beginning)
            if token.startswith("▁"):
                # If we have an ongoing word, add it to the list
                if current_word:
                    reconstructed_words.append(current_word)
                    # Store the final label for the reconstructed word
                    reconstructed_labels.append(current_labels[0] if current_labels else "I")

                # Start a new word
                current_word = token[1:]  # remove the leading '▁'
                current_labels = [label]
            elif label == 'B':
                if current_word:
                    reconstructed_words.append(current_word)
                    # Store the final label for the reconstructed word
                    reconstructed_labels.append(current_labels[0] if current_labels else "I")

                # Start a new word
                current_word = token
                current_labels = [label]
            else:
                # Continue building the current word
                current_word += token
                current_labels.append(label)

        # Append the last word if any
        if current_word:
            reconstructed_words.append(current_word)
            reconstructed_labels.append(current_labels[0] if current_labels else "I")
        return reconstructed_words, reconstructed_labels

    def split(self, text, max_seq_len=512, stride=100, batch_size=24):
        """
        Processes long text for prediction by splitting into manageable chunks,
        adding special tokens, and aggregating model outputs.

        Fixes:
        - Ensures start and end special tokens are added to each chunk appropriately.
        - Updates the attention mask to correctly account for special tokens.
        """
        # Tokenize the input text without truncation
        encodings = self.tokenizer(text, return_tensors="pt", add_special_tokens=False, truncation=False)
        input_ids = encodings["input_ids"].squeeze(0)
        input_tokens = self.tokenizer.convert_ids_to_tokens(input_ids.tolist(), skip_special_tokens=False)

        # Retrieve special tokens' IDs
        cls_token_id = self.tokenizer.cls_token_id  # Start token ID
        sep_token_id = self.tokenizer.sep_token_id  # End token ID

        chunks = []
        for i in range(0, len(input_ids), max_seq_len - stride - 2):  # Adjust for special tokens
            chunk = input_ids[i:i + (max_seq_len - 2)]  # Reserve space for start and end tokens
            # Prepend CLS token and append SEP token
            chunk = torch.cat(
                [torch.tensor([cls_token_id]), chunk, torch.tensor([sep_token_id])], dim=0
            )
            # Pad to max_seq_len if needed
            if len(chunk) < max_seq_len:
                chunk = torch.cat([chunk, torch.zeros(max_seq_len - len(chunk), dtype=torch.long)])
            chunks.append(chunk.unsqueeze(0))

        # Concatenate all chunks
        chunks = torch.cat(chunks)

        # Create attention mask: 1 for tokens (including special tokens), 0 for padding
        attention_mask = (chunks != 0).long()

        # Model predictions
        logits_list = []
        with torch.no_grad():
            for i in range(0, len(chunks), batch_size):
                batch_input_ids = chunks[i:i + batch_size].to(self.device)
                batch_attention_mask = attention_mask[i:i + batch_size].to(self.device)
                batch_attention_mask[:, 0] = 1
                # Forward pass through the model
                outputs = self.model(input_ids=batch_input_ids, attention_mask=batch_attention_mask)
                logits = outputs.logits.detach().cpu()
                logits_list.append(logits)

        # Combine logits from all chunks
        all_logits = torch.cat(logits_list, dim=0)

        # Aggregate logits over overlapping tokens
        final_logits = torch.zeros((len(input_ids), all_logits.size(-1)))
        token_counts = torch.zeros(len(input_ids))

        for i, chunk_logits in enumerate(all_logits):
            start_idx = i * (max_seq_len - stride - 2)
            end_idx = start_idx + max_seq_len - 2  # Exclude special tokens in aggregation
            chunk_length = min(len(input_ids) - start_idx, max_seq_len - 2)
            final_logits[start_idx:start_idx + chunk_length] += chunk_logits[1:1 + chunk_length]  # Exclude CLS
            token_counts[start_idx:start_idx + chunk_length] += 1

        # Average logits
        final_logits /= token_counts.unsqueeze(-1)

        # Predictions
        predictions = torch.argmax(final_logits, dim=-1)
        predicted_labels = [self.labels[pred.item()] for pred in predictions]  # Skip special tokens

        # Reconstruct sentences based on predicted labels
        sentences = []
        current_sentence = []
        input_tokens, predicted_labels = self.reconstruct_labels(input_tokens, predicted_labels)
        for token, label in zip(input_tokens, predicted_labels):
            if label == "B":
                if current_sentence:
                    sentences.append(" ".join(current_sentence))
                current_sentence = [token]
            elif label == "E":
                current_sentence.append(token)
                sentences.append(" ".join(current_sentence))
                current_sentence = []
            else:  # "I"
                current_sentence.append(token)
        if current_sentence:
            sentences.append(" ".join(current_sentence))

        sentences = [s.strip() for s in sentences]
        return sentences
