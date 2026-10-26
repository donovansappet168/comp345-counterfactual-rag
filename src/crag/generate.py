"""Llama 3.1 8B Instruct wrapper.

For one prompt we want three things:
  1. the greedy answer (this gives the truth label)
  2. N_SAMPLES sampled answers (these give the true-answer rate)
  3. the hidden state of the last prompt token at every layer (for the probes)

Usage:
    llm = Llama()
    result = llm.answer("Answer the question.\\n\\nQuestion: ...\\nAnswer:")
    result["greedy"], result["samples"], result["hidden"]
"""

import numpy as np
import torch
from numpy.typing import NDArray
from transformers import AutoModelForCausalLM, AutoTokenizer

from crag.config import CHAT_DATE, GENERATOR, MAX_NEW_TOKENS, N_SAMPLES, SEED
from crag.schema import Answer


class Llama:
    """Loads the model once and answers prompts with it."""

    def __init__(self, model_name: str = GENERATOR) -> None:
        """Load the tokenizer and the model in bf16, on the GPU if there is one.

        Args:
            model_name: Hugging Face model name (default: config.GENERATOR).
        """
        if torch.cuda.is_available(): # If machine has a CUDA GPU, use it
            self.device = "cuda"
        else: # Otherwise, use CPU
            self.device = "cpu"
        # Auto = picks the right model/tokenizer class from the model's config files - easy to switch model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name) # Load pretrained Llama tokenizer - turns text into token ids Llama can read (and back)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=torch.bfloat16, device_map=self.device
        ) # Load pretrained Llama model in 16-bit (~16 GB) - causal = reads tokens left to right + predicts the next token
        self.model.eval() # Evaluation mode - turns off training-only behavior (e.g. dropout); weights are never trained here

    def encode(self, prompt: str) -> torch.Tensor:
        """Turn the prompt into token ids in Llama's chat format.

        Llama Instruct expects a conversation, so the prompt goes in as one user
        message, followed by the header that starts the assistant's reply.

        Args:
            prompt: The filled-in template from prompt.py.

        Returns:
            Token ids, shape (1, number of prompt tokens), on the model's device.
        """
        messages = [{"role": "user", "content": prompt}] # Input prompt as a user message
        input_ids: torch.Tensor = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True,
            date_string=CHAT_DATE,
        )["input_ids"] # Formats conversation with Llama’s special marker tokens (hidden system message with "Today Date: CHAT_DATE", user turn, assistant header) and tokenizes the result into ids that the Llama model can read as a PyTorch tensor
        return input_ids.to(self.device) # Moves tensor to the same device as the model weights

    def decode(self, output_ids: torch.Tensor, prompt_length: int) -> str:
        """Turn generated token ids back into the answer text.

        Args:
            output_ids: One generated sequence (prompt tokens followed by answer tokens).
            prompt_length: Number of prompt tokens at the start, which are dropped.

        Returns:
            The answer text, without special tokens or surrounding spaces.
        """
        answer_ids = output_ids[prompt_length:] # Get answer token id list - drop prompt tokens
        text: str = self.tokenizer.decode(answer_ids, skip_special_tokens=True) # Decode token ids into a string of words, skipping special tokens (e.g. <|eot_id|>)
        return text.strip() # Remove spaces/newlines around the answer

    @torch.no_grad()
    def hidden_states(self, input_ids: torch.Tensor) -> NDArray[np.float32]:
        """Get the last prompt token's hidden state from every layer.

        One forward pass over the prompt. The model returns 33 hidden-state tensors:
        the embedding layer plus 32 transformer layers. We skip the embeddings and
        keep the last token's vector from each layer.

        Args:
            input_ids: Prompt token ids from encode().

        Returns:
            Array of shape (32, 4096): row i is layer i + 1.
        """
        out = self.model(input_ids, output_hidden_states=True) # Run the model once over the prompt tokens, keeping every layer's hidden states
        layers = out.hidden_states[1:] # Remove embedding layer - look only at the 32 transformer layers
        last_token = [layer[0, -1] for layer in layers] # For each layer: prompt 0 in the batch, last token (-1) -> its 4096-number vector in Llama 3.1 8B
        hidden: NDArray[np.float32] = torch.stack(last_token).float().cpu().numpy() # Stack the 32 vectors into a (32, 4096) array: bf16 -> float32, move to CPU, convert to NumPy
        return hidden # Return last-token hidden states, row i = layer i + 1

    @torch.no_grad()
    def greedy(self, input_ids: torch.Tensor) -> str:
        """Generate one answer by always taking the most likely next token.

        Args:
            input_ids: Prompt token ids from encode().

        Returns:
            The answer text.
        """
        out = self.model.generate(
            input_ids, do_sample=False, max_new_tokens=MAX_NEW_TOKENS,
            pad_token_id=self.tokenizer.eos_token_id,
        ) # Generate an answer by always taking the most likely next token: max size of response is MAX_NEW_TOKENS
        return self.decode(out[0], input_ids.shape[1]) # Decode answer tokens into text

    @torch.no_grad()
    def sample(self, input_ids: torch.Tensor, n: int, seed: int = SEED) -> list[str]:
        """Generate n answers by sampling from the model's output probabilities.

        Temperature 1 with no top-k or top-p, so each answer is a draw from the
        model's own distribution.

        Args:
            input_ids: Prompt token ids from encode().
            n: Number of answers to sample.
            seed: Random seed, so the samples are the same on every run.

        Returns:
            The n answer texts.
        """
        torch.manual_seed(seed) # Seed PyTorch's RNG so the samples repeat on every run
        out = self.model.generate(
            input_ids, do_sample=True, temperature=1.0, top_k=0, top_p=1.0,
            num_return_sequences=n, max_new_tokens=MAX_NEW_TOKENS,
            pad_token_id=self.tokenizer.eos_token_id,
        ) # Generate n answers, picking each token at random by its probability (temperature 1, no top-k/top-p cutoffs = the model's own distribution)
        return [self.decode(row, input_ids.shape[1]) for row in out] # Decode each sampled answer into text

    def answer(self, prompt: str, n_samples: int = N_SAMPLES,
               keep_hidden: bool = True) -> Answer:
        """Run everything we need for one prompt.

        Args:
            prompt: The filled-in template from prompt.py.
            n_samples: Number of sampled answers (0 to skip sampling).
            keep_hidden: Whether to also compute the hidden states.

        Returns:
            The greedy answer, the sampled answers, the hidden states (or None),
            and the number of prompt tokens.
        """
        input_ids = self.encode(prompt) # Prompt -> token ids in chat format
        result: Answer = {
            "prompt_tokens": input_ids.shape[1], # Prompt length in tokens
            "greedy": self.greedy(input_ids), # Greedy answer (gives the label)
            "samples": [], # Filled in below if sampling
            "hidden": None, # Filled in below if hidden states are wanted
        }
        if n_samples > 0: # If sampling is on
            result["samples"] = self.sample(input_ids, n_samples) # Draw n_samples answers (true-answer rate)
        if keep_hidden: # If hidden states are wanted (RAG prompts, for the probes)
            result["hidden"] = self.hidden_states(input_ids) # Last-token hidden states, (32, 4096)
        return result # Return greedy answer, samples, hidden states, prompt length
