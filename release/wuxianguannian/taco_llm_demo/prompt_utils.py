import math
import matplotlib.pyplot as plt
import json
import random
from transformers import PreTrainedTokenizerBase
from typing import Optional, List, Tuple


class BaseParser:
    def __init__(self, dataset: str, tokenizer: PreTrainedTokenizerBase):
        self.dataset = dataset
        self.tokenizer = tokenizer

    def load_dataset(self) -> List[Tuple[str, str]]:
        pass

    def tokenizen_dataset(
        self,
        dataset: List[Tuple[str, str]]
    ) -> List[Tuple[str, int, int]]:
        prompts = [prompt for prompt, _ in dataset]
        prompt_token_ids = self.tokenizer(prompts).input_ids

        completions = [completion for _, completion in dataset]
        completion_token_ids = self.tokenizer(completions).input_ids

        tokenized_dataset = []
        for i in range(len(dataset)):
            prompt_len = len(prompt_token_ids[i])
            output_len = len(completion_token_ids[i])
            tokenized_dataset.append((prompts[i], prompt_len, output_len))

        return tokenized_dataset

    def filter_dataset(
        self,
        dataset: List[Tuple[str, int, int]], 
        min_input_len: int,
        max_input_len: int,
        min_output_len: int,
        max_output_len: int,
        test_prompt_latency: bool = False,
    ) -> List[Tuple[str, int, int]]:
        # Filter out too long sequences.
        filtered_dataset: List[Tuple[str, int, int]] = []
        for prompt, prompt_len, output_len in dataset:
            if (prompt_len < min_input_len or prompt_len > max_input_len) \
                or (output_len < min_output_len or output_len > max_output_len):
                # Prune too short sequences.
                continue
            if test_prompt_latency:
                output_len = 1
            else:
                output_len = 550
            filtered_dataset.append((prompt, prompt_len, output_len))
        
        if len(filtered_dataset) == 0:
            prompt = "A" * 32768
            prompt_len = len(prompt)
            output_len = 170
            filtered_dataset.append((prompt, prompt_len, output_len))

        return filtered_dataset

    def sample_requests(
        self,
        num_requests: int,
        min_input_len: int,
        max_input_len: int,
        min_output_len: int,
        max_output_len: int,
        test_prompt_latency: bool = False,
    ) -> List[Tuple[str, int, int]]:
        dataset = self.load_dataset()

        tokenized_dataset = self.tokenizen_dataset(dataset)

        filter_dataset = self.filter_dataset(
            tokenized_dataset,
            min_input_len,
            max_input_len,
            min_output_len,
            max_output_len,
            test_prompt_latency=test_prompt_latency
        )

        sampled_requests = []
        multi = math.ceil(num_requests / len(filter_dataset))
        for i in range(multi):
            sampled_requests.extend(filter_dataset)
        sampled_requests = sampled_requests[:num_requests]

        self.print_prompt_stats(sampled_requests)

        return sampled_requests

    def sample_fix_requests(
        self,
        num_requests: int,
        min_input_len: int,
        max_input_len: int,
        min_output_len: int,
        max_output_len: int,
        test_prompt_latency: bool = False,
    ) -> List[Tuple[str, int, int]]:
        dataset = self.load_dataset()
        current_text = ""
        concatenated_texts = []
        filtered_dataset: List[Tuple[str, int, int]] = []

        if test_prompt_latency:
            output_len = 1
        else:
            output_len = min_output_len

        for text in dataset:
            current_text += text
            tokenized_text = self.tokenizer(current_text, return_tensors="pt")
            if len(tokenized_text["input_ids"][0]) > min_input_len:
                concatenated_texts.append({"text": current_text})
                filtered_dataset.append((current_text, len(tokenized_text["input_ids"][0]), output_len))
                # print(len(concatenated_texts))
                if (len(concatenated_texts)) == num_requests:
                    break
                current_text = ""

        sampled_requests = []
        multi = math.ceil(num_requests / len(filtered_dataset))
        for i in range(multi):
            sampled_requests.extend(filtered_dataset)
        sampled_requests = sampled_requests[:num_requests]

        self.print_prompt_stats(sampled_requests)

        return sampled_requests

    def print_prompt_stats(
        self,
        sampled_requests: List[Tuple[str, int, int]]
    ) -> None:
        total_prompt_len = 0
        total_output_len = 0
        for prompt, prompt_len, output_len in sampled_requests:
            total_prompt_len += prompt_len
            total_output_len += output_len

        sampled_requests_num = len(sampled_requests)
        avg_prompt_len = total_prompt_len / sampled_requests_num
        avg_output_len = total_output_len / sampled_requests_num

        print(f"-*-*-*-*-*- sampled prompt num:{sampled_requests_num}\n"
              f"-*-*-*-*-*- avg prompt len:{avg_prompt_len}\n"
              f"-*-*-*-*-*- avg output len:{avg_output_len}")


class ShareGPTParser(BaseParser):
    def load_dataset(self) -> List[Tuple[str, str]]:
        # Load the dataset.
        with open(self.dataset) as f:
            datas = json.load(f)
        dataset = []
        for data in datas:
            if len(data["conversations"]) < 3:
                continue
            out = self._build_prompt(data["conversations"])
            if out is None:
                continue
            if '```' in out[0]:
                # Filter out code generation related scenarios
                continue
            dataset.append(out)
        return dataset

    def _build_prompt(self, conversations: List) -> Optional[Tuple[str, str]]:
        delim = "@@@"
        while conversations[-1]["from"] == "human":
            conversations = conversations[:-1]
            if len(conversations) < 5:
                return None
        if conversations[-1]["from"] == "gpt":
            ref_gen = conversations[-1]["value"]
            conversations = conversations[:-1]
        else:
            return None
        prompt = f" {delim} ".join(
                [f"{x['from']}: {x['value']}" for x in conversations]) + delim + " gpt:"
        return (prompt, ref_gen)


class PythonProgramParser(BaseParser):
    def load_dataset(self) -> List[Tuple[str, str]]:
        # Load the dataset.
        with open(self.dataset) as f:
            datas = [json.loads(line.strip()) for line in f]
        # Only keep the first two turns of each conversation.
        dataset = []
        for data in datas:
            prompt = "\n#".join(data["text"].split("|"))
            define = data["code"].split(':', 1)[0]
            if "def " not in define:
                continue
            define = define.replace("NEW_LINE", "\n")
            prompt = "#" + prompt + "\n" + define + ":\n"
            dataset.append((prompt, data["code"]))
        return dataset


class FixSampleParser(BaseParser):
    def load_dataset(self) -> List[Tuple[str]]:
        # Load the dataset.
        with open(self.dataset) as f:
            datas = [json.loads(line.strip()) for line in f]
        # Only keep the first two turns of each conversation.
        dataset = []
        for data in datas:
            prompt = data["text"]
            if len(prompt) < 500:
                dataset.append((prompt))
        return dataset


class FixGPTSampleParser(BaseParser):    
    def load_dataset(self) -> List[Tuple[str, str]]:
        # Load the dataset.
        with open(self.dataset) as f:
            datas = json.load(f)
        dataset = []
        for data in datas:
            for conversations in data["conversations"]:
                # print(len(conversations))
                if conversations["from"] == "gpt" and len(conversations["value"]) < 500:
                    # print(conversations["value"])
                    dataset.append((conversations["value"]))
        # print(f"=======haiwu len dataset {len(dataset)}")
        for i in range(10):
            print(dataset[i])
        return dataset


def plot_hist(requests: List[Tuple[str, int, int]]) -> None:
    prompt_lens = [req[1] for req in requests]
    output_lens = [req[2] for req in requests]
    fig, ax=plt.subplots(figsize=(8,5))

    ax.hist(
        [prompt_lens, output_lens],
        bins=20,
        color=["blue", "yellow"],
        histtype='bar',
        label=["prompt", "output"]
    )

    ax.set_title('Prompt/Output len hist')
    ax.set_xlabel('Len')
    ax.set_ylabel('Frequency')
    ax.legend()
    plt.savefig("prompt_output.jpg")
    plt.close()
