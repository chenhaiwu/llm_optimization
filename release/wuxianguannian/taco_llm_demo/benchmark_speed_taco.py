import multiprocessing
import requests
import time
import random

import argparse
import asyncio
import json
import os
import ast
import random
import time
from typing import AsyncGenerator, List, Tuple, Iterable

import aiohttp
import numpy as np
from transformers import PreTrainedTokenizerBase
from taco_llm.transformers_utils.tokenizer import get_tokenizer
from prompt_utils import ShareGPTParser, PythonProgramParser, FixSampleParser


def get_requests(args, tokenizer):

    is_fix_requests = False
    if "ShareGPT" in args.dataset:
        dataset_parser = ShareGPTParser(args.dataset, tokenizer)
    elif "python-program" in args.dataset:
        dataset_parser = PythonProgramParser(args.dataset, tokenizer)
    elif "c4_sample.jsonl" in args.dataset:
        dataset_parser = FixSampleParser(args.dataset, tokenizer)
        is_fix_requests = True
    else:
        raise ValueError(f"Unknown supported dataset.")

    prompt_filename = f"request_{args.num_prompts}_1.npy"
    if not os.path.isfile(prompt_filename):
        # Test prompt processing latency
        if not is_fix_requests:
            input_requests = dataset_parser.sample_requests(
                args.num_prompts,
                args.min_prompt_len,
                args.max_prompt_len,
                args.min_gen_len,
                args.max_gen_len,
                test_prompt_latency=False
            )
        else:
            input_requests = dataset_parser.sample_fix_requests(
                args.num_prompts,
                args.min_prompt_len,
                args.max_prompt_len,
                args.min_gen_len,
                args.max_gen_len,
                test_prompt_latency=False
            )
        np.save(prompt_filename, input_requests)
    else:
        input_requests = np.load(prompt_filename)

    return input_requests


def post_http_request(prompt: str,
                      api_url: str,
                      top_p: float,
                      temperature: float,
                      n: int,
                      max_tokens: int,
                      use_beam_search: bool) -> requests.Response:
    headers = {"User-Agent": "Test Client"}
    
    #print("max_tokens : {}".format(max_tokens))
    if use_beam_search:
        temperature = 0
        top_p = 1.0
        if n == 1:
            n = 2
    pload = {
        "prompt": prompt,
        "n": n,
        "best_of": n,
        "use_beam_search": use_beam_search,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": top_p,
        "stream":True,
        "ignore_eos": True,
        # "incremental_return": True
#        "input_logprobs": True,
#        "stop": ["Human:"]
    }
#    print (pload)
    #print("prompt is :{}".format(prompt))
    response = requests.post(api_url, headers=headers, json=pload, stream=True)
    return response


def get_streaming_response(response: requests.Response) -> Iterable[List[str]]:
    first = True
    start = time.time()
    for chunk in response.iter_lines(
                                     #chunk_size=8192,
                                     #decode_unicode=False,
                                     delimiter=b'\x00'
                                     ):
        # print("chunk is : {}".format(chunk))
        if chunk:
            # assert chunk[0:5] == b'data:'
            # data = json.loads(chunk[5:].decode("utf-8"))
            json_obj = json.loads(chunk)
            word = json_obj['text']
            #print("data is : {}".format(word))
            
            if first:
                first = False
                start = time.time()
             
    return start
    #return 0


# 定义一个函数来模拟处理请求并记录统计信息
def process_request(prompt: str,
                    prompt_tokens: int,
                    output_tokens: int,
                    api_url: str,
                    top_p: float,
                    temperature: float,
                    n: int,
                    use_beam_search: bool,
                    results):
    #print("process_request_1")
    response = post_http_request(prompt, api_url, top_p, temperature,
                                 n, output_tokens, use_beam_search)
    start_time = time.time()
    #print("process_request_2")
    first_time = get_streaming_response(response)
    end_time = time.time()

    #print("start_time : {}, first_time : {}, end_time : {}".format(start_time, first_time, end_time))
    
    #print("prompt_tokens : {}, output_tokens :{} ".format(prompt_tokens, output_tokens))
    #print("process_request_3")
    results.append((first_time - start_time, end_time - start_time,
                    prompt_tokens, output_tokens))

def sample_requests(
    dataset_path: str,
    num_requests: int,
    tokenizer: PreTrainedTokenizerBase,
) -> List[Tuple[str, int, int]]:
    # Load the dataset.
    with open(dataset_path) as f:
        dataset = json.load(f)
    # Filter out the conversations with less than 2 turns.
    dataset = [
        data for data in dataset
        if len(data["conversations"]) >= 2
    ]
    # Only keep the first two turns of each conversation.
    dataset = [
        (data["conversations"][0]["value"], data["conversations"][1]["value"])
        for data in dataset
    ]
    # Tokenize the prompts and completions.
    prompts = [prompt for prompt, _ in dataset]
    prompt_token_ids = tokenizer(prompts).input_ids
    completions = [completion for _, completion in dataset]
    completion_token_ids = tokenizer(completions).input_ids
    tokenized_dataset = []

    for i in range(len(dataset)):
        output_len = len(completion_token_ids[i])
        tokenized_dataset.append((prompts[i], prompt_token_ids[i], output_len))

    # Filter out too short sequences.
    filtered_dataset: List[Tuple[str, int, int]] = []
    for prompt, prompt_token_ids, output_len in tokenized_dataset:
        prompt_len = len(prompt_token_ids)
        if prompt_len < 4 or output_len < 4:
            continue
        # if prompt_len > 1024 or prompt_len + output_len > 2048:
        #     # Prune too long sequences.
        #     continue
        filtered_dataset.append((prompt, prompt_len, output_len))
    # Sample the requests.
    sampled_requests = random.sample(filtered_dataset, num_requests)
    return sampled_requests
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--top-p", type=float, default=0.85)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument("--use-beam-search", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--parallel", type=int, default=1)
    parser.add_argument("--tokenizer", type=str, required=True,
                        help="Name or path of the tokenizer.")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Path to the dataset.")
    parser.add_argument("--num-prompts", type=int, default=1000,
                        help="Number of prompts to process.")
    parser.add_argument("--min-prompt-len", type=int, default=4)
    parser.add_argument("--max-prompt-len", type=int, default=512)
    parser.add_argument("--min-gen-len", type=int, default=4)
    parser.add_argument("--max-gen-len", type=int, default=512)
    
    args = parser.parse_args()


    api_url = f"http://{args.host}:{args.port}/generate"
    n = args.n
    use_beam_search = args.use_beam_search
    top_p = args.top_p
    temperature = args.temperature

    random.seed(args.seed)
    np.random.seed(args.seed)

    tokenizer = get_tokenizer(args.tokenizer, trust_remote_code=True)
    print(f"======haiwu====={len(tokenizer)}")

    """
    input_requests = sample_requests(args.dataset,
                                     args.num_prompts,
                                     tokenizer)
    """

    input_requests = get_requests(args, tokenizer)

    num_processes = args.parallel  # 设置要创建的进程数量
    print(f"num_processes: {num_processes}")

    # 使用multiprocessing.Pool创建多个进程
    pool = multiprocessing.Pool(processes=num_processes)
    manager = multiprocessing.Manager()
    results = manager.list()  # 使用Manager来创建共享的结果列表

    # 在多个进程中处理请求
    start_time = time.time()
    count = 0
    for input_request in input_requests:
        #print("input_request_0 is : {}, input_request_1 is : {}".format(input_request[1], input_request[2]))
        pool.apply_async(process_request, args=(input_request[0],
                                                input_request[1],
                                                input_request[2],
                                                api_url,
                                                top_p,
                                                temperature,
                                                n,
                                                use_beam_search,
                                                results))
    # 等待所有进程完成

    pool.close()

    pool.join()
    
    end_time = time.time()
    
    # 汇总统计信息
    total_first_time = 0
    total_cost_time = 0
    total_output_length = 0
    total_prompt_length = 0
    print("results is : {}".format(results))
    for first_time, cost_time, prompt_length, output_length in results:

        total_first_time += first_time
        total_cost_time += cost_time
        total_output_length += output_length
        total_prompt_length += prompt_length

    # 输出统计结果
    total_time = end_time - start_time
    print(f"Total requests:           {len(input_requests)}")
    print(f"Total time taken:         {total_time} seconds")
    print(f"Total prompt:             {total_prompt_length}")
    print(f"Total output:             {total_output_length}")    
    print(f"Average first token take: {total_first_time / len(input_requests)} seconds")
    if total_cost_time - total_first_time != 0:
        print(f"Average generated tokens: {total_output_length / (total_cost_time - total_first_time)} tokens/sec")
    
