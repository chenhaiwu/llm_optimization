"""Benchmark online serving throughput.
"""
import argparse
import asyncio
import json
import math
import random
import time
import os
import numpy as np
from typing import AsyncGenerator, Optional, List, Tuple
from tqdm import tqdm

import aiohttp
import numpy as np
from transformers import PreTrainedTokenizerBase
from taco_llm.transformers_utils.tokenizer import get_tokenizer

from prompt_utils import ShareGPTParser, PythonProgramParser, FixSampleParser
import requests

# (prompt len, output len, latency)
REQUEST_LATENCY: List[Tuple[int, int, float]] = []


async def get_request(
    input_requests: List[Tuple[str, int, int]],
    request_rate: float,
) -> AsyncGenerator[Tuple[str, int, int], None]:
    input_requests = iter(input_requests)
    for request in input_requests:
        yield request

        if request_rate == float("inf"):
            # If the request rate is infinity, then we don't need to wait.
            continue
        # Sample the request interval from the exponential distribution.
        interval = np.random.exponential(1.0 / request_rate)
        # The next request will be sent after the interval.
        await asyncio.sleep(interval)


async def send_request(
    backend: str,
    api_url: str,
    prompt: str,
    prompt_len: int,
    output_len: int,
    best_of: int,
    use_beam_search: bool,
    tokenizer,
    prompt_time=None
) -> None:
    request_start_time = time.time()

    headers = {"User-Agent": "Benchmark Client"}
    if backend == "taco_llm":
        pload = {
            "prompt": prompt,
            "n": 1,
            "best_of": best_of,
            "use_beam_search": use_beam_search,
            "temperature": 0.0 if use_beam_search else 0.0,
            "top_p": 1.0,
            "max_tokens": output_len,
            "ignore_eos": True,
            "stream": False,
        }
    elif backend == "tgi":
        assert not use_beam_search
        params = {
            "best_of": best_of,
            "max_new_tokens": output_len,
            "do_sample": False,
        }
        pload = {
            "inputs": prompt,
            "parameters": params,
        }
    elif backend == "light_llm":
        params = {
            "do_sample": False,
            "ignore_eos": False,
            "max_new_tokens": output_len,
        }
        pload = {
            "inputs": prompt,
            "parameters": params,
        }
    elif backend == "trt_llm":
        pload = {
            "text_input": prompt,
            "max_tokens": output_len,
            "bad_words": "",
            "stop_words": "",
            "end_id": 0,
        }
    else:
        raise ValueError(f"Unknown backend: {backend}")


    timeout = aiohttp.ClientTimeout(total=3 * 3600 * 100)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        # while True:
        #             async with session.post(api_url, headers=headers, json=pload) as response:
        #                 output = []
        #                 async for chunk, _ in response.content.iter_chunks():
        #                     # 处理每个块，例如打印输出
        #                     decoded_chunk = chunk.decode("utf-8")
        #                     output.append(decoded_chunk)
        #                     print(decoded_chunk, end="")

        #                 output = "".join(output)
        #                 json_objects = output.split("@@@")

        #                 for json_str in json_objects:
        #                     if json_str.strip():  # 过滤空字符串
        #                         # 检查字符串是否以正确的引号终止
        #                         if json_str.count('"') % 2 == 0:
        #                             json_obj = json.loads(json_str)
        #                             if "error" not in json_obj:
        #                                 break
        while True:
            async with session.post(api_url, headers=headers, json=pload) as response:
                chunks = []
                async for chunk, _ in response.content.iter_chunks():

                    chunks.append(chunk)
                    

            output = b"".join(chunks).decode("utf-8")
            output = json.loads(output)
            if "error" not in output:
                break

    if backend == "taco_llm":
        output_tokens = tokenizer(output["text"][0][len(prompt):]).input_ids
    elif backend == "tgi":
        output_tokens = tokenizer(output["generated_text"]).input_ids
    elif backend == "light_llm":
        output_tokens = tokenizer(output["generated_text"][0]).input_ids
    elif backend == "trt_llm":
        output_tokens = tokenizer(output["text_output"][len(prompt):]).input_ids
    else:
        raise ValueError(f"Unknown backend: {backend}")


    output_len = len(output_tokens) - 2

    request_end_time = time.time()
    request_latency = request_end_time - request_start_time

    if prompt_time is None:
        print(f"start time: {request_start_time}, "
              f"end time: {request_end_time}, "
              f"latency: {request_latency}, "
              f"output len: {output_len}")
    else:
        print(f"start time: {request_start_time}, "
              f"end time: {request_end_time}, "
              f"latency: {request_latency}, "
              f"output len: {output_len}, "
              f"latency of per output token: {(request_latency - prompt_time) / output_len * 1000:.2f}")
    REQUEST_LATENCY.append((prompt_len, max(output_len, 1), request_latency))


async def benchmark(
    backend: str,
    api_url: str,
    input_requests: List[Tuple[str, int, int]],
    best_of: int,
    use_beam_search: bool,
    request_rate: float,
    tokenizer,
    prompt_time=None
) -> None:
    tasks: List[asyncio.Task] = []
    async for request in get_request(input_requests, request_rate):
        prompt, prompt_len, output_len = request
        task = asyncio.create_task(
            send_request(
                backend,
                api_url,
                str(prompt),
                int(prompt_len),
                int(output_len),
                best_of,
                use_beam_search,
                tokenizer,
                prompt_time=prompt_time
            )
        )
        tasks.append(task)
    await asyncio.gather(*tasks)


def main(args: argparse.Namespace):
    print(args)
    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.backend == "trt_llm":
        api_url = f"http://{args.host}:{args.port}/v2/models/ensemble/generate"
    else:
        api_url = f"http://{args.host}:{args.port}/generate"

    tokenizer = get_tokenizer(args.tokenizer, trust_remote_code=args.trust_remote_code)
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
                test_prompt_latency=True
            )
        else:
            input_requests = dataset_parser.sample_fix_requests(
                args.num_prompts,
                args.min_prompt_len,
                args.max_prompt_len,
                args.min_gen_len,
                args.max_gen_len,
                test_prompt_latency=True
            )
        np.save(prompt_filename, input_requests)
    else:
        input_requests = np.load(prompt_filename)

    benchmark_start_time = time.time()
    asyncio.run(
        benchmark(
            args.backend,
            api_url,
            input_requests,
            args.best_of,
            args.use_beam_search,
            args.request_rate,
            tokenizer
        )
    )
    benchmark_end_time = time.time()

    prompt_time = benchmark_end_time - benchmark_start_time

    global REQUEST_LATENCY
    # Compute the latency statistics.
    avg_prompt_latency = np.mean([latency for _, _, latency in REQUEST_LATENCY])
    print(f"Prompt time: {prompt_time}, Average first token latency: {avg_prompt_latency:.2f} s")

    REQUEST_LATENCY = []
    throughput_filename = f"request_{args.num_prompts}_0.npy"
    if not os.path.isfile(throughput_filename):
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
        np.save(throughput_filename, input_requests)
    else:
        input_requests = np.load(throughput_filename)

    benchmark_start_time = time.time()
    asyncio.run(
        benchmark(
            args.backend,
            api_url,
            input_requests,
            args.best_of,
            args.use_beam_search,
            args.request_rate,
            tokenizer,
            prompt_time=avg_prompt_latency
        )
    )
    benchmark_end_time = time.time()

    benchmark_time = benchmark_end_time - benchmark_start_time

    total_prompt_tokens = sum(
        prompt_len
        for prompt_len, _, _ in REQUEST_LATENCY
    )

    total_output_tokens = sum(
        output_len
        for _, output_len, _ in REQUEST_LATENCY
    )
    
    total_tokens = total_prompt_tokens + total_output_tokens
    gen_time = benchmark_time - avg_prompt_latency

    print(f"Total time: {benchmark_time:.2f} s,"
          f"prompt_time: {avg_prompt_latency} s,"
          f"generation_time: {gen_time} s,"
          f"total tokens: {total_tokens},"
          f"total prompt tokens: {total_prompt_tokens},"
          f"total output tokens: {total_output_tokens}")

    print(f"Throughput: {args.num_prompts / benchmark_time:.2f} requests/s,"
          f"{total_output_tokens / gen_time:.2f} tokens/s.")

    # Compute the latency statistics.
    avg_latency = np.mean([latency for _, _, latency in REQUEST_LATENCY])

    print(f"Average latency: {avg_latency:.2f} s,"
          f"Average generation latency: {avg_latency - avg_prompt_latency:.2f}")

    avg_per_output_token_latency = np.mean([
        (latency - avg_prompt_latency) / output_len
        for _, output_len, latency in REQUEST_LATENCY
    ])

    print("Average latency per output token: "
          f"{avg_per_output_token_latency*1000:.2f} ms")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark the online serving throughput.")
    parser.add_argument("--backend", type=str, default="taco_llm",
                        choices=["taco_llm", "tgi", "light_llm", "trt_llm"])
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--dataset", type=str, required=True,
                        help="Path to the dataset.")
    parser.add_argument("--tokenizer", type=str, required=True,
                        help="Name or path of the tokenizer.")
    parser.add_argument("--best-of", type=int, default=1,
                        help="Generates `best_of` sequences per prompt and "
                             "returns the best one.")
    parser.add_argument("--use-beam-search", action="store_true")
    parser.add_argument("--num-prompts", type=int, default=1000,
                        help="Number of prompts to process.")
    parser.add_argument("--request-rate", type=float, default=float("inf"),
                        help="Number of requests per second. If this is inf, "
                             "then all the requests are sent at time 0. "
                             "Otherwise, we use Poisson process to synthesize "
                             "the request arrival times.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument('--trust-remote-code', action='store_true',
                        help='trust remote code from huggingface')
    parser.add_argument("--min-prompt-len", type=int, default=4)
    parser.add_argument("--max-prompt-len", type=int, default=512)
    parser.add_argument("--min-gen-len", type=int, default=4)
    parser.add_argument("--max-gen-len", type=int, default=512)
    parser.add_argument("--gpu-sharemem-size", type=int, default=49152)
    args = parser.parse_args()
    main(args)
