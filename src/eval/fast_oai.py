import openai
import os
from dotenv import load_dotenv
from typing import List, Optional, Tuple
import asyncio
from asyncio import Semaphore, Lock
import logging
from time import time, sleep


dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env')
load_dotenv(dotenv_path)


openai.api_key = os.getenv("OPENAI_API_KEY")###requires adding openai_api_key as environment variable manually
if openai.api_key is None:
    logging.warning("openai.api_key is None")

start = None
num_requests = 0

class TokenBucket:
    def __init__(self, rate: int):
        # rate is in requests per second
        self._rate = rate
        self._capacity = rate
        self._tokens = self._capacity
        self._last_refill = time()

    async def consume(self):
        while self._tokens < 1:
            self._refill()
            await asyncio.sleep(1)  # Sleep for some time before trying again
        self._tokens -= 1

    def _refill(self):
        now = time()
        time_passed = now - self._last_refill
        refill_amount = time_passed * self._rate
        self._tokens = min(self._capacity, self._tokens + refill_amount)
        self._last_refill = now

MaybeTokenBucket = Optional[TokenBucket]


async def _call_chat(system_prompt: str,
                     user_prompt:str,
                     temperature: float=1.,
                     token_bucket: MaybeTokenBucket=None,
                     timeout: int=20,
                     max_retries=5,
                     model="gpt-3.5-turbo") -> str:
    done = False
    messages= [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    backoff = 1
    retries = 0
    while not done:
        try:
            if token_bucket is not None:
                await token_bucket.consume()
            response = await asyncio.wait_for(openai.ChatCompletion.acreate(
                model=model,
                messages=messages,
                ), timeout=timeout)
            completion = response.choices[0].message.content
            total_tokens = response.usage.total_tokens
            done=True
        except asyncio.TimeoutError as e:
            if backoff > 128:
                print(f"Failed to call chat after {backoff} seconds due to  {e}")
                completion = None
                total_tokens = 0
                done = True
            await asyncio.sleep(backoff)
            backoff *= 2
        except Exception as e:
            await asyncio.sleep(backoff)
            backoff *= 2
            backoff = min(backoff, 64)
            retries += 1
            if retries >= max_retries:
                print(f"Failed to call chat after {retries} retries due to  {e}:\n\nMessages:{messages}\n\n")
                completion = None
                total_tokens = 0
                done = True
    return completion, total_tokens


async def _handle_chat(system_prompt: str, user_prompt: str,
                       token_bucket: TokenBucket, semaphore: Semaphore, lock: Lock, results_counter: dict,
                       model: str, timeout: int, temperature: float) -> str:
    async with semaphore:
        completion, toks = await _call_chat(system_prompt=system_prompt,
                                            user_prompt=user_prompt,
                                            temperature=temperature,
                                            token_bucket=token_bucket,
                                            timeout=timeout,
                                            model=model)

    async with lock:  # Ensure atomicity of operations
        results_counter['num_requests'] += 1
        results_counter['tokens'] += toks
        print_period = 10
        if results_counter['num_requests'] % print_period == 0:
            duration = time() - results_counter['start_time']
            duration_min = duration / 60
            cost = results_counter['cost_per_ktok'] * results_counter['tokens'] / 1000
            print(f"{results_counter['num_requests']=}, {duration=:.2f} rate per min={results_counter['num_requests'] / duration_min:.2f} tokens / request: {results_counter['tokens'] / results_counter['num_requests']:.2f} cost: {cost:.2f}")

    return completion


def call_chats(prompts: List[Tuple[str, str]],
               model: str="gpt-3.5-turbo",
               timeout: int=10,
               temperature: float=1.) -> List[str]:
    # prompts should be [(system_prompt, user_prompt), ...]
    max_concurrent_tasks = 20
    oai_quotas = {'gpt-3.5-turbo': 150, 'gpt-3.5-turbo-16k': 200, 'gpt-4': 200}
    oai_costs_per_ktok = {'gpt-3.5-turbo': 0.0015, 'gpt-3.5-turbo-16k': 0.003, 'gpt-4': 0.03}
    oai_quota_per_minute = oai_quotas[model]
    oai_quota_per_second = oai_quota_per_minute // 60
    semaphore = Semaphore(max_concurrent_tasks)
    token_bucket = TokenBucket(oai_quota_per_second)
    lock = Lock()
    results_counter = {'num_requests': 0, 'start_time': time(), 'tokens': 0, 'cost_per_ktok': oai_costs_per_ktok[model]}
    async def gather_tasks():
        tasks = [_handle_chat(system_prompt, user_prompt, token_bucket, semaphore,
                              lock, results_counter,
                              model, timeout, temperature) for system_prompt, user_prompt in prompts]
        return await asyncio.gather(*tasks)
    return asyncio.run(gather_tasks())


def test_chats():
    fun_sentences = [
    "The sky is blue today.",
    "Ducks quack to communicate.",
    "Bananas are my favorite fruit.",
    "Chocolate makes everything better.",
    "Singing in the rain is fun.",
    "Cats have nine lives, they say.",
    "The moon is made of cheese.",
    "Robots will take over the world.",
    "Pineapples belong on pizza.",
    "Unicorns are just horses with a twist."
    ]
    system_prompts = ["You are a pig-latinifiying bot. Please reproduce the user message in Pig Latin"] * len(fun_sentences)

    completions = call_chats(list(zip(system_prompts, fun_sentences)))
    for completion, fun_sentence in zip(completions, fun_sentences):
        print(f"{fun_sentence=}, {completion=}")


if __name__ == '__main__':
    test_chats()