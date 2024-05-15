import json

import requests

# url = "http://127.0.0.1:8002/generate"
text = '你好，如果我要去北京旅游，请帮我推荐景点？'
url = f"http://127.0.0.1:8081/generate"
use_stream = False
stream_return_new = True
headers = {"User-Agent": "Benchmark Client"}
pload = {
    "prompt": text,
    #"n": 1,
    #"best_of": 1,
    # "top_k":5,
    #"frequency_penalty":1.1,
    "use_beam_search": False,
    "temperature": 0,
    # "top_p": 0.85,
    "max_tokens": 100,
    "ignore_eos": False,
    "stream": use_stream,
    "incremental_return":stream_return_new,
}
import time
start_time= time.time()
response = requests.post(url,headers=headers, json=pload, stream=use_stream)



if use_stream:
    for chunk in response.iter_lines( delimiter=b'\x00'):

        text_str = chunk.decode("utf-8")
        # print("text is : {}".format(text_str))
        if text_str is not None and text_str != '' and text_str != ' ':
            
            json_obj = json.loads(text_str)
            word = json_obj['text']
            print(word, end='')
            # time.sleep(0.1)
            import sys
            # sys.stdout.flush()

else:
    print(response.json())
end_time= time.time()
print(end_time-start_time)