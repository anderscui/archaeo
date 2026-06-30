# coding=utf-8
import os

from dotenv import load_dotenv
from openai import OpenAI

os.environ["http_proxy"] = "http://127.0.0.1:10808"
os.environ["https_proxy"] = "http://127.0.0.1:10808"

load_dotenv()
client = OpenAI(
    api_key=os.environ['OPENAI_API_KEY'],
    base_url='https://api.legions.cc/v1'
)

try:
    response = client.responses.create(
        model='gpt-5.5',
        input='Say OK only.'
    )
    print(response.output_text)
    print('OpenAI API key works.')
except Exception as e:
    print('OpenAI API key failed.')
    print(type(e).__name__, e)
