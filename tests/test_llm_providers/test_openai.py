# coding=utf-8
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(
    api_key=os.environ['OPENAI_API_KEY'],
    base_url=os.environ['OPENAI_BASE_URL']
)


def test_openai_basic():
    try:
        # 5.5
        response = client.responses.create(
            model='gpt-5.5',
            input='Say OK only.'
        )
        print(response.output_text)
        print('OpenAI API key works.')
    except Exception as e:
        print('OpenAI API key failed.')
        print(type(e).__name__, e)
