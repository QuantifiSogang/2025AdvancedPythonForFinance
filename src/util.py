import os
import openai

def set_api() :
    with open('../config/api.key') as file :
        lines = file.readlines()
        api_key = lines[0].strip()
        serp_api_key = lines[1].strip()
        langsmith_api_key = lines[2].strip()

    openai.api_key = api_key
    os.environ['OPENAI_API_KEY'] = openai.api_key