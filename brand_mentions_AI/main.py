import pandas as pd
from pydantic import BaseModel, Field
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.models.gemini import GeminiModel
from pydantic_ai.providers.google_gla import GoogleGLAProvider
from pydantic_ai import Agent
import asyncio
from dotenv import load_dotenv
import os

load_dotenv()


openai_model = OpenAIModel('gpt-4o-mini', provider=OpenAIProvider(api_key=os.getenv("OPENAI_API_KEY")))
gemini_model = GeminiModel('gemini-2.0-flash', provider=GoogleGLAProvider(api_key=os.getenv("GEMINI_API_KEY")))

class LLMResponse(BaseModel):
    response: str = Field(description='final response')

class LLMBrand(BaseModel):
    brand_mention: list = Field(description='Brands / platforms / business names mentioned in the response')


def load():
    return pd.read_csv("/mnt/c/Users/saidl/Downloads/LLM_Brand_Test_Questions.csv")

async def process_row(text, model):

    agent = Agent(model,
        output_type=LLMResponse,
        system_prompt="You are a helpful assistant that provides a detailed response to the question asked. ",
    )
    response = await agent.run(text)

    brand_agent = Agent(openai_model,
        output_type=LLMBrand,
        system_prompt="You are a helpful assistant capable of extracting brands, companies, businesses from text. For example, looking for things like 'Meta', 'Twitter', 'Salesforce', 'Nike', 'Logitech', 'Morrisons', 'Cadbury'. Not looking for things like 'Artificial Intelligence', 'Cloud based solutions', 'Fleet Management Systems', 'AI', 'Cloud', 'Fleet Management'.",
    )

    brand_response = await brand_agent.run(response.output.response)

    return brand_response.output.brand_mention, response.output.response


async def main():
    df = load() # Assume load() returns your DataFrame  
    df_sample = df.sample(2000, random_state=42).copy()
    # 1. Create the list of coroutines  
    tasks_openai = [process_row(text, openai_model) for text in df_sample['Question']]  
    tasks_gemini = [process_row(text, openai_model) for text in df_sample['Question']]  

    # 2. Run them concurrently and gather results  
    # results will be like: [(bm1, r1), (bm2, r2), (bm3, r3), (bm4, r4), (bm5, r5)]  
    print("Starting async processing...")  
    results_openai  = await asyncio.gather(*tasks_openai)  
    results_gemini  = await asyncio.gather(*tasks_gemini)  
    print("Async processing finished.")  

    # 3. Unzip the list of tuples into two separate lists/tuples  
    # zip(*results) effectively transposes the list of tuples  
    # map(list, ...) converts the resulting tuples from zip into lists  
    brand_mentions_list_openai, responses_list_openai = map(list, zip(*results_openai))  
    brand_mentions_list_gemini, responses_list_gemini = map(list, zip(*results_gemini))  

    # 4. Assign the separated lists to the DataFrame columns  
    df_sample['brand_mentions_gpt4o'] = brand_mentions_list_openai
    df_sample['brand_mentions_gemini'] = brand_mentions_list_gemini
    df_sample['response_gpt4o'] = responses_list_openai
    df_sample['response_gemini'] = responses_list_gemini  
    

    df_sample.to_csv("/mnt/c/Users/saidl/Downloads/responses.csv", index=False)

    return 'Done'

if __name__ == "__main__":
    print(asyncio.run(main()))
