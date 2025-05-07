import pandas as pd
from pydantic import BaseModel, Field
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai import Agent
import asyncio
from dotenv import load_dotenv
import os

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai_model = OpenAIModel('gpt-4o-mini', provider=OpenAIProvider(api_key=OPENAI_API_KEY))

class LLMResponse(BaseModel):
    response: str = Field(description='final response')

class LLMBrand(BaseModel):
    brand_mention: list = Field(description='Brands / platforms / business names mentioned in the response')


def load():
    return pd.read_csv("/mnt/c/Users/saidl/Downloads/drfsearch.csv")

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
    df_sample = df.sample(10, random_state=42).copy()
    # 1. Create the list of coroutines  
    tasks = [process_row(text, openai_model) for text in df['Question']]  

    # 2. Run them concurrently and gather results  
    # results will be like: [(bm1, r1), (bm2, r2), (bm3, r3), (bm4, r4), (bm5, r5)]  
    print("Starting async processing...")  
    results = await asyncio.gather(*tasks)  
    print("Async processing finished.")  

    # 3. Unzip the list of tuples into two separate lists/tuples  
    # zip(*results) effectively transposes the list of tuples  
    # map(list, ...) converts the resulting tuples from zip into lists  
    brand_mentions_list, responses_list = map(list, zip(*results))  

    # 4. Assign the separated lists to the DataFrame columns  
    df['brand_mentions_gpt4o'] = brand_mentions_list  
    df['response_gpt4o'] = responses_list  

    df.to_csv("/mnt/c/Users/saidl/Downloads/responses.csv", index=False)

    return 'Done'

if __name__ == "__main__":
    print(asyncio.run(main()))
