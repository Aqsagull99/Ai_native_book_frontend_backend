import asyncio
import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()

print('OPENROUTER_API_KEY:', os.getenv('OPENROUTER_API_KEY', 'NOT SET')[:30] + '...')
print('OPENROUTER_MODEL:', os.getenv('OPENROUTER_MODEL', 'NOT SET'))

from src.services.translation_agent import translation_agent

async def test():
    print('Testing translation agent...')
    result = await translation_agent.translate_content('Hello, how are you?', 'test_user', 'test_chapter')
    print(f'Result: {result}')

asyncio.run(test())
