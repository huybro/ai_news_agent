import asyncpg
import os
from dotenv import load_dotenv

load_dotenv()
POSTGRES_URI = os.getenv("POSTGRES_URI")

async def test_connection():
    conn = await asyncpg.connect(POSTGRES_URI)
    result = await conn.fetchval("SELECT current_schema()")
    print(f"Current schema: {result}")
    await conn.close()

import asyncio
asyncio.run(test_connection())