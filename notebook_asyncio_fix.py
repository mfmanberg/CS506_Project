# Windows Asyncio Fix for Jupyter Notebooks
# Add this as the FIRST cell in notebooks that crash with "Kernel died"

import sys
if sys.platform == 'win32':
    import asyncio
    # Fix Windows ProactorEventLoop issues
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    print("Windows asyncio policy set to WindowsSelectorEventLoopPolicy")

# Memory optimization
import gc
gc.collect()

print("Kernel startup optimizations applied")
