import asyncio
import functools
from collections.abc import Callable

from ..exceptions import APIError, ImageGenerationError
from .logger import logger


def running(retry: int = 0) -> Callable:
    """
    Decorator to check if GeminiClient is running before making a request.

    Parameters
    ----------
    retry: `int`, optional
        Max number of retries when `gemini_webapi.APIError` is raised.
    """

    def decorator(func):
        @functools.wraps(func)
        async def wrapper(client, *args, retry=retry, **kwargs):
            try:
                if not client._running:
                    logger.debug(f"[running] Client not running. Re-initializing before {func.__name__}...")
                    await client.init(
                        timeout=client.timeout,
                        auto_close=client.auto_close,
                        close_delay=client.close_delay,
                        auto_refresh=client.auto_refresh,
                        refresh_interval=client.refresh_interval,
                        verbose=True,
                    )
                    if client._running:
                        return await func(client, *args, **kwargs)

                    # Should not reach here
                    raise APIError(
                        f"Invalid function call: GeminiClient.{func.__name__}. Client initialization failed."
                    )
                else:
                    return await func(client, *args, **kwargs)
            except APIError as e:
                # Image generation takes too long, only retry once
                if isinstance(e, ImageGenerationError):
                    retry = min(1, retry)

                if retry > 0:
                    logger.warning(f"[running] APIError: {e}. Retrying {func.__name__} ({retry} retries left)...")
                    await asyncio.sleep(1)
                    return await wrapper(client, *args, retry=retry - 1, **kwargs)

                raise

        return wrapper

    return decorator
