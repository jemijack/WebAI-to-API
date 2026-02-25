import asyncio
import re
from asyncio import Task
from pathlib import Path
from typing import Any, Optional

import orjson as json
from curl_cffi.requests import AsyncSession
from curl_cffi.requests.exceptions import Timeout as ReadTimeout

from .components import GemMixin
from .constants import Endpoint, ErrorCode, Headers, Model
from .exceptions import (
    APIError,
    AuthError,
    GeminiError,
    ImageGenerationError,
    ModelInvalid,
    TemporarilyBlocked,
    TimeoutError,
    UsageLimitExceeded,
)
from .types import (
    Candidate,
    Gem,
    GeneratedImage,
    ModelOutput,
    RPCData,
    WebImage,
)
from .utils import (
    extract_json_from_response,
    get_access_token,
    get_nested_value,
    logger,
    parse_file_name,
    rotate_1psidts,
    rotate_tasks,
    running,
    upload_file,
)


class GeminiClient(GemMixin):
    """
    Async client interface for gemini.google.com.

    `secure_1psid` must be provided unless the optional dependency `browser-cookie3` is installed, and
    you have logged in to google.com in your local browser.

    Parameters
    ----------
    secure_1psid: `str`, optional
        __Secure-1PSID cookie value.
    secure_1psidts: `str`, optional
        __Secure-1PSIDTS cookie value, some google accounts don't require this value, provide only if it's in the cookie list.
    proxy: `str`, optional
        Proxy URL.
    kwargs: `dict`, optional
        Additional arguments (reserved for compatibility).
    """

    __slots__ = [
        "cookies",
        "proxy",
        "_running",
        "client",
        "access_token",
        "timeout",
        "auto_close",
        "close_delay",
        "close_task",
        "auto_refresh",
        "refresh_interval",
        "_gems",  # From GemMixin
        "kwargs",
    ]

    def __init__(
        self,
        secure_1psid: str | None = None,
        secure_1psidts: str | None = None,
        proxy: str | None = None,
        **kwargs,
    ):
        super().__init__()
        self.cookies = {}
        self.proxy = proxy
        self._running: bool = False
        self.client: AsyncSession | None = None
        self.access_token: str | None = None
        self.timeout: float = 300
        self.auto_close: bool = False
        self.close_delay: float = 300
        self.close_task: Task | None = None
        self.auto_refresh: bool = True
        self.refresh_interval: float = 540
        self.kwargs = kwargs

        if secure_1psid:
            self.cookies["__Secure-1PSID"] = secure_1psid
            if secure_1psidts:
                self.cookies["__Secure-1PSIDTS"] = secure_1psidts

    async def init(
        self,
        timeout: float = 300,
        auto_close: bool = False,
        close_delay: float = 300,
        auto_refresh: bool = True,
        refresh_interval: float = 540,
        verbose: bool = True,
    ) -> None:
        """
        Get SNlM0e value as access token. Without this token posting will fail with 400 bad request.
        """

        try:
            access_token, valid_cookies = await get_access_token(
                base_cookies=self.cookies, proxy=self.proxy, verbose=verbose
            )

            self.client = AsyncSession(
                timeout=timeout,
                proxy=self.proxy,
                allow_redirects=True,
                headers=Headers.GEMINI.value,
                cookies=valid_cookies,
                impersonate="chrome",
            )
            self.access_token = access_token
            self.cookies = valid_cookies
            self._running = True

            self.timeout = timeout
            self.auto_close = auto_close
            self.close_delay = close_delay
            if self.auto_close:
                await self.reset_close_task()

            self.auto_refresh = auto_refresh
            self.refresh_interval = refresh_interval
            if task := rotate_tasks.get(self.cookies["__Secure-1PSID"]):
                task.cancel()
            if self.auto_refresh:
                rotate_tasks[self.cookies["__Secure-1PSID"]] = asyncio.create_task(
                    self.start_auto_refresh()
                )

            if verbose:
                logger.success("Gemini client initialized successfully.")
        except Exception:
            await self.close()
            raise

    async def close(self, delay: float = 0) -> None:
        if delay:
            await asyncio.sleep(delay)

        self._running = False

        if self.close_task:
            self.close_task.cancel()
            self.close_task = None

        if self.client:
            await self.client.aclose()

    async def reset_close_task(self) -> None:
        if self.close_task:
            self.close_task.cancel()
            self.close_task = None

        self.close_task = asyncio.create_task(self.close(self.close_delay))

    async def start_auto_refresh(self) -> None:
        while True:
            new_1psidts: str | None = None
            try:
                new_1psidts = await rotate_1psidts(self.cookies, self.proxy)
            except AuthError:
                if task := rotate_tasks.get(self.cookies.get("__Secure-1PSID", "")):
                    task.cancel()
                logger.warning(
                    "AuthError: Failed to refresh cookies. Auto refresh task canceled."
                )
                return
            except Exception as exc:
                logger.warning(f"Unexpected error while refreshing cookies: {exc}")

            if new_1psidts:
                self.cookies["__Secure-1PSIDTS"] = new_1psidts
                if self._running:
                    self.client.cookies.set("__Secure-1PSIDTS", new_1psidts)
                logger.debug("Cookies refreshed. New __Secure-1PSIDTS applied.")

            await asyncio.sleep(self.refresh_interval)

    @running(retry=2)
    async def generate_content(
        self,
        prompt: str,
        files: list[str | Path] | None = None,
        model: Model | str | dict = Model.UNSPECIFIED,
        gem: Gem | str | None = None,
        chat: Optional["ChatSession"] = None,
        **kwargs,
    ) -> ModelOutput:
        """
        Generates contents with prompt.
        Each call uses a fresh HTTP session to avoid HTTP/2 connection state issues
        with Pro-tier models.
        """

        assert prompt, "Prompt cannot be empty."

        if isinstance(model, str):
            model = Model.from_name(model)
        elif isinstance(model, dict):
            model = Model.from_dict(model)
        elif not isinstance(model, Model):
            raise TypeError(
                f"'model' must be a `gemini_webapi.constants.Model` instance, "
                f"string, or dictionary; got `{type(model).__name__}`"
            )

        if isinstance(gem, Gem):
            gem_id = gem.id
        else:
            gem_id = gem

        if self.auto_close:
            await self.reset_close_task()

        logger.debug(
            f"[generate_content] model={model.model_name!r} "
            f"model_header={model.model_header} "
            f"access_token_present={bool(self.access_token)} "
            f"endpoint={Endpoint.GENERATE.value}"
        )

        try:
            async with AsyncSession(
                timeout=self.timeout,
                proxy=self.proxy,
                allow_redirects=True,
                headers=Headers.GEMINI.value,
                cookies=self.cookies,
                impersonate="chrome",
            ) as session:
                response = await session.post(
                    Endpoint.GENERATE.value,
                    headers=model.model_header,
                    data={
                        "at": self.access_token,
                        "f.req": json.dumps(
                            [
                                None,
                                json.dumps(
                                    [
                                        files
                                        and [
                                            prompt,
                                            0,
                                            None,
                                            [
                                                [
                                                    [await upload_file(file, self.proxy)],
                                                    parse_file_name(file),
                                                ]
                                                for file in files
                                            ],
                                        ]
                                        or [prompt],
                                        None,
                                        chat and chat.metadata,
                                    ]
                                    + (gem_id and [None] * 16 + [gem_id] or [])
                                ).decode(),
                            ]
                        ).decode(),
                    },
                )
        except ReadTimeout:
            raise TimeoutError(
                "Generate content request timed out, please try again. If the problem persists, "
                "consider setting a higher `timeout` value when initializing GeminiClient."
            )

        logger.debug(
            f"[generate_content] response status={response.status_code} "
            f"body_preview={response.text[:600]!r}"
        )

        if response.status_code != 200:
            await self.close()
            raise APIError(
                f"Failed to generate contents. Request failed with status code {response.status_code}"
            )
        else:
            response_json: list[Any] = []
            body: list[Any] = []
            body_index = 0

            try:
                response_json = extract_json_from_response(response.text)

                for part_index, part in enumerate(response_json):
                    try:
                        part_body = get_nested_value(part, [2])
                        if not part_body:
                            continue

                        part_json = json.loads(part_body)
                        if get_nested_value(part_json, [4]):
                            body_index, body = part_index, part_json
                            break
                    except json.JSONDecodeError:
                        continue

                if not body:
                    raise Exception
            except Exception:
                await self.close()

                try:
                    error_code = get_nested_value(response_json, [0, 5, 2, 0, 1, 0], -1)
                    logger.debug(
                        f"[generate_content] parse failed. "
                        f"error_code={error_code} "
                        f"full_response={response.text!r}"
                    )
                    match error_code:
                        case ErrorCode.USAGE_LIMIT_EXCEEDED:
                            raise UsageLimitExceeded(
                                f"Failed to generate contents. Usage limit of {model.model_name} model has exceeded. Please try switching to another model."
                            )
                        case ErrorCode.MODEL_INCONSISTENT:
                            raise ModelInvalid(
                                "Failed to generate contents. The specified model is inconsistent with the chat history. Please make sure to pass the same "
                                "`model` parameter when starting a chat session with previous metadata."
                            )
                        case ErrorCode.MODEL_HEADER_INVALID:
                            raise ModelInvalid(
                                "Failed to generate contents. The specified model is not available. Please update gemini_webapi to the latest version. "
                                "If the error persists and is caused by the package, please report it on GitHub."
                            )
                        case ErrorCode.IP_TEMPORARILY_BLOCKED:
                            raise TemporarilyBlocked(
                                "Failed to generate contents. Your IP address is temporarily blocked by Google. Please try using a proxy or waiting for a while."
                            )
                        case _:
                            raise Exception
                except GeminiError:
                    raise
                except Exception:
                    logger.debug(f"Invalid response: {response.text}")
                    raise APIError(
                        "Failed to generate contents. Invalid response data received. Client will try to re-initialize on next request."
                    )

            try:
                candidate_list: list[Any] = get_nested_value(body, [4], [])
                output_candidates: list[Candidate] = []

                for candidate_index, candidate in enumerate(candidate_list):
                    rcid = get_nested_value(candidate, [0])
                    if not rcid:
                        continue  # Skip candidate if it has no rcid

                    # Text output and thoughts
                    text = get_nested_value(candidate, [1, 0], "")
                    if re.match(
                        r"^http://googleusercontent\.com/card_content/\d+", text
                    ):
                        text = get_nested_value(candidate, [22, 0]) or text

                    thoughts = get_nested_value(candidate, [37, 0, 0])

                    # Web images
                    web_images = []
                    for web_img_data in get_nested_value(candidate, [12, 1], []):
                        url = get_nested_value(web_img_data, [0, 0, 0])
                        if not url:
                            continue

                        web_images.append(
                            WebImage(
                                url=url,
                                title=get_nested_value(web_img_data, [7, 0], ""),
                                alt=get_nested_value(web_img_data, [0, 4], ""),
                                proxy=self.proxy,
                            )
                        )

                    # Generated images
                    generated_images = []
                    if get_nested_value(candidate, [12, 7, 0]):
                        img_body = None
                        for img_part_index, part in enumerate(response_json):
                            if img_part_index < body_index:
                                continue
                            try:
                                img_part_body = get_nested_value(part, [2])
                                if not img_part_body:
                                    continue

                                img_part_json = json.loads(img_part_body)
                                if get_nested_value(
                                    img_part_json, [4, candidate_index, 12, 7, 0]
                                ):
                                    img_body = img_part_json
                                    break
                            except json.JSONDecodeError:
                                continue

                        if not img_body:
                            raise ImageGenerationError(
                                "Failed to parse generated images. Please update gemini_webapi to the latest version. "
                                "If the error persists and is caused by the package, please report it on GitHub."
                            )

                        img_candidate = get_nested_value(
                            img_body, [4, candidate_index], []
                        )

                        if finished_text := get_nested_value(
                            img_candidate, [1, 0]
                        ):  # Only overwrite if new text is returned after image generation
                            text = re.sub(
                                r"http://googleusercontent\.com/image_generation_content/\d+",
                                "",
                                finished_text,
                            ).rstrip()

                        for img_index, gen_img_data in enumerate(
                            get_nested_value(img_candidate, [12, 7, 0], [])
                        ):
                            url = get_nested_value(gen_img_data, [0, 3, 3])
                            if not url:
                                continue

                            img_num = get_nested_value(gen_img_data, [3, 6])
                            title = (
                                f"[Generated Image {img_num}]"
                                if img_num
                                else "[Generated Image]"
                            )

                            alt_list = get_nested_value(gen_img_data, [3, 5], [])
                            alt = (
                                get_nested_value(alt_list, [img_index])
                                or get_nested_value(alt_list, [0])
                                or ""
                            )

                            generated_images.append(
                                GeneratedImage(
                                    url=url,
                                    title=title,
                                    alt=alt,
                                    proxy=self.proxy,
                                    cookies=self.cookies,
                                )
                            )

                    output_candidates.append(
                        Candidate(
                            rcid=rcid,
                            text=text,
                            thoughts=thoughts,
                            web_images=web_images,
                            generated_images=generated_images,
                        )
                    )

                if not output_candidates:
                    raise GeminiError(
                        "Failed to generate contents. No output data found in response."
                    )

                output = ModelOutput(
                    metadata=get_nested_value(body, [1], []),
                    candidates=output_candidates,
                )
            except (TypeError, IndexError) as e:
                logger.debug(
                    f"{type(e).__name__}: {e}; Invalid response structure: {response.text}"
                )
                raise APIError(
                    "Failed to parse response body. Data structure is invalid."
                )

            if isinstance(chat, ChatSession):
                chat.last_output = output

            return output

    def start_chat(self, **kwargs) -> "ChatSession":
        return ChatSession(geminiclient=self, **kwargs)

    async def _batch_execute(self, payloads: list[RPCData], **kwargs):
        try:
            response = await self.client.post(
                Endpoint.BATCH_EXEC,
                data={
                    "at": self.access_token,
                    "f.req": json.dumps(
                        [[payload.serialize() for payload in payloads]]
                    ).decode(),
                },
                **kwargs,
            )
        except ReadTimeout:
            raise TimeoutError(
                "Batch execute request timed out, please try again. If the problem persists, "
                "consider setting a higher `timeout` value when initializing GeminiClient."
            )

        if response.status_code != 200:
            await self.close()
            raise APIError(
                f"Batch execution failed with status code {response.status_code}"
            )

        return response


class ChatSession:
    __slots__ = [
        "__metadata",
        "geminiclient",
        "last_output",
        "model",
        "gem",
    ]

    def __init__(
        self,
        geminiclient: GeminiClient,
        metadata: list[str | None] | None = None,
        cid: str | None = None,
        rid: str | None = None,
        rcid: str | None = None,
        model: Model | str | dict = Model.UNSPECIFIED,
        gem: Gem | str | None = None,
    ):
        self.__metadata: list[str | None] = [None, None, None]
        self.geminiclient: GeminiClient = geminiclient
        self.last_output: ModelOutput | None = None
        self.model: Model | str | dict = model
        self.gem: Gem | str | None = gem

        if metadata:
            self.metadata = metadata
        if cid:
            self.cid = cid
        if rid:
            self.rid = rid
        if rcid:
            self.rcid = rcid

    def __str__(self):
        return f"ChatSession(cid='{self.cid}', rid='{self.rid}', rcid='{self.rcid}')"

    __repr__ = __str__

    def __setattr__(self, name: str, value: Any) -> None:
        super().__setattr__(name, value)
        if name == "last_output" and isinstance(value, ModelOutput):
            self.metadata = value.metadata
            self.rcid = value.rcid

    async def send_message(
        self,
        prompt: str,
        files: list[str | Path] | None = None,
        **kwargs,
    ) -> ModelOutput:
        return await self.geminiclient.generate_content(
            prompt=prompt,
            files=files,
            model=self.model,
            gem=self.gem,
            chat=self,
            **kwargs,
        )

    def choose_candidate(self, index: int) -> ModelOutput:
        if not self.last_output:
            raise ValueError("No previous output data found in this chat session.")

        if index >= len(self.last_output.candidates):
            raise ValueError(
                f"Index {index} exceeds the number of candidates in last model output."
            )

        self.last_output.chosen = index
        self.rcid = self.last_output.rcid
        return self.last_output

    @property
    def metadata(self):
        return self.__metadata

    @metadata.setter
    def metadata(self, value: list[str]):
        if len(value) > 3:
            raise ValueError("metadata cannot exceed 3 elements")
        self.__metadata[: len(value)] = value

    @property
    def cid(self):
        return self.__metadata[0]

    @cid.setter
    def cid(self, value: str):
        self.__metadata[0] = value

    @property
    def rid(self):
        return self.__metadata[1]

    @rid.setter
    def rid(self, value: str):
        self.__metadata[1] = value

    @property
    def rcid(self):
        return self.__metadata[2]

    @rcid.setter
    def rcid(self, value: str):
        self.__metadata[2] = value
