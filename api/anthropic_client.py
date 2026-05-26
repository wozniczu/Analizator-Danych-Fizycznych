##
## @file anthropic_client.py
## @brief Klient Anthropic Claude (Messages API): streaming, tool use, Code execution i Files API.
##

import io
from typing import List, Dict, Any, Generator, Optional, Tuple
from anthropic import Anthropic, AnthropicError
from api.base import APIClient, APIResponse, StreamChunk, APIError
from utils.logger import logger

try:
    from config.api_config import MODELS
except ImportError:
    MODELS = {"anthropic": ["claude-sonnet-4.5", "claude-opus-4.5", "claude-haiku-4.5"]}

CODE_EXECUTION_BETA = "code-execution-2025-08-25"
CODE_EXECUTION_TOOL = {"type": "code_execution_20250825", "name": "code_execution"}
FILES_API_BETA = "files-api-2025-04-14"


def _normalize_model_name(model: str) -> str:
    """!
    @brief Normalizuje nazwę modelu (kropka w wersji → myślnik, np. 4.5 → 4-5).

    @param model Nazwa modelu (np. claude-sonnet-4.5).
    @return Nazwa z poprawioną wersją dla API.
    """
    if ".5" in model or ".6" in model:
        return model.replace(".5", "-5").replace(".6", "-6")
    return model


def _extract_system_and_messages(messages: List[Dict[str, Any]]) -> tuple:
    """!
    @brief Rozdziela listę wiadomości na prompt systemowy i listę user/assistant.

    @param messages Lista słowników z kluczami role, content.
    @return Krotka (system_prompt, user_messages).
    """
    system_prompt = None
    user_messages = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "system":
            system_prompt = content if isinstance(content, str) else None
            continue
        if role in ("user", "assistant"):
            user_messages.append({"role": role, "content": content})
    return system_prompt, user_messages


def _messages_contain_container_upload(user_messages: List[Dict[str, Any]]) -> bool:
    """!
    @brief Sprawdza, czy w treści wiadomości występuje blok container_upload (Files API).

    @param user_messages Lista wiadomości user/assistant.
    @return True, jeśli wykryto container_upload.
    """
    for msg in user_messages:
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            typ = block.get("type") if isinstance(block, dict) else getattr(block, "type", None)
            if typ == "container_upload":
                return True
    return False


def _openai_tool_to_anthropic(tool: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """!
    @brief Konwertuje definicję narzędzia z formatu OpenAI (function) na Anthropic (name, input_schema).

    @param tool Słownik narzędzia w formacie OpenAI.
    @return Słownik w formacie Anthropic lub None przy braku name.
    """
    if not isinstance(tool, dict):
        return None
    if tool.get("type") == "function" and "function" in tool:
        fn = tool["function"]
        name = fn.get("name")
        if not name:
            return None
        params = fn.get("parameters") or {}
        return {
            "name": name,
            "description": fn.get("description") or "",
            "input_schema": {
                "type": params.get("type", "object"),
                "properties": params.get("properties", {}),
                "required": params.get("required", []),
            },
        }
    return None


def _prepare_anthropic_tools(tools: Any) -> Optional[List[Dict[str, Any]]]:
    """!
    @brief Przygotowuje listę narzędzi dla Messages API: code_interpreter → None (use_code_execution), function → konwersja, Anthropic → bez zmian.

    @param tools Lista narzędzi (OpenAI lub Anthropic).
    @return Lista w formacie Anthropic lub None.
    """
    if not tools or not isinstance(tools, list):
        return None
    out = []
    for t in tools:
        if not isinstance(t, dict):
            continue
        if t.get("type") == "code_interpreter" and "container" in t:
            return None
        if t.get("type") == "function":
            converted = _openai_tool_to_anthropic(t)
            if converted:
                out.append(converted)
        elif t.get("name") and t.get("input_schema"):
            out.append(t)
    return out if out else None


def _extract_text_from_content(content: List[Any]) -> str:
    """!
    @brief Skleja tekst z bloków content (typ text oraz ewentualne atrybuty .text).

    @param content Lista bloków odpowiedzi (obiekty SDK lub dict).
    @return Złączony tekst.
    """
    parts = []
    for block in content or []:
        if hasattr(block, "text") and block.text:
            parts.append(block.text)
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", "") or "")
    return "".join(parts)


def _extract_thinking_from_content(content: List[Any]) -> str:
    """!
    @brief Zbiera treść bloków type=thinking (Extended Thinking) w jeden łańcuch.

    @param content Lista bloków odpowiedzi.
    @return Złączony tekst rozumowania lub pusty string.
    """
    parts = []
    for block in content or []:
        typ = _block_type(block)
        if typ != "thinking":
            continue
        thinking = _block_attr(block, "thinking", "") or ""
        if isinstance(thinking, str) and thinking.strip():
            parts.append(thinking.strip())
    return "\n\n".join(parts) if parts else ""


def _adjust_thinking_for_max_tokens(
    max_tokens: int,
    thinking: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """!
    @brief Dopasowuje budget_tokens do limitu max_tokens (wymóg API: max_tokens > budget_tokens).

    @param max_tokens Limit tokenów odpowiedzi w bieżącym wywołaniu.
    @param thinking Parametr thinking z kwargs lub None.
    @return Skorygowany słownik thinking albo None, gdy thinking nie może być użyty.
    """
    if not thinking or not isinstance(thinking, dict):
        return thinking
    if thinking.get("type") != "enabled":
        return thinking
    try:
        budget = int(thinking.get("budget_tokens", 0))
    except (TypeError, ValueError):
        return thinking
    if max_tokens > budget:
        return thinking
    if max_tokens <= 1:
        logger.warning(
            "Extended Thinking pominięte: max_tokens=%s za małe",
            max_tokens,
        )
        return None
    adjusted_budget = max_tokens - 1
    logger.debug(
        f"Obniżono thinking.budget_tokens z {budget} do {adjusted_budget} "
        f"(max_tokens={max_tokens})"
    )
    return {**thinking, "budget_tokens": adjusted_budget}


def _block_type(block: Any) -> Optional[str]:
    """!
    @brief Zwraca typ bloku niezależnie od reprezentacji (obiekt SDK lub dict).

    @param block Pojedynczy blok content.
    @return Wartość type lub None.
    """
    if hasattr(block, "type"):
        return getattr(block, "type", None)
    if isinstance(block, dict):
        return block.get("type")
    return None


def _block_attr(block: Any, key: str, default: Any = None) -> Any:
    """!
    @brief Pobiera atrybut lub klucz z bloku (SDK lub dict).

    @param block Blok content.
    @param key Nazwa atrybutu/klucza.
    @param default Wartość przy braku klucza.
    @return Wartość atrybutu lub default.
    """
    if hasattr(block, key):
        return getattr(block, key, default)
    if isinstance(block, dict):
        return block.get(key, default)
    return default


def _extract_code_execution_from_content(content: List[Any]) -> Dict[str, Any]:
    """!
    @brief Z bloków server_tool_use i *_tool_result wyciąga kod, stdout/stderr i listę file_id (Files API).

    @param content Lista bloków odpowiedzi.
    @return Słownik: generated_code, execution_output, output_file_ids.
    """
    code_parts = []
    output_parts = []
    output_file_ids: List[Tuple[str, str]] = []

    for block in content or []:
        typ = _block_type(block)
        if not typ:
            continue
        if typ == "server_tool_use":
            inp = _block_attr(block, "input")
            if inp is not None:
                cmd = _block_attr(inp, "command") or _block_attr(inp, "code")
                if cmd:
                    code_parts.append(cmd if isinstance(cmd, str) else str(cmd))
                file_text = _block_attr(inp, "file_text")
                if file_text:
                    code_parts.append(file_text if isinstance(file_text, str) else str(file_text))
        elif typ == "bash_code_execution_tool_result":
            cnt = _block_attr(block, "content")
            if isinstance(cnt, dict):
                stdout = cnt.get("stdout") or ""
                stderr = cnt.get("stderr") or ""
                if stdout:
                    output_parts.append(stdout)
                if stderr:
                    output_parts.append(f"[stderr]\n{stderr}")
                inner = cnt.get("content") if isinstance(cnt.get("content"), list) else []
                for file_obj in inner:
                    fid = file_obj.get("file_id") if isinstance(file_obj, dict) else getattr(file_obj, "file_id", None)
                    fname = file_obj.get("filename", "") if isinstance(file_obj, dict) else getattr(file_obj, "filename", "") or ""
                    if fid:
                        output_file_ids.append((fid, fname or fid))
            elif hasattr(cnt, "stdout"):
                if getattr(cnt, "stdout", None):
                    output_parts.append(getattr(cnt, "stdout", ""))
                if getattr(cnt, "stderr", None):
                    output_parts.append(f"[stderr]\n{getattr(cnt, 'stderr', '')}")
                inner = getattr(cnt, "content", None)
                if isinstance(inner, list):
                    for file_obj in inner:
                        fid = getattr(file_obj, "file_id", None) or (file_obj.get("file_id") if isinstance(file_obj, dict) else None)
                        fname = getattr(file_obj, "filename", "") or (file_obj.get("filename", "") if isinstance(file_obj, dict) else "") or ""
                        if fid:
                            output_file_ids.append((fid, fname or fid))
        elif typ == "text_editor_code_execution_tool_result":
            cnt = _block_attr(block, "content")
            if isinstance(cnt, dict) and cnt.get("content"):
                output_parts.append(cnt.get("content", ""))

    return {
        "generated_code": "\n\n---\n\n".join(code_parts).strip() or None,
        "execution_output": "\n\n".join(output_parts).strip() or None,
        "output_file_ids": output_file_ids,
    }


class AnthropicClient(APIClient):
    """!
    @brief Klient Anthropic Claude (Messages API): zapytania, streaming, code execution, Files API.
    """

    def __init__(self, api_key: str) -> None:
        """!
        @brief Inicjalizuje klienta SDK i waliduje klucz.

        @param api_key Klucz API (format sk-ant-*).
        """
        super().__init__(api_key)
        self.client = Anthropic(api_key=api_key)
        logger.info("Zainicjalizowano Anthropic client")

    def _validate_api_key(self) -> None:
        """!
        @brief Sprawdza prefix sk-ant-, rzuca ValueError przy błędnym formacie.
        """
        if not self.api_key.startswith("sk-ant-"):
            raise ValueError(
                "Nieprawidłowy format klucza Anthropic (powinien zaczynać się od 'sk-ant-')"
            )

    def query(
        self,
        messages: List[Dict[str, Any]],
        model: str = "claude-sonnet-4.5",
        temperature: float = 0.0,
        max_tokens: int = 1500,
        **kwargs
    ) -> APIResponse:
        """!
        @brief Wysyła zapytanie do Messages API (system, tools, code execution, temperature/top_p).
        Przy Extended Thinking (kwargs thinking) używa streamingu.
        @param messages Lista wiadomości [{role, content}].
        @param model Identyfikator modelu (np. claude-sonnet-4.5).
        @param temperature Losowość (używana tylko gdy brak top_p).
        @param max_tokens Maks. liczba tokenów odpowiedzi.
        @return APIResponse z text, metadata (code_execution, reasoning_content).
        @exception APIError Błąd API lub walidacji.
        """
        try:
            use_code_execution = kwargs.pop("use_code_execution", False)
            tools_arg = kwargs.pop("tools", None)

            system_prompt, user_messages = _extract_system_and_messages(messages)
            model_id = _normalize_model_name(model)

            create_kwargs: Dict[str, Any] = {
                "model": model_id,
                "messages": user_messages,
                "max_tokens": max_tokens,
            }
            if system_prompt is not None:
                create_kwargs["system"] = system_prompt
            if kwargs.get("top_p") is None and temperature is not None:
                create_kwargs["temperature"] = float(temperature)
            thinking_param = _adjust_thinking_for_max_tokens(
                max_tokens, kwargs.pop("thinking", None)
            )
            if thinking_param is not None:
                create_kwargs["thinking"] = thinking_param

            stream_factory: Any
            if use_code_execution:
                create_kwargs["tools"] = [CODE_EXECUTION_TOOL]
                betas = [CODE_EXECUTION_BETA]
                if _messages_contain_container_upload(user_messages):
                    betas.append(FILES_API_BETA)
                create_kwargs["betas"] = betas
                if thinking_param is not None:
                    with self.client.beta.messages.stream(**create_kwargs) as stream:
                        response = stream.get_final_message()
                else:
                    try:
                        response = self.client.beta.messages.create(**create_kwargs)
                    except AttributeError:
                        response = self.client.messages.create(
                            tools=create_kwargs["tools"],
                            **{k: v for k, v in create_kwargs.items() if k not in ("tools", "betas")}
                        )
            else:
                anthropic_tools = _prepare_anthropic_tools(tools_arg)
                if anthropic_tools is not None:
                    create_kwargs["tools"] = anthropic_tools
                if kwargs.get("top_p") is not None:
                    create_kwargs["top_p"] = float(kwargs["top_p"])
                if kwargs.get("top_k") is not None:
                    create_kwargs["top_k"] = int(kwargs["top_k"])
                for key in ("stop_sequences", "metadata", "service_tier"):
                    if key in kwargs and kwargs[key] is not None:
                        create_kwargs[key] = kwargs[key]
                if thinking_param is not None:
                    with self.client.messages.stream(**create_kwargs) as stream:
                        response = stream.get_final_message()
                else:
                    response = self.client.messages.create(**create_kwargs)

            text = _extract_text_from_content(response.content)
            if not text and response.content:
                text = getattr(response.content[0], "text", None) or ""

            metadata = {
                "id": response.id,
                "type": getattr(response, "type", None),
            }
            if use_code_execution and response.content:
                code_exec = _extract_code_execution_from_content(response.content)
                metadata["code_execution"] = code_exec
            thinking_text = _extract_thinking_from_content(response.content)
            if thinking_text:
                metadata["reasoning_content"] = thinking_text

            return APIResponse(
                text=text,
                model=response.model,
                tokens_input=response.usage.input_tokens,
                tokens_output=response.usage.output_tokens,
                finish_reason=response.stop_reason or "end_turn",
                metadata=metadata,
            )

        except AnthropicError as e:
            logger.error(f"Anthropic API error: {e}", exc_info=True)
            raise APIError(f"Anthropic API error: {str(e)}") from e
        except Exception as e:
            logger.error(f"Unexpected error: {e}", exc_info=True)
            raise APIError(f"Unexpected error: {str(e)}") from e

    def query_stream(
        self,
        messages: List[Dict[str, Any]],
        model: str = "claude-sonnet-4.5",
        temperature: float = 0.0,
        max_tokens: int = 1500,
        **kwargs
    ) -> Generator[StreamChunk, None, None]:
        """!
        @brief Zwraca generator fragmentów odpowiedzi (text_delta, thinking_delta).

        @param messages Lista wiadomości.
        @param model Identyfikator modelu.
        @param temperature Losowość (ignorowana gdy podano top_p).
        @return Generator StreamChunk.
        @exception APIError Błąd streamingu.
        """
        try:
            use_code_execution = kwargs.pop("use_code_execution", False)
            tools_arg = kwargs.pop("tools", None)
            system_prompt, user_messages = _extract_system_and_messages(messages)
            model_id = _normalize_model_name(model)

            stream_kwargs: Dict[str, Any] = {
                "model": model_id,
                "messages": user_messages,
                "max_tokens": max_tokens,
            }
            if system_prompt is not None:
                stream_kwargs["system"] = system_prompt
            if kwargs.get("top_p") is not None:
                stream_kwargs["top_p"] = float(kwargs["top_p"])
            elif temperature is not None:
                stream_kwargs["temperature"] = float(temperature)
            if kwargs.get("top_k") is not None:
                stream_kwargs["top_k"] = int(kwargs["top_k"])
            thinking_param = _adjust_thinking_for_max_tokens(
                max_tokens, kwargs.pop("thinking", None)
            )
            if thinking_param is not None:
                stream_kwargs["thinking"] = thinking_param

            if use_code_execution:
                stream_kwargs["tools"] = [CODE_EXECUTION_TOOL]
                betas = [CODE_EXECUTION_BETA]
                if _messages_contain_container_upload(user_messages):
                    betas.append(FILES_API_BETA)
                stream_kwargs["betas"] = betas
                stream_factory = self.client.beta.messages.stream
            else:
                anthropic_tools = _prepare_anthropic_tools(tools_arg)
                if anthropic_tools is not None:
                    stream_kwargs["tools"] = anthropic_tools
                for key in ("stop_sequences", "metadata", "service_tier"):
                    if key in kwargs and kwargs[key] is not None:
                        stream_kwargs[key] = kwargs[key]
                stream_factory = self.client.messages.stream

            with stream_factory(**stream_kwargs) as stream:
                for event in stream:
                    if getattr(event, "type", None) != "content_block_delta":
                        continue
                    delta = getattr(event, "delta", None)
                    if delta is None:
                        continue
                    delta_type = getattr(delta, "type", None) or (delta.get("type") if isinstance(delta, dict) else None)
                    if delta_type == "thinking_delta":
                        thinking_frag = getattr(delta, "thinking", None) or (delta.get("thinking", "") if isinstance(delta, dict) else "") or ""
                        if thinking_frag:
                            yield StreamChunk(content=None, finish_reason=None, reasoning_content=thinking_frag)
                    elif delta_type == "text_delta":
                        text_frag = getattr(delta, "text", None) or (delta.get("text", "") if isinstance(delta, dict) else "") or ""
                        if text_frag:
                            yield StreamChunk(content=text_frag, finish_reason=None)

                response = stream.get_final_message()
                text = _extract_text_from_content(response.content)
                metadata = {
                    "id": response.id,
                    "type": getattr(response, "type", None),
                }
                if use_code_execution and response.content:
                    metadata["code_execution"] = _extract_code_execution_from_content(response.content)
                thinking_text = _extract_thinking_from_content(response.content)
                if thinking_text:
                    metadata["reasoning_content"] = thinking_text
                yield StreamChunk(
                    content=None,
                    finish_reason=response.stop_reason or "end_turn",
                    metadata={
                        "response": APIResponse(
                            text=text,
                            model=response.model,
                            tokens_input=response.usage.input_tokens,
                            tokens_output=response.usage.output_tokens,
                            finish_reason=response.stop_reason or "end_turn",
                            metadata=metadata,
                        )
                    },
                )

        except AnthropicError as e:
            logger.error(f"Anthropic streaming error: {e}", exc_info=True)
            raise APIError(f"Anthropic streaming error: {str(e)}") from e

    def upload_file_for_code_execution(
        self,
        file_content: bytes,
        filename: str = "data.csv",
    ) -> str:
        """!
        @brief Wgrywa plik przez Files API (beta), file_id używany w content jako container_upload (pd.read_csv w code execution).

        @param file_content Zawartość pliku (bytes).
        @param filename Nazwa pliku (np. data.csv).
        @return file_id do wstawienia w content: [{"type": "container_upload", "file_id": "..."}].
        @exception APIError Błąd wgrywania lub brak file_id w odpowiedzi.
        """
        try:
            file_like = io.BytesIO(file_content)
            result = self.client.beta.files.upload(
                file=(filename, file_like, "text/csv"),
                betas=[FILES_API_BETA],
            )
            file_id = getattr(result, "id", None) or getattr(result, "file_id", None) or ""
            if not file_id:
                raise APIError("Files API nie zwróciło file_id po wgraniu pliku.")
            logger.info(f"Wgrano plik do Anthropic Files API: file_id={file_id}")
            return file_id
        except AnthropicError as e:
            logger.error(f"Anthropic file upload error: {e}", exc_info=True)
            raise APIError(f"Nie udało się wgrać pliku: {e}") from e
        except Exception as e:
            logger.error(f"File upload error: {e}", exc_info=True)
            raise APIError(f"Nie udało się wgrać pliku: {e}") from e

    def download_file(self, file_id: str) -> Dict[str, Any]:
        """!
        @brief Pobiera plik wygenerowany przez code execution (Files API beta).

        @param file_id Identyfikator pliku zwrócony w output_file_ids.
        @return Słownik z kluczami: data (bytes), filename (str).
        @exception APIError Błąd pobierania.
        """
        try:
            meta = self.client.beta.files.retrieve_metadata(
                file_id, betas=[FILES_API_BETA]
            )
            filename = getattr(meta, "filename", None) or getattr(meta, "name", "") or f"{file_id}.bin"
            content = self.client.beta.files.download(
                file_id, betas=[FILES_API_BETA]
            )
            if hasattr(content, "read"):
                data = content.read()
            elif hasattr(content, "content"):
                data = content.content
            elif isinstance(content, bytes):
                data = content
            else:
                data = bytes(content)
            return {"data": data, "filename": filename}
        except Exception as e:
            logger.warning(f"Anthropic download_file {file_id}: {e}")
            raise

    def get_available_models(self) -> List[str]:
        """!
        @brief Zwraca listę modeli Claude z konfiguracji (lub domyślną).

        @return Lista nazw modeli.
        """
        return list(MODELS.get("anthropic", ["claude-sonnet-4.5", "claude-opus-4.5", "claude-haiku-4.5"]))
