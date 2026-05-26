##
## @file openai_client.py
## @brief Klient OpenAI Responses API: zapytania, streaming, kontenery, pliki.
##

import io
import re
from typing import Any, Dict, Generator, List, Tuple
from openai import OpenAI, OpenAIError
from api.base import APIClient, APIResponse, StreamChunk, APIError
from utils.logger import logger


class OpenAIClient(APIClient):
    """!
    @brief Klient OpenAI Responses API z obsługą Code Interpreter i kontenerów.
    """

    def __init__(self, api_key: str) -> None:
        """!
        @brief Inicjalizuje klienta SDK i waliduje klucz.

        @param api_key Klucz API (prefix sk-).
        """
        super().__init__(api_key)
        self.client = OpenAI(api_key=api_key)
        logger.info("Zainicjalizowano OpenAI client (Responses API)")

    def _validate_api_key(self) -> None:
        """!
        @brief Sprawdza prefix sk-, rzuca ValueError przy błędnym formacie.
        """
        if not self.api_key.startswith('sk-'):
            raise ValueError(
                "Nieprawidłowy format klucza OpenAI (powinien zaczynać się od 'sk-')"
            )

    def _extract_instructions_and_input(
        self,
        messages: List[Dict[str, Any]]
    ) -> Tuple[str | None, List[Dict[str, Any]]]:
        """!
        @brief Dzieli wiadomości na instructions (system) i input (developer/user/assistant) dla Responses API.

        @param messages Lista słowników {role, content}.
        @return Krotka (instructions, input_messages).
        """
        instructions_parts: List[str] = []
        input_messages: List[Dict[str, Any]] = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                instructions_parts.append(str(content))
                continue

            normalized_role = role if role in {"developer", "user", "assistant"} else "user"
            input_messages.append({
                "role": normalized_role,
                "content": content
            })

        instructions = "\n\n".join(instructions_parts).strip() or None
        return instructions, input_messages

    def _extract_usage_tokens(self, response: Any) -> Tuple[int, int]:
        """!
        @brief Pobiera input_tokens/prompt_tokens i output_tokens/completion_tokens z obiektu usage.

        @param response Obiekt odpowiedzi Responses API.
        @return Krotka (input_tokens, output_tokens).
        """
        usage = getattr(response, "usage", None)
        if usage is None:
            return 0, 0

        input_tokens = (
            getattr(usage, "input_tokens", None)
            or getattr(usage, "prompt_tokens", None)
            or 0
        )
        output_tokens = (
            getattr(usage, "output_tokens", None)
            or getattr(usage, "completion_tokens", None)
            or 0
        )
        return int(input_tokens), int(output_tokens)

    def _extract_output_text(self, response: Any) -> str:
        """!
        @brief Zwraca output_text z odpowiedzi, fallback na output[].content przy innym kształcie obiektu.

        @param response Obiekt odpowiedzi Responses API.
        @return Tekst odpowiedzi.
        """
        text = getattr(response, "output_text", None)
        if isinstance(text, str):
            return text

        output = getattr(response, "output", None) or []
        chunks: List[str] = []
        for item in output:
            if getattr(item, "type", None) != "message":
                continue
            for part in getattr(item, "content", []) or []:
                part_type = getattr(part, "type", None)
                if part_type in {"output_text", "text"}:
                    part_text = getattr(part, "text", None)
                    if part_text:
                        chunks.append(part_text)
        return "".join(chunks)

    def _extract_code_interpreter_calls(self, response: Any) -> list:
        """!
        @brief Wyodrębnia wywołania Code Interpreter z output (code, results, container_id).

        @param response Obiekt odpowiedzi Responses API.
        @return Lista słowników: code, results (logi/pliki/obrazy), container_id.
        """
        calls: list = []
        output = getattr(response, "output", None) or []
        for item in output:
            if getattr(item, "type", None) != "code_interpreter_call":
                continue

            code = getattr(item, "code", "") or ""
            container_id = getattr(item, "container_id", None)
            results: list = []

            logger.debug(
                f"CI call: container_id={container_id}, "
                f"results_count={len(getattr(item, 'results', []) or [])}"
            )

            for result in getattr(item, "results", []) or []:
                rtype = getattr(result, "type", "")
                logger.debug(
                    f"  CI result type='{rtype}', "
                    f"attrs={[a for a in dir(result) if not a.startswith('_')]}"
                )
                if rtype == "logs":
                    results.append({
                        "type": "logs",
                        "logs": getattr(result, "logs", ""),
                    })
                elif rtype == "files":
                    files = [
                        {
                            "file_id": getattr(f, "file_id", ""),
                            "filename": getattr(f, "filename", ""),
                        }
                        for f in (getattr(result, "files", []) or [])
                    ]
                    results.append({"type": "files", "files": files})
                    logger.debug(f"  CI files: {files}")
                elif rtype == "image":
                    file_id = getattr(result, "file_id", "") or ""
                    filename = getattr(result, "filename", "") or ""
                    image_obj = getattr(result, "image", None)
                    if image_obj:
                        file_id = file_id or getattr(image_obj, "file_id", "") or ""
                        filename = filename or getattr(image_obj, "filename", "") or ""
                    if not filename:
                        filename = f"ci_image_{file_id or 'unknown'}.png"
                    if file_id:
                        results.append({
                            "type": "files",
                            "files": [{"file_id": file_id, "filename": filename}],
                        })
                        logger.info(f"  CI image result: file_id={file_id}, filename={filename}")
                    else:
                        logger.warning(f"  CI image result bez file_id: {result}")
                else:
                    logger.warning(
                        f"  Nieobsługiwany typ wyniku CI: '{rtype}', "
                        f"attrs={[a for a in dir(result) if not a.startswith('_')]}"
                    )

            calls.append({
                "code": code,
                "results": results,
                "container_id": container_id,
            })

        logger.info(
            f"Wyodrębniono {len(calls)} wywołań CI, "
            f"łącznie {sum(len(c['results']) for c in calls)} wyników"
        )
        return calls

    def _is_unsupported_temperature_error(self, error: OpenAIError) -> bool:
        """!
        @brief Sprawdza, czy komunikat błędu dotyczy nieobsługiwanego parametru temperature.

        @param error Wyjątek OpenAI.
        @return True, jeśli parametr temperature jest nieobsługiwany.
        """
        return "Unsupported parameter: 'temperature'" in str(error)

    def _remove_unsupported_parameter(
        self,
        error: OpenAIError,
        request_payload: Dict[str, Any]
    ) -> bool:
        """!
        @brief Wykrywa z komunikatu błędu nazwę parametru i usuwa go z request_payload.

        @param error Wyjątek OpenAI.
        @param request_payload Słownik payloadu (modyfikowany w miejscu).
        @return True, jeśli parametr został usunięty.
        """
        match = re.search(r"Unsupported parameter: '([^']+)'", str(error))
        if not match:
            return False

        parameter = match.group(1)
        if parameter not in request_payload:
            return False

        request_payload.pop(parameter, None)
        logger.warning(f"Usuwam nieobsługiwany parametr Responses API: {parameter}")
        return True

    def query(
        self,
        messages: List[Dict[str, str]],
        model: str = 'gpt-5.2',
        temperature: float = 0.0,
        max_tokens: int = 1500,
        **kwargs
    ) -> APIResponse:
        """!
        @brief Wysyła zapytanie do Responses API, obsługuje fallback przy nieobsługiwanych parametrach.

        @param messages Lista wiadomości.
        @param model Nazwa modelu (np. gpt-5.2).
        @param temperature Losowość (opcjonalnie usuwana przy błędzie API).
        @param max_tokens Maks. liczba tokenów.
        @return APIResponse z text, metadata (code_interpreter_calls).
        @exception APIError Błąd API.
        """
        try:
            logger.debug(f"OpenAI query: model={model}, temp={temperature}")

            instructions, input_messages = self._extract_instructions_and_input(messages)
            request_payload: Dict[str, Any] = {
                "model": model,
                "input": input_messages,
                "max_output_tokens": max_tokens
            }
            if instructions:
                request_payload["instructions"] = instructions
            if temperature is not None:
                request_payload["temperature"] = temperature

            reasoning = kwargs.pop("reasoning", None)
            if reasoning is not None:
                request_payload["reasoning"] = reasoning

            request_payload.update(kwargs)

            try:
                response = self.client.responses.create(**request_payload)
            except OpenAIError as e:
                if self._remove_unsupported_parameter(e, request_payload):
                    response = self.client.responses.create(**request_payload)
                elif self._is_unsupported_temperature_error(e) and "temperature" in request_payload:
                    logger.warning(
                        f"Model {model} nie wspiera temperature; ponawiam bez temperature."
                    )
                    request_payload.pop("temperature", None)
                    response = self.client.responses.create(**request_payload)
                else:
                    raise
            tokens_input, tokens_output = self._extract_usage_tokens(response)
            finish_reason = getattr(response, "status", None) or "completed"

            metadata: Dict[str, Any] = {
                'id': response.id,
                'created': getattr(response, 'created_at', None),
                'status': getattr(response, 'status', None),
            }

            ci_calls = self._extract_code_interpreter_calls(response)
            if ci_calls:
                metadata['code_interpreter_calls'] = ci_calls

            return APIResponse(
                text=self._extract_output_text(response),
                model=response.model,
                tokens_input=tokens_input,
                tokens_output=tokens_output,
                finish_reason=finish_reason,
                metadata=metadata
            )

        except OpenAIError as e:
            logger.error(f"OpenAI API error: {e}", exc_info=True)
            raise APIError(f"OpenAI API error: {str(e)}") from e
        except Exception as e:
            logger.error(f"Unexpected error: {e}", exc_info=True)
            raise APIError(f"Unexpected error: {str(e)}") from e

    def query_stream(
        self,
        messages: List[Dict[str, str]],
        model: str = 'gpt-5.2',
        temperature: float = 0.0,
        max_tokens: int = 1500,
        **kwargs
    ) -> Generator[StreamChunk, None, None]:
        """!
        @brief Zwraca generator fragmentów odpowiedzi (response.output_text.delta), fallback na inne eventy.

        @param messages Lista wiadomości.
        @param model Nazwa modelu.
        @param temperature Losowość (może być usunięta przy błędzie).
        @param max_tokens Maks. liczba tokenów.
        @return Generator StreamChunk.
        @exception APIError Błąd streamingu.
        """
        try:
            logger.debug(f"OpenAI stream: model={model}")

            instructions, input_messages = self._extract_instructions_and_input(messages)
            request_payload: Dict[str, Any] = {
                "model": model,
                "input": input_messages,
                "max_output_tokens": max_tokens,
                "stream": True
            }
            if instructions:
                request_payload["instructions"] = instructions
            if temperature is not None:
                request_payload["temperature"] = temperature

            reasoning = kwargs.pop("reasoning", None)
            if reasoning is not None:
                request_payload["reasoning"] = reasoning
            request_payload.update(kwargs)

            try:
                stream = self.client.responses.create(**request_payload)
            except OpenAIError as e:
                if self._remove_unsupported_parameter(e, request_payload):
                    stream = self.client.responses.create(**request_payload)
                elif self._is_unsupported_temperature_error(e) and "temperature" in request_payload:
                    logger.warning(
                        f"Model {model} nie wspiera temperature (stream); ponawiam bez temperature."
                    )
                    request_payload.pop("temperature", None)
                    stream = self.client.responses.create(**request_payload)
                else:
                    raise

            for event in stream:
                event_type = getattr(event, "type", "")

                if event_type == "response.output_text.delta":
                    delta = getattr(event, "delta", None)
                    if delta:
                        yield StreamChunk(content=delta, finish_reason=None)
                    continue

                if event_type == "response.output_text.added":
                    added_text = getattr(event, "text", None)
                    if added_text:
                        yield StreamChunk(content=added_text, finish_reason=None)
                    continue

                if event_type in {"response.completed", "response.failed"}:
                    response = getattr(event, "response", None)
                    metadata = None
                    if response is not None:
                        tokens_input, tokens_output = self._extract_usage_tokens(response)
                        response_metadata: Dict[str, Any] = {
                            'id': getattr(response, 'id', None),
                            'created': getattr(response, 'created_at', None),
                            'status': getattr(response, 'status', None),
                        }
                        ci_calls = self._extract_code_interpreter_calls(response)
                        if ci_calls:
                            response_metadata['code_interpreter_calls'] = ci_calls
                        metadata = {
                            "response": APIResponse(
                                text=self._extract_output_text(response),
                                model=getattr(response, "model", model),
                                tokens_input=tokens_input,
                                tokens_output=tokens_output,
                                finish_reason=getattr(response, "status", None) or event_type,
                                metadata=response_metadata,
                            )
                        }
                    yield StreamChunk(
                        content=None,
                        finish_reason=getattr(event, "type", event_type),
                        metadata=metadata,
                    )
                    continue

                generic_delta = getattr(event, "delta", None)
                if isinstance(generic_delta, str) and generic_delta:
                    yield StreamChunk(
                        content=generic_delta,
                        finish_reason=None
                    )

        except OpenAIError as e:
            logger.error(f"OpenAI streaming error: {e}", exc_info=True)
            raise APIError(f"OpenAI streaming error: {str(e)}") from e

    def create_container(
        self,
        name: str = "data-analysis",
        memory_limit: str = "1g",
    ) -> str:
        """!
        @brief Tworzy kontener Code Interpreter, memory_limit przekazywany przez extra_body (SDK 2.x).

        @param name Nazwa kontenera.
        @param memory_limit Limit pamięci: "1g", "4g", "16g", "64g".
        @return Identyfikator kontenera (cntr_...).
        @exception APIError Błąd tworzenia kontenera.
        """
        try:
            container = self.client.containers.create(
                name=name,
                extra_body={"memory_limit": memory_limit},
            )
            cid = getattr(container, "id", "") or ""
            logger.info(f"Utworzono kontener: {cid}")
            return cid
        except Exception as e:
            logger.error(f"Błąd tworzenia kontenera: {e}", exc_info=True)
            raise APIError(f"Nie udało się utworzyć kontenera: {e}") from e

    def upload_container_file(
        self,
        container_id: str,
        file_content: bytes,
        filename: str = "data_csv",
    ) -> str:
        """!
        @brief Wgrywa plik do kontenera, w kodzie CI dostępny np. przez pd.read_csv(path).

        @param container_id Identyfikator kontenera (cntr_...).
        @param file_content Zawartość pliku (bytes).
        @param filename Nazwa w kontenerze (alfanum, _, -, bez kropek).
        @return Ścieżka w kontenerze (np. /mnt/data/data_csv).
        @exception APIError Błąd wgrywania.
        """
        try:
            file_like = io.BytesIO(file_content)
            cf = self.client.containers.files.create(
                container_id,
                file=file_like,
            )
            path = getattr(cf, "path", "") or "/mnt/data/uploaded_file"
            logger.info(f"Wgrano plik do kontenera {container_id}: {path}")
            return path
        except Exception as e:
            logger.error(
                f"Błąd wgrywania pliku do kontenera {container_id}: {e}",
                exc_info=True,
            )
            raise APIError(f"Nie udało się wgrać pliku do kontenera: {e}") from e

    def list_container_files(self, container_id: str) -> list:
        """!
        @brief Zwraca listę plików w kontenerze (file_id, filename).

        @param container_id Identyfikator kontenera (cntr_...).
        @return Lista słowników {file_id, filename}, pusta przy błędzie.
        """
        try:
            listing = self.client.containers.files.list(container_id)
            items = listing.data if hasattr(listing, 'data') else listing
            files: list = []
            for f in items:
                logger.debug(
                    f"Container file: id={getattr(f, 'id', '?')}, "
                    f"path={getattr(f, 'path', '?')}, "
                    f"bytes={getattr(f, 'bytes', '?')}, "
                    f"source={getattr(f, 'source', '?')}"
                )

                file_id = (
                    getattr(f, "file_id", "")
                    or getattr(f, "id", "")
                    or ""
                )
                filename = (
                    getattr(f, "filename", "")
                    or getattr(f, "name", "")
                    or getattr(f, "path", "")
                    or ""
                )
                files.append({
                    "file_id": file_id,
                    "filename": filename,
                })
            logger.info(
                f"Pliki w kontenerze {container_id}: "
                f"{[(fi['file_id'], fi['filename']) for fi in files]}"
            )
            return files
        except Exception as e:
            logger.warning(f"Nie udało się wylistować plików kontenera {container_id}: {e}")
            return []

    def download_container_file(
        self, container_id: str, file_id: str
    ) -> bytes:
        """!
        @brief Pobiera zawartość pliku z kontenera (cfile_...).

        @param container_id Identyfikator kontenera (cntr_...).
        @param file_id Identyfikator pliku (cfile_...).
        @return Zawartość pliku (bytes).
        """
        try:
            content_resource = self.client.containers.files.content
            if callable(content_resource):
                response = content_resource(container_id, file_id)
            else:
                response = content_resource.retrieve(
                    file_id,
                    container_id=container_id,
                )

            if isinstance(response, bytes):
                return response
            if hasattr(response, 'read'):
                return response.read()  # type: ignore[union-attr]
            if hasattr(response, 'content'):
                return response.content  # type: ignore[union-attr]
            return bytes(response)  # type: ignore[arg-type]
        except Exception as e:
            logger.error(
                f"Błąd pobierania pliku z kontenera "
                f"({container_id}/{file_id}): {e}"
            )
            raise

    def get_available_models(self) -> List[str]:
        """!
        @brief Zwraca listę obsługiwanych modeli Responses API.

        @return Lista nazw modeli.
        """
        return ['gpt-5.2', 'gpt-5', 'gpt-4o', 'gpt-4o-mini']