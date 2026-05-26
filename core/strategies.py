##
## @file strategies.py
## @brief Strategie analizy danych z wykorzystaniem modeli AI.
##
import os
import pandas as pd
from typing import Any, Callable, Dict, List, Optional
import re

from api.base import APIClient, APIResponse
from core.chart_output_names import apply_savefig_names_to_generated_outputs
from core.prompt_builder import (
    PromptBuilder,
    anthropic_code_execution_data_loading,
    build_code_execution_user_prompt,
    code_execution_system_prompt,
    format_code_execution_data_info,
    openai_code_execution_data_loading,
    system_prompt_with_formatting,
)
from compute.statistics import StatisticsCalculator
from utils.logger import logger


def _strategy_query_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """! @brief Usuwa klucze wewnętrzne strategii przed przekazaniem do klienta API."""
    return {
        k: v for k, v in kwargs.items()
        if k not in ("max_tokens", "prior_chat_messages", "stream_callback")
    }


def _max_tokens_from_kwargs(kwargs: Dict[str, Any], default: int = 1500) -> int:
    """! @brief Limit tokenów wyjściowych z parametrów GUI."""
    value = kwargs.get("max_tokens", default)
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return default


def _query_with_optional_stream(
    api_client: APIClient,
    stream_callback: Optional[Callable[[str], None]],
    *,
    messages: list,
    model: str,
    temperature: float,
    max_tokens: int,
    **api_kwargs: Any,
) -> APIResponse:
    """! @brief Wywołuje query_stream z callbackiem lub zwykłe query, gdy brak streamingu."""
    if stream_callback:
        streamed_text_parts: List[str] = []
        response = None
        for chunk in api_client.query_stream(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            **api_kwargs,
        ):
            if chunk.content:
                streamed_text_parts.append(chunk.content)
                stream_callback(chunk.content)
            chunk_response = (chunk.metadata or {}).get("response")
            if chunk_response is not None:
                response = chunk_response

        if response is None:
            return APIResponse(
                text="".join(streamed_text_parts),
                model=model,
                tokens_input=0,
                tokens_output=len(streamed_text_parts),
                finish_reason="completed",
                metadata={},
            )
        if not response.text:
            response.text = "".join(streamed_text_parts)
        return response

    return api_client.query(
        messages=messages,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        **api_kwargs,
    )


class AnalysisStrategy:
    """!
    @brief Abstrakcyjna klasa bazowa dla strategii analizy.
    
    @details Definiuje interfejs dla wszystkich strategii analizy danych.
    
    @see DirectAnalysisStrategy, CodeGenerationStrategy, HybridStrategy
    """
    
    def __init__(self, api_client: APIClient) -> None:
        """!
        @brief Konstruktor klasy bazowej.
        
        @param api_client Klient API do komunikacji z modelem AI.
        """
        self.api_client = api_client
        self.prompt_builder = PromptBuilder()
    
    def analyze(
        self,
        data: pd.DataFrame,
        question: str,
        **kwargs
    ) -> Dict[str, Any]:
        """!
        @brief Przeprowadza analizę danych.
        
        @param data DataFrame z danymi do analizy.
        @param question Pytanie użytkownika dotyczące danych.
        @param kwargs Dodatkowe parametry (model, temperature, max_tokens).
        
        @return Słownik z wynikami analizy.
        
        @exception NotImplementedError Klasa bazowa nie implementuje tej metody.
        """
        raise NotImplementedError("Subclass must implement analyze()")


class DirectAnalysisStrategy(AnalysisStrategy):
    """!
    @brief Strategia bezpośredniej analizy danych przez AI.
    
    @details Dane są przesyłane bezpośrednio do modelu w prompcie.
             AI analizuje dane i generuje interpretację bez wykonywania
             lokalnych obliczeń.
    
    @par Zalety:
         - Prosta implementacja
         - AI ma pełen kontekst danych
    
    @par Wady:
         - Ograniczenie rozmiaru danych przez limit tokenów
         - Potencjalnie wyższe koszty API
    
    @see AnalysisStrategy
    """
    
    def analyze(
        self,
        data: pd.DataFrame,
        question: str,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        prior_chat_messages: Optional[List[Dict[str, str]]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """!
        @brief Wysyła podsumowanie danych i pytanie do modelu, zwraca odpowiedź bez lokalnych obliczeń.

        @param data DataFrame z danymi.
        @param question Pytanie użytkownika.
        @param model Nazwa modelu.
        @param temperature Losowość.
        @param prior_chat_messages Opcjonalna wcześniejsza rozmowa (user/assistant) przed bieżącym pytaniem.
        @return Słownik success, response, tokens, model, strategy, przy błędzie error.
        """
        logger.info(f"DirectAnalysis: question='{question[:50]}...'")

        data_summary = self.prompt_builder.format_data_summary(data, max_rows=None)

        system_prompt = system_prompt_with_formatting("")

        prompt = self.prompt_builder.build_direct_analysis_prompt(
            data_summary, question, None
        )

        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
            if prior_chat_messages:
                messages = [messages[0]] + prior_chat_messages + [messages[1]]

            response = self.api_client.query(
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=_max_tokens_from_kwargs(kwargs),
                **_strategy_query_kwargs(kwargs)
            )
            
            out = {
                'success': True,
                'response': response.text,
                'tokens_input': response.tokens_input,
                'tokens_output': response.tokens_output,
                'model': response.model,
                'strategy': 'direct',
                'request_messages': messages,
            }
            if response.metadata.get("reasoning_content"):
                out['reasoning_content'] = response.metadata["reasoning_content"]
            return out
            
        except Exception as e:
            logger.error(f"DirectAnalysis error: {e}", exc_info=True)
            return {
                'success': False,
                'error': str(e),
                'strategy': 'direct'
            }


class CodeGenerationStrategy(AnalysisStrategy):
    """!
    @brief Strategia analizy z użyciem narzędzia Code Interpreter OpenAI.

    @details Model AI otrzymuje dane w formacie CSV i samodzielnie
             pisze oraz uruchamia kod Python w sandboxowanym kontenerze
             OpenAI.  Wyniki obliczeń są interpretowane przez AI
             i zwracane użytkownikowi.

    @par Zalety:
         - Pełny sandbox - bezpieczne wykonywanie kodu po stronie API
         - Dostęp do bibliotek numerycznych (pandas, numpy, scipy, …)
         - Model iteracyjnie poprawia kod w razie błędów

    @par Wymagania:
         - Dostawca API: wyłącznie OpenAI (Responses API)
         - Narzędzie ``code_interpreter`` z obiektem ``container``

    @see AnalysisStrategy
    """

    DATA_FILE_ID = "data_csv"

    def analyze(
        self,
        data: pd.DataFrame,
        question: str,
        model: str = "gpt-4o",
        temperature: float = 0.0,
        prior_chat_messages: Optional[List[Dict[str, str]]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """!
        @brief Analiza danych z użyciem Code Interpreter.

        @param data      DataFrame z danymi do analizy.
        @param question  Pytanie użytkownika.
        @param model     Nazwa modelu OpenAI.
        @param temperature Parametr temperature.
        @param prior_chat_messages Wcześniejsza rozmowa (user/assistant), opcjonalnie.
        @param kwargs    Dodatkowe parametry przekazywane do API.

        @return Słownik z kluczami: success, response, generated_code,
                execution_output, tokens_input, tokens_output, model, strategy.
        """
        from api.openai_client import OpenAIClient
        from api.anthropic_client import AnthropicClient

        if isinstance(self.api_client, AnthropicClient):
            return self._analyze_with_anthropic_code_execution(
                data, question, model, temperature,
                prior_chat_messages=prior_chat_messages,
                **kwargs
            )

        if not isinstance(self.api_client, OpenAIClient):
            return {
                'success': False,
                'error': (
                    'Strategia wykonywania kodu jest dostępna dla OpenAI (Code Interpreter) '
                    'oraz Anthropic (Code execution tool). Obecny dostawca nie jest obsługiwany.'
                ),
                'strategy': 'code_generation'
            }

        logger.info(f"CodeInterpreter: question='{question[:50]}...'")

        csv_bytes = data.to_csv(index=False).encode("utf-8")
        container_id = self.api_client.create_container(
            name="physics-data-analysis",
            memory_limit="1g",
        )
        data_path = self.api_client.upload_container_file(
            container_id,
            file_content=csv_bytes,
            filename=self.DATA_FILE_ID,
        )

        data_info = format_code_execution_data_info(data)
        system_prompt = code_execution_system_prompt()
        user_prompt = build_code_execution_user_prompt(
            openai_code_execution_data_loading(data_path),
            data_info,
            question,
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        if prior_chat_messages:
            messages = [messages[0]] + prior_chat_messages + [messages[1]]

        tools = [{
            "type": "code_interpreter",
            "container": container_id,
        }]

        stream_callback = kwargs.get("stream_callback")
        extra_kwargs = _strategy_query_kwargs(kwargs)
        max_tokens = _max_tokens_from_kwargs(kwargs)

        try:
            response = _query_with_optional_stream(
                self.api_client,
                stream_callback,
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                tools=tools,
                **extra_kwargs,
            )

            ci_calls = response.metadata.get('code_interpreter_calls', [])
            logger.info(
                f"CI calls w metadanych: {len(ci_calls)}, "
                f"klucze metadata: {list(response.metadata.keys())}"
            )

            generated_code = "\n\n".join(
                call.get('code', '') for call in ci_calls
            ).strip() or None

            execution_logs = "\n".join(
                result.get('logs', '')
                for call in ci_calls
                for result in call.get('results', [])
                if result.get('type') == 'logs' and result.get('logs')
            ).strip() or None

            _IMAGE_EXTS = ('.png', '.jpg', '.jpeg', '.gif', '.webp')
            _IMAGE_SIGNATURES = (
                b'\x89PNG', b'\xff\xd8\xff', b'GIF87a', b'GIF89a', b'RIFF',
            )
            generated_images: List[Dict[str, Any]] = []
            generated_files: List[Dict[str, Any]] = []
            seen_file_ids: set = set()

            def _add_file(container_id: str, file_id: str, filename: str, data: bytes) -> None:
                if not file_id or file_id in seen_file_ids:
                    return
                seen_file_ids.add(file_id)
                base = os.path.basename(filename) if filename else f"{file_id}"
                if not base or base == filename and "/" in str(filename):
                    base = os.path.basename(str(filename)) or f"{file_id}"
                entry = {'filename': base, 'data': data}
                generated_files.append(entry)
                if base.lower().endswith(_IMAGE_EXTS) or (
                    len(data) >= 4 and any(data[:len(s)] == s for s in _IMAGE_SIGNATURES)
                ):
                    generated_images.append({'filename': base, 'data': data})

            for i, call in enumerate(ci_calls):
                cid = call.get('container_id')
                results = call.get('results', [])
                logger.info(
                    f"CI call [{i}]: container_id={cid}, "
                    f"results={len(results)}, result_types={[r.get('type') for r in results]}"
                )
                if not cid:
                    continue
                for result in results:
                    if result.get('type') != 'files':
                        continue
                    for file_info in result.get('files', []):
                        file_id = file_info.get('file_id', '')
                        filename = file_info.get('filename', '') or ''
                        if not file_id:
                            continue
                        try:
                            file_bytes = self.api_client.download_container_file(
                                cid, file_id
                            )
                            _add_file(cid, file_id, filename, file_bytes)
                            logger.info(f"Pobrano plik z kontenera: {filename or file_id} ({len(file_bytes)} B)")
                        except Exception as dl_err:
                            logger.warning(f"Nie udało się pobrać pliku {filename or file_id}: {dl_err}")

            seen_containers: set = set()
            if not generated_files:
                for call in ci_calls:
                    fallback_cid = call.get('container_id')
                    if not fallback_cid or fallback_cid in seen_containers:
                        continue
                    seen_containers.add(fallback_cid)
                    logger.info(f"Brak plików w results - listowanie plików kontenera {fallback_cid}")
                    try:
                        container_files = self.api_client.list_container_files(fallback_cid)
                        for file_info in container_files:
                            fallback_file_id = file_info.get('file_id', '')
                            filename = file_info.get('filename', '') or file_info.get('path', '')
                            if not fallback_file_id or fallback_file_id in seen_file_ids:
                                continue
                            try:
                                file_bytes = self.api_client.download_container_file(fallback_cid, fallback_file_id)
                                if not filename:
                                    if len(file_bytes) >= 4 and file_bytes[:4] == b'\x89PNG':
                                        filename = f"{fallback_file_id}.png"
                                    elif len(file_bytes) >= 3 and file_bytes[:3] == b'\xff\xd8\xff':
                                        filename = f"{fallback_file_id}.jpg"
                                    else:
                                        filename = f"{fallback_file_id}.bin"
                                _add_file(fallback_cid, fallback_file_id, filename, file_bytes)
                                logger.info(f"Pobrano plik (fallback): {filename} ({len(file_bytes)} B)")
                            except Exception as dl_err:
                                logger.warning(f"Nie udało się pobrać {fallback_file_id}: {dl_err}")
                    except Exception as list_err:
                        logger.warning(f"Nie udało się wylistować plików kontenera: {list_err}")

            apply_savefig_names_to_generated_outputs(
                generated_files, generated_images, generated_code
            )

            logger.info(f"Łącznie pobrano {len(generated_files)} plików AI (w tym {len(generated_images)} obrazów)")

            out = {
                'success': True,
                'response': response.text,
                'generated_code': generated_code,
                'execution_output': execution_logs,
                'generated_images': generated_images,
                'generated_files': generated_files,
                'tokens_input': response.tokens_input,
                'tokens_output': response.tokens_output,
                'model': response.model,
                'strategy': 'code_generation',
                'request_messages': messages,
            }
            if response.metadata.get("reasoning_content"):
                out['reasoning_content'] = response.metadata["reasoning_content"]
            return out

        except Exception as e:
            logger.error(f"CodeInterpreter error: {e}", exc_info=True)
            return {
                'success': False,
                'error': str(e),
                'strategy': 'code_generation'
            }

    def _analyze_with_anthropic_code_execution(
        self,
        data: pd.DataFrame,
        question: str,
        model: str = "claude-sonnet-4.5",
        temperature: float = 0.0,
        prior_chat_messages: Optional[List[Dict[str, str]]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """!
        @brief Analiza z Code execution (beta) Anthropic: plik CSV przez Files API (container_upload), pd.read_csv w środowisku wykonania.

        @param data DataFrame z danymi.
        @param question Pytanie użytkownika.
        @param model Nazwa modelu (np. claude-sonnet-4.5).
        @param temperature Losowość.
        @return Słownik jak analyze() (success, response, generated_code, execution_output, pliki).
        """
        logger.info(f"Anthropic Code execution: question='{question[:50]}...'")

        csv_bytes = data.to_csv(index=False).encode("utf-8")
        data_filename = "data_csv"
        try:
            file_id = self.api_client.upload_file_for_code_execution(
                file_content=csv_bytes,
                filename=data_filename + ".csv",
            )
        except Exception as e:
            logger.error(f"Anthropic upload pliku: {e}", exc_info=True)
            return {
                "success": False,
                "error": f"Nie udało się wgrać pliku z danymi: {e}",
                "strategy": "code_generation",
            }

        csv_filename = data_filename + ".csv"
        data_info = format_code_execution_data_info(data)
        system_prompt = code_execution_system_prompt()
        user_text = build_code_execution_user_prompt(
            anthropic_code_execution_data_loading(csv_filename),
            data_info,
            question,
        )

        user_content = [
            {"type": "text", "text": user_text},
            {"type": "container_upload", "file_id": file_id},
        ]

        messages = [{"role": "system", "content": system_prompt}]
        if prior_chat_messages:
            messages.extend(prior_chat_messages)
        messages.append({"role": "user", "content": user_content})

        stream_callback = kwargs.get("stream_callback")
        extra_kwargs = _strategy_query_kwargs(kwargs)
        max_tokens = _max_tokens_from_kwargs(kwargs)

        try:
            response = _query_with_optional_stream(
                self.api_client,
                stream_callback,
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                use_code_execution=True,
                **extra_kwargs,
            )
            code_exec = response.metadata.get("code_execution") or {}
            generated_code = code_exec.get("generated_code")
            execution_output = code_exec.get("execution_output")
            generated_files: List[Dict[str, Any]] = []
            generated_images: List[Dict[str, Any]] = []
            output_file_ids = code_exec.get("output_file_ids") or []
            if output_file_ids and hasattr(self.api_client, "download_file"):
                _img_exts = (".png", ".jpg", ".jpeg", ".gif", ".webp")
                for out_file_id, default_name in output_file_ids:
                    try:
                        info = self.api_client.download_file(out_file_id)
                        file_bytes = info.get("data", b"")
                        filename = info.get("filename", default_name) or default_name or f"{out_file_id}.bin"
                        if isinstance(filename, str) and "/" in filename:
                            filename = os.path.basename(filename)
                        generated_files.append({"filename": filename, "data": file_bytes})
                        if filename.lower().endswith(_img_exts):
                            generated_images.append({"filename": filename, "data": file_bytes})
                        logger.info(f"Pobrano plik Anthropic: {filename} ({len(file_bytes)} B)")
                    except Exception as dl_err:
                        logger.warning(f"Nie udało się pobrać pliku {out_file_id}: {dl_err}")
            apply_savefig_names_to_generated_outputs(
                generated_files, generated_images, generated_code
            )
            out = {
                'success': True,
                'response': response.text,
                'generated_code': generated_code,
                'execution_output': execution_output,
                'generated_images': generated_images,
                'generated_files': generated_files,
                'tokens_input': response.tokens_input,
                'tokens_output': response.tokens_output,
                'model': response.model,
                'strategy': 'code_generation',
                'request_messages': messages,
            }
            if response.metadata.get("reasoning_content"):
                out['reasoning_content'] = response.metadata["reasoning_content"]
            return out
        except Exception as e:
            logger.error(f"Anthropic Code execution error: {e}", exc_info=True)
            return {
                'success': False,
                'error': str(e),
                'strategy': 'code_generation'
            }


class HybridStrategy(AnalysisStrategy):
    """!
    @brief Strategia hybrydowa łącząca AI z lokalnymi obliczeniami.
    
    @details Realizuje trzyetapowy proces analizy:
             1. AI planuje jakie operacje wykonać
             2. Obliczenia są wykonywane lokalnie przez StatisticsCalculator
             3. AI interpretuje wyniki w kontekście fizycznym
    
    @par Zalety:
         - Precyzyjne obliczenia numeryczne
         - Niższe koszty (mniej tokenów na dane)
         - Lepsza interpretowalność wyników
    
    @par Obsługiwane operacje:
         - basic_stats: statystyki opisowe
         - correlation: macierz korelacji
         - linear_fit: regresja liniowa
         - polynomial_fit: dopasowanie wielomianu stopnia 2
         - derivative: pochodna ilorazowa
         - derivative_gradient_at_t: pochodna numeryczna
         - detect_outliers: wykrywanie anomalii
    
    @see AnalysisStrategy, StatisticsCalculator
    """
    
    def __init__(self, api_client: APIClient) -> None:
        """!
        @brief Konstruktor strategii hybrydowej.
        
        @param api_client Klient API do komunikacji z modelem AI.
        """
        super().__init__(api_client)
        self.calculator = StatisticsCalculator()
    
    def analyze(
        self,
        data: pd.DataFrame,
        question: str,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        prior_chat_messages: Optional[List[Dict[str, str]]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """!
        @brief Trzy etapy: planowanie operacji przez AI, wykonanie lokalne (StatisticsCalculator), interpretacja wyników przez AI.

        @param data DataFrame z danymi.
        @param question Pytanie użytkownika.
        @param model Nazwa modelu.
        @param temperature Losowość.
        @param prior_chat_messages Wcześniejsza rozmowa (user/assistant), opcjonalnie.
        @return Słownik success, plan, operations, numerical_results, response, tokens, model, strategy.
        """
        logger.info(f"HybridStrategy: question='{question[:50]}...'")

        try:
            plan = self._plan_analysis(
                data, question, model, temperature,
                prior_chat_messages=prior_chat_messages,
                **kwargs
            )

            if not plan['success']:
                return plan

            results = self._execute_computations(data, plan['operations'])

            interpretation = self._interpret_results(
                question, results, model, temperature,
                prior_chat_messages=prior_chat_messages,
                **kwargs
            )
            request_messages = self.build_interpretation_messages(
                question, results, prior_chat_messages=prior_chat_messages
            )

            response_text = interpretation['interpretation']

            return {
                'success': True,
                'plan': plan['plan_text'],
                'operations': plan['operations'],
                'numerical_results': results,
                'response': response_text,
                'tokens_input': plan['tokens_input'] + interpretation['tokens_input'],
                'tokens_output': plan['tokens_output'] + interpretation['tokens_output'],
                'model': model,
                'strategy': 'hybrid',
                'request_messages': request_messages,
            }
            
        except Exception as e:
            logger.error(f"HybridStrategy error: {e}", exc_info=True)
            return {
                'success': False,
                'error': str(e),
                'strategy': 'hybrid'
            }
    
    def _plan_analysis(
        self,
        data: pd.DataFrame,
        question: str,
        model: str,
        temperature: float,
        prior_chat_messages: Optional[List[Dict[str, str]]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """!
        @brief Pyta model o listę operacji (basic_stats, correlation, linear_fit, itd.), parsuje odpowiedź do listy operacji.

        @param data DataFrame (do opisu w prompcie).
        @param question Pytanie użytkownika.
        @param model Nazwa modelu.
        @param temperature Losowość.
        @param prior_chat_messages Wcześniejsza rozmowa (user/assistant), opcjonalnie.
        @return Słownik success, plan_text, operations, tokens_input, tokens_output; przy błędzie error.
        """
        data_summary = f"""
Dostępne kolumny: {list(data.columns)}
Liczba próbek: {len(data)}
Typy danych: {data.dtypes.to_dict()}
"""
        
        system_prompt = system_prompt_with_formatting(
            """Jesteś ekspertem planującym analizę danych fizycznych.
Wybierz odpowiednie operacje z dostępnych:
- basic_stats: statystyki opisowe
- correlation: macierz korelacji
- linear_fit: regresja liniowa (podaj kolumny x i y)
- polynomial_fit: wielomian stopnia 2 względem x→y (podaj x,y)
- detect_outliers: wykrywanie anomalii
- derivative: pochodna ilorazowa O(h) dla profilu całego szeregu (x,y)
- derivative_gradient_at_t: pochodna numeryczna (podaj x,y,t_target)

Odpowiedz TYLKO nazwami operacji oddzielonymi przecinkami, np: basic_stats, linear_fit(col1,col2)
Dla polynomial_fit lub derivative lub derivative_gradient_at_t lub linear_fit dodaj w nawiasie kolumny, np: polynomial_fit(t_s,speed_mps)"""
        )
        
        prompt = f"""{data_summary}

Pytanie użytkownika: {question}

Jakie operacje należy wykonać?"""
        
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
            if prior_chat_messages:
                messages = [messages[0]] + prior_chat_messages + [messages[1]]

            response = self.api_client.query(
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=_max_tokens_from_kwargs(kwargs),
                **_strategy_query_kwargs(kwargs)
            )

            operations = self._parse_operations(response.text, data)

            return {
                'success': True,
                'plan_text': response.text,
                'operations': operations,
                'tokens_input': response.tokens_input,
                'tokens_output': response.tokens_output
            }
            
        except Exception as e:
            logger.error(f"Planning error: {e}", exc_info=True)
            return {
                'success': False,
                'error': f"Błąd planowania: {str(e)}"
            }
    
    def _parse_operations(self, text: str, data: pd.DataFrame) -> list:
        """!
        @brief Wyciąga z tekstu nazwy operacji (basic_stats, linear_fit(x,y), derivative(x,y), itd.) i waliduje kolumny.

        @param text Odpowiedź modelu (np. "basic_stats, linear_fit(col1,col2)").
        @param data DataFrame do sprawdzenia istnienia kolumn.
        @return Lista słowników {type, params} bez duplikatów, przy braku dopasowań zwraca [basic_stats].
        """
        operations = []
        numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
        seen_simple_ops: set = set()

        patterns = {
            'basic_stats': r'basic_stats',
            'correlation': r'correlation',
            'linear_fit': r'linear_fit\(([^,]+),([^)]+)\)',
            'polynomial_fit': r'polynomial_fit\(([^,]+),([^)]+)\)',
            'detect_outliers': r'detect_outliers',
            'derivative_gradient_at_t': r'derivative_gradient_at_t\(([^,]+),([^)]+)\)',
            'derivative': r'derivative\(([^,]+),([^)]+)\)',
        }
        parameterized = frozenset({
            'linear_fit', 'derivative', 'polynomial_fit', 'derivative_gradient_at_t',
        })

        for op_type, pattern in patterns.items():
            matches = re.finditer(pattern, text, re.IGNORECASE)

            for match in matches:
                if op_type in parameterized:
                    col_x = match.group(1).strip()
                    col_y = match.group(2).strip()
                    key = (op_type, col_x, col_y)
                    if col_x in numeric_cols and col_y in numeric_cols and key not in seen_simple_ops:
                        seen_simple_ops.add(key)
                        operations.append({
                            'type': op_type,
                            'params': {'x': col_x, 'y': col_y},
                        })
                else:
                    if op_type not in seen_simple_ops:
                        seen_simple_ops.add(op_type)
                        operations.append({
                            'type': op_type,
                            'params': {}
                        })
                    break

        if not operations:
            operations.append({'type': 'basic_stats', 'params': {}})

        logger.debug(f"Parsed operations: {operations}")
        return operations

    def _execute_computations(
        self,
        data: pd.DataFrame,
        operations: list
    ) -> Dict[str, Any]:
        """!
        @brief Wykonuje operacje (basic_stats, correlation, linear_fit, detect_outliers, derivative) przez StatisticsCalculator.

        @param data DataFrame z danymi.
        @param operations Lista słowników {type, params}.
        @return Słownik z kluczami odpowiadającymi typom operacji, przy błędzie wpis error.
        """
        results = {}
        
        for op in operations:
            op_type = op['type']
            params = op['params']
            
            try:
                if op_type == 'basic_stats':
                    results['basic_stats'] = self.calculator.basic_statistics(data)
                    
                elif op_type == 'correlation':
                    results['correlation'] = self.calculator.correlation_matrix(data)
                    
                elif op_type == 'linear_fit':
                    x_col = params['x']
                    y_col = params['y']
                    results['linear_fit'] = self.calculator.linear_regression(
                        data[x_col], data[y_col]
                    )
                    results['linear_fit']['x_column'] = x_col
                    results['linear_fit']['y_column'] = y_col
                    
                elif op_type == 'detect_outliers':
                    results['outliers'] = self.calculator.detect_outliers(data)
                    
                elif op_type == 'derivative':
                    x_col = params['x']
                    y_col = params['y']
                    results['derivative'] = self.calculator.numerical_derivative(
                        data[x_col], data[y_col]
                    )
                    results['derivative']['x_column'] = x_col
                    results['derivative']['y_column'] = y_col

                elif op_type == 'polynomial_fit':
                    x_col = params['x']
                    y_col = params['y']
                    results['polynomial_fit'] = self.calculator.polynomial_fit_degree2(
                        data[x_col], data[y_col],
                    )
                    results['polynomial_fit']['x_column'] = x_col
                    results['polynomial_fit']['y_column'] = y_col

                elif op_type == 'derivative_gradient_at_t':
                    x_col = params['x']
                    y_col = params['y']
                    t_tgt = float(params.get('t_target', 5.0))
                    results['derivative_gradient_at_t'] = (
                        self.calculator.numerical_gradient_at_row_nearest_t(
                            data[x_col], data[y_col], t_target=t_tgt,
                        )
                    )
                    results['derivative_gradient_at_t']['x_column'] = x_col
                    results['derivative_gradient_at_t']['y_column'] = y_col
            except Exception as e:
                logger.error(f"Computation error for {op_type}: {e}")
                results[op_type] = {'error': str(e)}
        
        return results
    
    def build_interpretation_messages(
        self,
        question: str,
        results: Dict[str, Any],
        prior_chat_messages: Optional[List[Dict[str, str]]] = None,
    ) -> list:
        """!
        @brief Buduje listę wiadomości [system, user] do etapu interpretacji, używane też w streamingu hybrydowym.

        @param question Oryginalne pytanie użytkownika.
        @param results Wyniki obliczeń (_format_results_for_ai).
        @param prior_chat_messages Wcześniejsza rozmowa (user/assistant), opcjonalnie.
        @return Lista [{role, content}, ...] gotowa do query/query_stream.
        """
        system_prompt = system_prompt_with_formatting(
            "Jesteś fizykiem interpretującym wyniki analizy danych "
            "eksperymentalnych.\nWyjaśnij co oznaczają obliczone wartości "
            "w kontekście fizycznym.\nZidentyfikuj kluczowe zależności "
            "i zasugeruj wnioski."
        )

        results_text = self._format_results_for_ai(results)

        prompt = (
            f"Pytanie użytkownika: {question}\n\n"
            f"Wyniki obliczeń:\n{results_text}\n\n"
            "Zinterpretuj te wyniki w kontekście fizycznym "
            "i odpowiedz na pytanie użytkownika."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        if prior_chat_messages:
            messages = [messages[0]] + prior_chat_messages + [messages[1]]
        return messages

    def _interpret_results(
        self,
        question: str,
        results: Dict[str, Any],
        model: str,
        temperature: float,
        prior_chat_messages: Optional[List[Dict[str, str]]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """!
        @brief Wysyła pytanie i sformatowane wyniki do modelu, zwraca interpretację i liczbę tokenów.

        @param question Oryginalne pytanie.
        @param results Wyniki z _execute_computations.
        @param model Nazwa modelu.
        @param temperature Losowość.
        @param prior_chat_messages Wcześniejsza rozmowa (user/assistant), opcjonalnie.
        @return Słownik interpretation, tokens_input, tokens_output, przy błędzie surowe wyniki w interpretation.
        """
        messages = self.build_interpretation_messages(
            question, results, prior_chat_messages=prior_chat_messages
        )

        try:
            response = self.api_client.query(
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=_max_tokens_from_kwargs(kwargs),
                **_strategy_query_kwargs(kwargs)
            )
            
            return {
                'interpretation': response.text,
                'tokens_input': response.tokens_input,
                'tokens_output': response.tokens_output
            }
            
        except Exception as e:
            logger.error(f"Interpretation error: {e}", exc_info=True)
            results_text = self._format_results_for_ai(results)
            return {
                'interpretation': f"Błąd interpretacji: {str(e)}\n\nSurowe wyniki:\n{results_text}",
                'tokens_input': 0,
                'tokens_output': 0
            }
    
    def _format_results_for_ai(self, results: Dict[str, Any]) -> str:
        """!
        @brief Zamienia wyniki (basic_stats, correlation, linear_fit, outliers, derivative) na tekst do promptu.

        @param results Słownik zwrócony przez _execute_computations.
        @return Tekst z wartościami liczbowymi i etykietami.
        """
        formatted = []

        def _fmt(val: Any) -> str:
            try:
                return f"{float(val):.4f}"
            except (TypeError, ValueError):
                return str(val)

        for key, value in results.items():
            if key == 'basic_stats':
                formatted.append("Statystyki opisowe:")
                for col, stats in value.items():
                    formatted.append(f"  {col}:")
                    for stat_name, stat_val in stats.items():
                        formatted.append(f"    {stat_name}: {_fmt(stat_val)}")

            elif key == 'correlation':
                formatted.append("\nMacierz korelacji (Pearson):")
                for col_a, row in value.items():
                    for col_b, corr_val in row.items():
                        if col_a < col_b:
                            formatted.append(f"  {col_a} ↔ {col_b}: {_fmt(corr_val)}")

            elif key == 'linear_fit':
                formatted.append(f"\nRegresja liniowa ({value.get('x_column')} vs {value.get('y_column')}):")
                formatted.append(f"  Nachylenie: {_fmt(value['slope'])}")
                formatted.append(f"  Przecięcie: {_fmt(value['intercept'])}")
                formatted.append(f"  R²: {_fmt(value['r_squared'])}")

            elif key == 'outliers':
                formatted.append(f"\nWykryte anomalie:")
                formatted.append(f"  Liczba: {len(value)}")
                if value:
                    formatted.append(f"  Indeksy: {value[:10]}")

            elif key == 'derivative':
                formatted.append(f"\nPochodna ({value.get('y_column')} po {value.get('x_column')}):")
                formatted.append(f"  Średnia: {_fmt(value['mean'])}")
                formatted.append(f"  Odch. std: {_fmt(value['std'])}")

            elif key == 'polynomial_fit' and isinstance(value, dict):
                formatted.append(
                    f"\nDopasowanie wielomianu deg 2 ({value.get('y_column')} vs {value.get('x_column')}):",
                )
                if value.get("error"):
                    formatted.append(f"  {value['error']}")
                else:
                    formatted.append(
                        f"  a*t^2+b*t+c: a={_fmt(value['a'])} b={_fmt(value['b'])} c={_fmt(value['c'])}",
                    )
                    formatted.append(f"  R² (na wyjściowych argumentach dopasowania): {_fmt(value['r_squared'])}")

            elif key == 'derivative_gradient_at_t' and isinstance(value, dict):
                formatted.append(
                    f"\nPochodna (numpy.gradient) {value.get('y_column')} wg {value.get('x_column')}:",
                )
                if value.get("error"):
                    formatted.append(f"  {value['error']}")
                else:
                    tt = float(value.get("t_target", 5.0))
                    formatted.append(f"  Wiersz przy x najbliższym docelowego t={tt}: x={_fmt(value['t_row'])}")
                    formatted.append(f"  dy/dx: {_fmt(value['pochodna_t'])}")

        return "\n".join(formatted)
