#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

__author__ = "bibow"

import base64
import json
import logging
import threading
import uuid
from decimal import Decimal
from enum import Enum
from io import BytesIO
from queue import Queue
from typing import Any, Dict, List, Optional

import anthropic
import pendulum
from llm_provider_base import (
    LlmProviderHandler,
    endpoint_base,
    fetch_upstream_models,
)
from silvaengine_utility import Debugger, convert_decimal_to_number
from silvaengine_utility.performance_monitor import performance_monitor
from silvaengine_utility.serializer import Serializer


class AnthropicBetaVersion(str, Enum):
    """Enum for Anthropic API beta versions"""

    MCP_CLIENT = "mcp-client-2025-04-04"
    CODE_EXECUTION = "code-execution-2025-08-25"
    FILES_API = "files-api-2025-04-14"
    WEB_FETCH = "web-fetch-2025-09-10"
    INTERLEAVED_THINKING = "interleaved-thinking-2025-05-14"
    SKILLS = "skills-2025-10-02"


# ----------------------------
# Anthropic Response Streaming with Function Handling and History
# ----------------------------
class AnthropicProvider(LlmProviderHandler):
    """
    A handler class for managing interactions with the Anthropic API.
    Provides functionality for streaming responses, handling function calls,
    maintaining conversation history, and managing outputs.

    BanyanOS provider-plugin contract (LLM Provider Plugin plan §5.1/A.2):
    single-turn (tool_mode="return"); tools execute in agent_engine.
    """

    # ------------------------------------------------------------------
    # Plugin metadata (discovered by llm_engine via the entry point)
    # ------------------------------------------------------------------
    api_protocol = "anthropic"
    #: llm_engine built-in protocol this plugin takes over (plan D8).
    replaces_protocol = "ANTHROPIC"
    display_name = "Anthropic"
    contract_version = "1"
    fallback_protocol = None
    capabilities = {
        "streaming": True,
        "tools": True,
        "vision": True,
        "files": True,
        "reasoning": True,
        "citations": True,
        "prompt_cache": True,
        "provider_tools": ["web_search", "web_fetch", "code_execution"],
        "list_models": True,
    }
    default_pool_name = "httpx_llm_anthropic"

    @classmethod
    def list_models(cls, ctx: Any) -> List[Dict[str, str]]:
        """Anthropic model list: ``GET {endpoint}/v1/models``."""
        key = str((getattr(ctx, "credentials", None) or {}).get("api_key") or "")
        base = endpoint_base(ctx, "https://api.anthropic.com")
        if base.endswith("/v1"):
            base = base[: -len("/v1")]
        return fetch_upstream_models(
            ctx, f"{base}/v1/models",
            headers={"x-api-key": key, "anthropic-version": "2023-06-01"} if key else {},
        )

    def __init__(
        self,
        logger: logging.Logger,
        agent: Dict[str, Any],
        **setting: Dict[str, Any],
    ) -> None:
        """
        Initializes the Anthropic event handler with required configuration.
        Sets up logging, API client, and model settings.

        Args:
            logger: Logger instance for recording events and errors
            agent: Dictionary containing agent configuration and available tools
            setting: Additional configuration settings as key-value pairs
        """
        LlmProviderHandler.__init__(self, logger, agent, **setting)

        # Plugin contract (plan D7 / CREDS): the pooled httpx client is
        # injected by llm_engine's HandlerProviderAdapter (plan §5.2.1) and
        # credentials come from ctx.credentials — never from
        # agent["configuration"]. SDK retries are disabled: ProviderCaller
        # is the single retry/timeout owner.
        http_client = getattr(self.ctx, "http_client", None) if self.ctx else None
        if http_client is None:
            raise Exception(
                "AnthropicProvider requires ctx.http_client (the pooled "
                "client injected by llm_engine's HandlerProviderAdapter, "
                "plan §5.2.1)"
            )
        api_key = (self.credentials.get("api_key") or "").strip() or None
        base_url = (self.agent.get("configuration", {}) or {}).get("base_url")
        if all(
            setting.get(k) for k in ["aws_access_key", "aws_secret_key", "aws_region"]
        ):
            aws_credentials = {
                "aws_access_key": setting["aws_access_key"],
                "aws_secret_key": setting["aws_secret_key"],
                "aws_region": setting["aws_region"],
            }
            self.client = anthropic.AnthropicBedrock(
                **aws_credentials,
                http_client=http_client,
                max_retries=0,
            )
        elif all(setting.get(k) for k in ["project_id", "region"]):
            vertex_credentials = {
                "project_id": setting["project_id"],
                "region": setting["region"],
            }
            self.client = anthropic.AnthropicVertex(
                **vertex_credentials,
                http_client=http_client,
                max_retries=0,
            )
        else:
            self.client = anthropic.Anthropic(
                api_key=api_key,
                http_client=http_client,
                max_retries=0,
            )
            if base_url:
                self.client.base_url = base_url

        if "enabled_tools" in self.agent["configuration"]:
            # Add tools if available - matching example.py structure.
            # Neutral tool shape (plan §5.5): OpenAI function entries carry
            # the name under "function"; Anthropic-shaped entries at top.

            def _tool_name(tool: Any) -> Optional[str]:
                if not isinstance(tool, dict):
                    return None
                if isinstance(tool.get("function"), dict):
                    return tool["function"].get("name")
                return tool.get("name")

            enabled_tools = []
            if "tools" in self.agent["configuration"]:
                for tool in self.agent["configuration"]["tools"]:
                    if _tool_name(tool) not in self.agent["configuration"].get(
                        "enabled_tools", []
                    ):
                        continue
                    enabled_tools.append(tool)
            self.agent["configuration"]["tools"] = enabled_tools

        # Convert Decimal to appropriate types and build model settings (performance optimization)
        self.model_setting = {
            "system": self.agent["instructions"],
        }
        for k, v in self.agent.get("configuration", {}).items():
            if k not in ["api_key", "base_url", "text"]:
                if k == "max_tokens":
                    self.model_setting[k] = int(v)
                elif k == "temperature":
                    self.model_setting[k] = float(v)
                elif isinstance(v, Decimal):
                    self.model_setting[k] = float(v)
                else:
                    self.model_setting[k] = convert_decimal_to_number(v)

        # Neutral → Anthropic tool shape (plan §5.5: handlers convert tools;
        # agent_engine arrives with OpenAI function-shaped tools).
        _neutral_tools = self.model_setting.get("tools")
        if (
            isinstance(_neutral_tools, list)
            and _neutral_tools
            and isinstance(_neutral_tools[0], dict)
            and isinstance(_neutral_tools[0].get("function"), dict)
        ):
            self.model_setting["tools"] = [
                {
                    "name": (tool.get("function") or {}).get("name") or "",
                    "description": (tool.get("function") or {}).get("description")
                    or "",
                    "input_schema": (tool.get("function") or {}).get("parameters")
                    or {"type": "object", "properties": {}},
                }
                for tool in self.model_setting["tools"]
                if isinstance(tool, dict)
            ]

        # P5（plan §6.3 providerTools）：provider 原生工具声明（typed
        # dicts，如 web_search/code_execution beta shapes）合并进 tools。
        _provider_tools = self.model_setting.pop("provider_tools", None)
        if isinstance(_provider_tools, list) and _provider_tools:
            self.model_setting["tools"] = list(
                self.model_setting.get("tools") or []
            ) + [t for t in _provider_tools if isinstance(t, dict)]

        # Cache frequently accessed configuration values (performance optimization)
        self.output_format_type = (
            self.model_setting.get("text", {"format": {"type": "text"}})
            .get("format", {"type": "text"})
            .get("type", "text")
        )

        # Validate reasoning/thinking configuration if present
        if "thinking" in self.model_setting:
            if not isinstance(self.model_setting["thinking"], dict):
                if self.logger and self.logger.isEnabledFor(logging.WARNING):
                    self.logger.warning(
                        "Thinking configuration should be a dictionary. "
                        "Thinking features may not work correctly."
                    )
            elif not self.model_setting["thinking"].get("type"):
                if self.logger and self.logger.isEnabledFor(logging.WARNING):
                    self.logger.warning(
                        "Thinking type is not specified in configuration. "
                        "Extended thinking may not be enabled."
                    )

        # Validate skills configuration if present
        if "skills" in self.model_setting:
            skills = self.model_setting["skills"]
            if not isinstance(skills, list):
                if self.logger and self.logger.isEnabledFor(logging.WARNING):
                    self.logger.warning(
                        "Skills configuration should be a list. "
                        "Skills features may not work correctly."
                    )
            elif len(skills) > 8:
                if self.logger and self.logger.isEnabledFor(logging.WARNING):
                    self.logger.warning(
                        f"Maximum 8 skills allowed per request, got {len(skills)}. "
                        "Only the first 8 skills will be used."
                    )
                self.model_setting["skills"] = skills[:8]

        # Container ID for multi-turn conversations with skills
        self._container_id = None

        self.assistant_messages = []

        # Initialize timeline tracking
        self._global_start_time = None
        self._ask_model_depth = 0
        self.enable_timeline_log = setting.get("enable_timeline_log", False)

    def _get_elapsed_time(self) -> float:
        """
        Get elapsed time in milliseconds from the first ask_model call.

        Returns:
            Elapsed time in milliseconds, or 0 if global start time not set
        """
        if not hasattr(self, "_global_start_time") or self._global_start_time is None:
            return 0.0
        return (pendulum.now("UTC") - self._global_start_time).total_seconds() * 1000

    def reset_timeline(self) -> None:
        """
        Reset the global timeline for a new run.
        Should be called at the start of each new user interaction/run.
        """
        self._global_start_time = None
        if self.enable_timeline_log and self.logger.isEnabledFor(logging.INFO):
            self.logger.info("[TIMELINE] Timeline reset for new run")

    def invoke_model(self, **kwargs: Dict[str, Any]) -> Any:
        """
        Makes an API call to the Anthropic model with provided messages.
        Handles both streaming and non-streaming requests.

        Args:
            kwargs: Dictionary containing input messages and streaming configuration

        Returns:
            Response from the Anthropic API

        Raises:
            Exception: If the API call fails for any reason
        """
        try:
            invoke_start = pendulum.now("UTC")
            messages = list(filter(lambda x: bool(x["content"]), kwargs["input"]))
            # Convert any Decimal values to numbers for JSON serialization
            messages = convert_decimal_to_number(messages)
            betas = self._get_betas(messages)

            # Prepare API call parameters
            api_params = {
                "messages": messages,
                "stream": kwargs["stream"],
            }

            # Add thinking configuration if present
            if "thinking" in self.model_setting and isinstance(
                self.model_setting["thinking"], dict
            ):
                thinking_config = self.model_setting["thinking"]
                # Only include thinking in API call if type is specified
                if thinking_config.get("type"):
                    # Create thinking parameter with type and optional budget
                    thinking_param = {"type": thinking_config["type"]}
                    if "budget_tokens" in thinking_config:
                        thinking_param["budget_tokens"] = thinking_config[
                            "budget_tokens"
                        ]
                    api_params["thinking"] = thinking_param

            # Add container configuration for skills
            if "skills" in self.model_setting and isinstance(
                self.model_setting["skills"], list
            ):
                skills = self.model_setting["skills"]
                if skills:
                    container = {"skills": skills}
                    # Reuse container ID for multi-turn conversations
                    if self._container_id:
                        container["id"] = self._container_id
                    api_params["container"] = container

                    # Ensure code_execution tool is present (required for skills)
                    tools = self.model_setting.get("tools", [])
                    has_code_execution = any(
                        tool.get("type") == "code_execution_20250825"
                        or tool.get("name") == "code_execution"
                        for tool in tools
                    )
                    if not has_code_execution:
                        # Add code execution tool if not present
                        if "tools" not in self.model_setting:
                            self.model_setting["tools"] = []
                        self.model_setting["tools"].append(
                            {
                                "type": "code_execution_20250825",
                                "name": "code_execution",
                            }
                        )

            # Filter out "thinking" and "skills" from model_setting to avoid duplication
            # since we handle them explicitly in api_params
            filtered_model_setting = {
                k: v
                for k, v in self.model_setting.items()
                if k not in ["thinking", "skills"]
            }

            if betas:
                api_params["betas"] = betas
                result = self.client.beta.messages.create(
                    **dict(filtered_model_setting, **api_params)
                )
            else:
                result = self.client.messages.create(
                    **dict(filtered_model_setting, **api_params)
                )

            if self.enable_timeline_log and self.logger.isEnabledFor(logging.INFO):
                invoke_end = pendulum.now("UTC")
                invoke_time = (invoke_end - invoke_start).total_seconds() * 1000
                elapsed = self._get_elapsed_time()
                self.logger.info(
                    f"[TIMELINE] T+{elapsed:.2f}ms: API call returned (took {invoke_time:.2f}ms)"
                )

            return result
        except Exception as e:
            Debugger.info(
                variable=e,
                stage=f"{__name__}:invoke_model",
                delimiter="#",
            )
            raise Exception(f"Failed to invoke model: {str(e)}")

    @performance_monitor.monitor_operation(operation_name="Anthorpic")
    def ask_model(
        self,
        input_messages: List[Dict[str, Any]],
        queue: Queue = None,
        stream_event: threading.Event = None,
        input_files: List[str] = [],
        model_setting: Dict[str, Any] = None,
    ) -> Optional[str]:
        """
        Primary method for sending requests to the Anthropic API.
        Supports both streaming and non-streaming modes, with optional model configuration overrides.

        Args:
            input_messages: List of messages representing conversation history and current query
            queue: Optional queue for handling streaming responses
            stream_event: Optional event for signaling stream completion
            input_files: Optional list of input files to process
            model_setting: Optional dictionary to override default model settings

        Returns:
            String containing response ID for non-streaming requests, None for streaming

        Raises:
            Exception: If request processing fails
        """
        # Track preparation time
        ask_model_start = pendulum.now("UTC")

        # Track recursion depth to identify top-level vs recursive calls
        if not hasattr(self, "_ask_model_depth"):
            self._ask_model_depth = 0

        self._ask_model_depth += 1
        is_top_level = self._ask_model_depth == 1

        # Establish per-run state here rather than relying on __init__:
        # handler instances are cached and reused across requests, so building
        # these in __init__ and mutating them in place (final_output.update(...))
        # meant concurrent runs shared one set of containers and overwrote each
        # other's results — final_output *is* the response payload. Rebinding,
        # never mutating, is what isolates each run.
        #
        # Only at top level: this method recurses for tool-call follow-ups, and
        # final_output / _short_term_memory accumulate across those levels.
        #
        # Inlined rather than a base-class helper on purpose: llm_provider_base
        # is a separately installed package, so calling a new base method would
        # raise AttributeError wherever that package is older than this one.
        if is_top_level:
            self.accumulated_text = ""
            self.final_output = {}
            self.uploaded_files = []
            self._short_term_memory = []

        # Initialize global start time only on top-level ask_model call
        # Recursive calls will use the same start time for the entire run timeline
        if is_top_level:
            self._global_start_time = ask_model_start

            # Reset reasoning_summary for new conversation turn
            # Recursive calls (function call loops) will continue accumulating
            if "reasoning_summary" in self.final_output:
                del self.final_output["reasoning_summary"]

            if self.enable_timeline_log and self.logger.isEnabledFor(logging.INFO):
                self.logger.info("[TIMELINE] T+0ms: Run started - First ask_model call")
        else:
            if self.enable_timeline_log and self.logger.isEnabledFor(logging.INFO):
                elapsed = self._get_elapsed_time()
                self.logger.info(
                    f"[TIMELINE] T+{elapsed:.2f}ms: Recursive ask_model call started"
                )

        try:
            if not self.client:
                self.logger.error("No Anthropic client provided.")
                return None

            stream = True if queue is not None else False

            # Add model-specific settings if provided
            if model_setting:
                # Rebind rather than mutate: model_setting is built once in
                # __init__ and shared with every other instance of this cached
                # handler, so an in-place update would leak this request's
                # overrides into all later requests.
                self.model_setting = {**self.model_setting, **model_setting}

            # Clean up input messages to remove broken tool sequences (performance optimization)
            cleanup_start = pendulum.now("UTC")
            cleanup_end = pendulum.now("UTC")
            cleanup_time = (cleanup_end - cleanup_start).total_seconds() * 1000

            timestamp = pendulum.now("UTC").int_timestamp
            # Optimized UUID generation - use .hex instead of str() conversion
            run_id = f"run-antropic-{self.model_setting['model']}-{timestamp}-{uuid.uuid4().hex[:8]}"

            if input_files:
                input_messages = self._process_input_files(input_files, input_messages)

            if self.enable_timeline_log and self.logger.isEnabledFor(logging.INFO):
                # Track total preparation time before API call
                preparation_end = pendulum.now("UTC")
                preparation_time = (
                    preparation_end - ask_model_start
                ).total_seconds() * 1000
                elapsed = self._get_elapsed_time()
                self.logger.info(
                    f"[TIMELINE] T+{elapsed:.2f}ms: Preparation complete (took {preparation_time:.2f}ms, cleanup: {cleanup_time:.2f}ms)"
                )

            response = self.invoke_model(
                **{
                    "input": input_messages,
                    "stream": stream,
                }
            )

            # If streaming is enabled, process chunks
            if stream:
                queue.put({"name": "run_id", "value": run_id})
                self.handle_stream(
                    response,
                    input_messages,
                    stream_event=stream_event,
                )
            else:
                self.handle_response(response, input_messages)

            return run_id
        except Exception as e:
            self.logger.error(f"Error in ask_model: {str(e)}")
            raise Exception(f"Failed to process model request: {str(e)}")
        finally:
            # Decrement depth when exiting ask_model
            self._ask_model_depth -= 1

            # Reset timeline when returning to depth 0 (top-level call complete)
            if self._ask_model_depth == 0:
                if self.enable_timeline_log and self.logger.isEnabledFor(logging.INFO):
                    elapsed = self._get_elapsed_time()
                    self.logger.info(
                        f"[TIMELINE] T+{elapsed:.2f}ms: Run complete - Resetting timeline"
                    )
                self._global_start_time = None

    def _get_betas(self, input_messages: List[Dict[str, Any]]) -> List[str]:
        """
        Determine which beta features are needed based on configuration and input messages.

        Args:
            input_messages: List of message dictionaries to check

        Returns:
            List of beta version strings required for the API call
        """
        betas = []

        # Check for MCP servers
        if "mcp_servers" in self.model_setting:
            betas.append(AnthropicBetaVersion.MCP_CLIENT.value)

        # Check for code execution tool
        if any(
            tool["name"] == "code_execution"
            for tool in self.model_setting.get("tools", [])
        ):
            betas.append(AnthropicBetaVersion.CODE_EXECUTION.value)

        # Check for web fetch tool
        if any(
            tool["name"] == "web_fetch" for tool in self.model_setting.get("tools", [])
        ):
            betas.append(AnthropicBetaVersion.WEB_FETCH.value)

        # Check for extended thinking/reasoning
        # Interleaved thinking allows Claude to reason between tool calls
        if "thinking" in self.model_setting and isinstance(
            self.model_setting["thinking"], dict
        ):
            thinking_config = self.model_setting["thinking"]
            # Enable interleaved thinking for Claude 4 models or when explicitly configured
            if thinking_config.get("type") in ["enabled", "interleaved"]:
                betas.append(AnthropicBetaVersion.INTERLEAVED_THINKING.value)

        # Check for skills configuration
        # Skills require both skills beta and code execution beta
        if "skills" in self.model_setting and isinstance(
            self.model_setting["skills"], list
        ):
            if self.model_setting["skills"]:
                betas.append(AnthropicBetaVersion.SKILLS.value)
                # Skills require code execution tool
                if AnthropicBetaVersion.CODE_EXECUTION.value not in betas:
                    betas.append(AnthropicBetaVersion.CODE_EXECUTION.value)

        # Check for file-related content in messages
        for message in input_messages:
            if isinstance(message.get("content"), list):
                for content in message["content"]:
                    if (
                        content.get("type") == "document"
                        and AnthropicBetaVersion.FILES_API.value not in betas
                    ):
                        betas.append(AnthropicBetaVersion.FILES_API.value)
                    elif (
                        content.get("type") == "container_upload"
                        and AnthropicBetaVersion.CODE_EXECUTION.value not in betas
                    ):
                        betas.append(AnthropicBetaVersion.CODE_EXECUTION.value)
        return betas

    def _process_input_files(
        self, input_files: List[Dict[str, Any]], input_messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Process and upload input files, attaching them to either code interpreter or user message.

        Args:
            input_files: List of file dictionaries containing file data
            input_messages: List of conversation messages

        Returns:
            Updated input_messages with file references
        """
        code_execution_tool = next(
            (
                tool
                for tool in self.model_setting.get("tools", [])
                if tool.get("type") == "code_execution_20250522"
            ),
            None,
        )

        # Upload each file to OpenAI and store metadata
        file_ids = []
        for input_file in input_files:
            uploaded_file = self.insert_file(**input_file)
            file_ids.append(uploaded_file.id)
            self.uploaded_files.append(
                {
                    "file_id": uploaded_file.id,
                    "code_execution_tool": True if code_execution_tool else False,
                }
            )

        # If code interpreter not available, attach to user message
        if input_messages and input_messages[-1]["role"] == "user":
            # Construct message content with original text and file references
            message_content = [{"type": "text", "text": input_messages[-1]["content"]}]

            if code_execution_tool:
                message_content.extend(
                    {"type": "container_upload", "file_id": file_id}
                    for file_id in file_ids
                )

            else:
                message_content.extend(
                    {"type": "document", "source": {"type": "file", "file_id": file_id}}
                    for file_id in file_ids
                )

            # Update the last message with combined content
            input_messages[-1]["content"] = message_content

        return input_messages

    def _update_conversation_history(
        self,
        function_call_data: Dict[str, Any],
        function_output: Any,
        input_messages: List[Dict[str, Any]],
    ) -> None:
        """
        Updates the conversation history with function call results.
        Formats and appends function output as a user message.

        Args:
            function_call_data: Metadata about the executed function
            function_output: Result from function execution
            input_messages: Current conversation history to update
        """

        input_messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": function_call_data[
                            "id"
                        ],  # from the API response
                        "content": str(function_output),  # from running your tool
                    }
                ],
            }
        )  # Append the function response

    def _provider_state(self, blocks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Signed replay state (plan §5.3): tagged with protocol + model."""
        state: Dict[str, Any] = {
            "api_protocol": self.api_protocol,
            "model": self.model_setting.get("model"),
            "blocks": blocks,
        }
        if self._container_id:
            state["container_id"] = self._container_id
        return state

    @staticmethod
    def _usage_from_parts(
        input_tokens: Any,
        output_tokens: Any,
        cache_read: Any = 0,
        cache_creation: Any = 0,
    ) -> Dict[str, int]:
        """Anthropic usage → plan §5.3 shape (incl. cache tokens, Q6)."""
        usage = {
            "prompt_tokens": int(input_tokens or 0),
            "completion_tokens": int(output_tokens or 0),
            "total_tokens": int(input_tokens or 0) + int(output_tokens or 0),
        }
        if cache_read:
            usage["cache_read_tokens"] = int(cache_read)
        if cache_creation:
            usage["cache_creation_tokens"] = int(cache_creation)
        return usage

    def _map_usage(self, raw: Any) -> Dict[str, int]:
        """Map an Anthropic SDK usage object/dict (plan Q6: only
        openai_completions reported usage before; the others needed it)."""
        if raw is None:
            return {}

        def _int(value: Any) -> int:
            try:
                return int(value or 0)
            except (TypeError, ValueError):
                return 0

        if isinstance(raw, dict):
            return self._usage_from_parts(
                _int(raw.get("input_tokens")),
                _int(raw.get("output_tokens")),
                _int(raw.get("cache_read_input_tokens")),
                _int(raw.get("cache_creation_input_tokens")),
            )
        return self._usage_from_parts(
            _int(getattr(raw, "input_tokens", 0)),
            _int(getattr(raw, "output_tokens", 0)),
            _int(getattr(raw, "cache_read_input_tokens", 0)),
            _int(getattr(raw, "cache_creation_input_tokens", 0)),
        )

    def to_native_messages(
        self, messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Neutral (OpenAI-style) → Anthropic native (plan §5.5 / A.2).

        - system-role messages are dropped: the system prompt flows through
          ``model_setting["system"]`` (from ``agent["instructions"]``);
        - assistant messages with ``provider_state.blocks`` replay verbatim
          (signed thinking / tool_use blocks); a skills ``container_id`` is
          latched for container continuity; stale state from a different
          protocol/model is dropped (state only, not the message);
        - neutral ``tool_calls`` (agent_engine round-trip shape) convert to
          ``tool_use`` blocks; ``tool`` messages become ``tool_result``.
        """
        native: List[Dict[str, Any]] = []
        for message in messages:
            role = str(message.get("role", "")).lower()

            if role == "system":
                continue

            if role == "assistant":
                state = message.get("provider_state")
                if isinstance(state, dict) and state.get("container_id"):
                    self._container_id = state.get("container_id")
                if (
                    isinstance(state, dict)
                    and state.get("api_protocol") == self.api_protocol
                    and state.get("blocks")
                ):
                    if state.get("model") not in (
                        None,
                        self.model_setting.get("model"),
                    ):
                        state = None  # model switched → drop stale state (§5.5)
                    else:
                        blocks: List[Dict[str, Any]] = []
                        for block in state.get("blocks") or []:
                            item = dict(block)
                            if (
                                item.get("type") == "tool_use"
                                and isinstance(item.get("input"), str)
                            ):
                                try:
                                    item["input"] = json.loads(item["input"] or "{}")
                                except json.JSONDecodeError:
                                    item["input"] = {}
                            blocks.append(item)
                        native.append({"role": "assistant", "content": blocks})
                        continue
                content: List[Dict[str, Any]] = []
                text = message.get("content") or ""
                if text:
                    content.append({"type": "text", "text": text})
                for tool_call in message.get("tool_calls") or []:
                    function = tool_call.get("function") or {}
                    arguments = function.get("arguments") or "{}"
                    if isinstance(arguments, str):
                        try:
                            tool_input = json.loads(arguments or "{}")
                        except json.JSONDecodeError:
                            tool_input = {}
                    else:
                        tool_input = dict(arguments or {})
                    content.append(
                        {
                            "type": "tool_use",
                            "id": str(tool_call.get("id") or ""),
                            "name": str(function.get("name") or ""),
                            "input": (
                                tool_input if isinstance(tool_input, dict) else {}
                            ),
                        }
                    )
                native.append(
                    {
                        "role": "assistant",
                        "content": content or [{"type": "text", "text": ""}],
                    }
                )
                continue

            if role == "tool":
                native.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": str(
                                    message.get("tool_call_id") or ""
                                ),
                                "content": str(message.get("content") or ""),
                            }
                        ],
                    }
                )
                continue

            native.append({"role": "user", "content": message.get("content") or ""})
        return native

    def _log_citation(
        self, citation: Any, idx: int, is_streaming: bool = False
    ) -> None:
        """
        Private helper to log a single citation.
        Handles both dict and object formats for citations.

        Args:
            citation: The citation object to log
            idx: The index of the citation (for display purposes)
            is_streaming: Whether this is from streaming mode
        """
        if not self.logger.isEnabledFor(logging.INFO):
            return

        # Check citation type
        citation_type = None
        if isinstance(citation, dict):
            citation_type = citation.get("type", "N/A")
        elif hasattr(citation, "type"):
            citation_type = citation.type

        # Only log web_search_result_location citations
        if citation_type != "web_search_result_location":
            return

        # Extract citation details - handle both dict and object formats
        if isinstance(citation, dict):
            url = citation.get("url", "N/A")
            title = citation.get("title", "N/A")
            cited_text = citation.get("cited_text", "N/A")
        else:
            url = citation.url if hasattr(citation, "url") else "N/A"
            title = citation.title if hasattr(citation, "title") else "N/A"
            cited_text = (
                citation.cited_text if hasattr(citation, "cited_text") else "N/A"
            )

        # Truncate cited text if needed
        cited_text_preview = (
            cited_text[:100]
            if isinstance(cited_text, str) and len(cited_text) > 100
            else cited_text
        )

        # Log based on mode
        if is_streaming:
            self.logger.info(
                f"Citation received in stream - URL: {url}, "
                f"Title: {title}, "
                f"Cited text: {cited_text_preview}..."
            )
        else:
            self.logger.info(
                f"  Citation {idx}: Title: {title}, "
                f"URL: {url}, "
                f"Cited text: {cited_text_preview}..."
            )

    def _handle_citations(
        self,
        citations_list: List[Any],
        text_preview: str = None,
        is_streaming: bool = False,
    ) -> None:
        """
        Private helper to handle citations from text blocks.
        Logs citation information for both streaming and non-streaming modes.

        Args:
            citations_list: List of citation objects
            text_preview: Optional preview of the text containing citations
            is_streaming: Whether this is from streaming mode
        """
        # P5（plan §5.3）：citations 收集进 final_output（capabilities
        # 含 citations 时由 adapter 经 LlmInvokeResultType.citations 透出）。
        collected = list(self.final_output.get("citations") or [])
        for citation in citations_list or []:
            if isinstance(citation, dict):
                collected.append(dict(citation))
            elif hasattr(citation, "model_dump"):
                collected.append(citation.model_dump(exclude_none=True))
            else:
                collected.append({"text": str(citation)})
        self.final_output["citations"] = collected

        if not self.logger.isEnabledFor(logging.INFO):
            return

        if not citations_list:
            return

        # Log header with citation count
        if is_streaming:
            self.logger.info(
                f"Text block with citations started - Citation count: {len(citations_list)}"
            )
        else:
            log_msg = f"Text with citations - Citation count: {len(citations_list)}"
            if text_preview:
                log_msg += f", Text: {text_preview[:100]}..."
            self.logger.info(log_msg)

        # Log first 3 citations
        for idx, citation in enumerate(citations_list[:3], 1):
            self._log_citation(citation, idx, is_streaming=is_streaming)

    def _handle_citation_delta(self, delta: Any) -> None:
        """
        Private helper to handle citation deltas in streaming mode.
        Logs individual citations as they arrive.

        Args:
            delta: The delta object containing citation information
        """
        if not self.logger.isEnabledFor(logging.INFO):
            return

        # Extract citation from delta
        citation = delta.citation if hasattr(delta, "citation") else None
        if citation:
            # Use _log_citation with streaming mode (idx not used for streaming single citations)
            self._log_citation(citation, idx=0, is_streaming=True)

    def _handle_server_tool_use(
        self,
        content: Any,
        tool_input: Dict[str, Any] = None,
        is_streaming: bool = False,
    ) -> None:
        """
        Private helper to handle server_tool_use blocks (web_search and web_fetch).
        Logs tool initiation for both streaming and non-streaming modes.

        Args:
            content: The server_tool_use content block (can be a dict or object)
            tool_input: Optional parsed tool input dictionary (for streaming when input is ready)
            is_streaming: Whether this is from streaming mode
        """
        if not self.logger.isEnabledFor(logging.INFO):
            return

        # Extract tool metadata - handle both dict and object formats
        if isinstance(content, dict):
            tool_id = content.get("id", "N/A")
            tool_name = content.get("name", "N/A")
            content_input = content.get("input", {})
        else:
            tool_id = content.id if hasattr(content, "id") else "N/A"
            tool_name = content.name if hasattr(content, "name") else "N/A"
            content_input = content.input if hasattr(content, "input") else {}

        # Use provided tool_input if available (for streaming), otherwise use content_input
        input_to_use = tool_input if tool_input is not None else content_input

        mode_prefix = "in stream" if is_streaming else ""

        # Log based on tool type
        if tool_name == "web_search":
            query = (
                input_to_use.get("query", "N/A")
                if isinstance(input_to_use, dict)
                else "N/A"
            )
            if is_streaming and tool_input is None:
                # Initial log when tool starts (before input is parsed)
                self.logger.info(
                    f"Server tool use started in stream - Name: {tool_name}, ID: {tool_id}"
                )
            else:
                # Log with query details
                log_msg = (
                    f"Web search query {mode_prefix}"
                    if is_streaming
                    else "Web search initiated"
                )
                self.logger.info(f"{log_msg} - ID: {tool_id}, Query: '{query}'")
        elif tool_name == "web_fetch":
            url = (
                input_to_use.get("url", "N/A")
                if isinstance(input_to_use, dict)
                else "N/A"
            )
            if is_streaming and tool_input is None:
                # Initial log when tool starts (before input is parsed)
                self.logger.info(
                    f"Server tool use started in stream - Name: {tool_name}, ID: {tool_id}"
                )
            else:
                # Log with URL details
                log_msg = (
                    f"Web fetch URL {mode_prefix}"
                    if is_streaming
                    else "Web fetch initiated"
                )
                self.logger.info(f"{log_msg} - ID: {tool_id}, URL: '{url}'")
        else:
            # Generic server tool logging
            if is_streaming and tool_input is None:
                self.logger.info(
                    f"Server tool use started in stream - Name: {tool_name}, ID: {tool_id}"
                )
            else:
                self.logger.info(
                    f"Server tool use - Name: {tool_name}, ID: {tool_id}, Input: {input_to_use}"
                )

    def _handle_web_search_tool_result(
        self, content: Any, is_streaming: bool = False
    ) -> None:
        """
        Private helper to handle web_search_tool_result blocks.
        Logs search results and errors for both streaming and non-streaming modes.

        Args:
            content: The web_search_tool_result content block
            is_streaming: Whether this is from streaming mode
        """
        tool_use_id = content.tool_use_id if hasattr(content, "tool_use_id") else "N/A"
        mode_prefix = "in stream" if is_streaming else ""

        # Check for errors first (error is in content object, not array)
        search_content = content.content if hasattr(content, "content") else None

        if not search_content:
            return

        # Check if content is an error object (dict or object)
        is_error = False
        if isinstance(search_content, dict):
            content_type = search_content.get("type", "")
            if content_type == "web_search_tool_result_error":
                is_error = True
                if self.logger.isEnabledFor(logging.ERROR):
                    error_code = search_content.get("error_code", "Unknown error")
                    self.logger.error(
                        f"Web search error {mode_prefix} - Tool Use ID: {tool_use_id}, "
                        f"Error code: {error_code}"
                    )
        elif (
            hasattr(search_content, "type")
            and search_content.type == "web_search_tool_result_error"
        ):
            is_error = True
            if self.logger.isEnabledFor(logging.ERROR):
                error_code = (
                    search_content.error_code
                    if hasattr(search_content, "error_code")
                    else "Unknown error"
                )
                self.logger.error(
                    f"Web search error {mode_prefix} - Tool Use ID: {tool_use_id}, "
                    f"Error code: {error_code}"
                )

        if is_error:
            return

        # If not error, content should be an array of results
        if self.logger.isEnabledFor(logging.INFO):
            if isinstance(search_content, list):
                result_count = len(search_content)

                self.logger.info(
                    f"Web search results received {mode_prefix} - Tool Use ID: {tool_use_id}, "
                    f"Results: {result_count}"
                )

                # Log details of each result from content array
                for idx, result in enumerate(search_content[:5], 1):  # Log first 5
                    # Handle both dict and object formats
                    result_type = (
                        result.get("type")
                        if isinstance(result, dict)
                        else (result.type if hasattr(result, "type") else None)
                    )

                    if result_type == "web_search_result":
                        if isinstance(result, dict):
                            result_detail = {
                                "url": result.get("url", "N/A"),
                                "title": result.get("title", "N/A"),
                                "page_age": result.get("page_age", "N/A"),
                            }
                        else:
                            result_detail = {
                                "url": result.url if hasattr(result, "url") else "N/A",
                                "title": (
                                    result.title if hasattr(result, "title") else "N/A"
                                ),
                                "page_age": (
                                    result.page_age
                                    if hasattr(result, "page_age")
                                    else "N/A"
                                ),
                            }
                        self.logger.info(
                            f"  Result {idx}: Title: {result_detail['title']}, "
                            f"URL: {result_detail['url']}, Page Age: {result_detail['page_age']}"
                        )

    def _handle_web_fetch_tool_result(
        self, content: Any, is_streaming: bool = False
    ) -> None:
        """
        Private helper to handle web_fetch_tool_result blocks.
        Logs fetch results and errors for both streaming and non-streaming modes.

        Args:
            content: The web_fetch_tool_result content block
            is_streaming: Whether this is from streaming mode
        """
        tool_use_id = content.tool_use_id if hasattr(content, "tool_use_id") else "N/A"
        mode_prefix = "in stream" if is_streaming else ""

        # Extract from nested structure: content.content (web_fetch_result)
        fetch_result = content.content if hasattr(content, "content") else None

        if not fetch_result:
            return

        # Check for errors first (error_code is inside fetch_result)
        is_error = False
        if isinstance(fetch_result, dict):
            fetch_type = fetch_result.get("type", "")
            if fetch_type == "web_fetch_tool_result_error":
                is_error = True
                if self.logger.isEnabledFor(logging.ERROR):
                    error_code = fetch_result.get("error_code", "Unknown error")
                    self.logger.error(
                        f"Web fetch error {mode_prefix} - Tool Use ID: {tool_use_id}, "
                        f"Error code: {error_code}"
                    )
        else:
            if (
                hasattr(fetch_result, "type")
                and fetch_result.type == "web_fetch_tool_error"
            ):
                is_error = True
                if self.logger.isEnabledFor(logging.ERROR):
                    error_code = (
                        fetch_result.error_code
                        if hasattr(fetch_result, "error_code")
                        else "Unknown error"
                    )
                    self.logger.error(
                        f"Web fetch error {mode_prefix} - Tool Use ID: {tool_use_id}, "
                        f"Error code: {error_code}"
                    )

        if is_error:
            return

        # Handle both dict and object formats for successful fetch
        if self.logger.isEnabledFor(logging.INFO):
            if isinstance(fetch_result, dict):
                url = fetch_result.get("url", "N/A")
                retrieved_at = fetch_result.get("retrieved_at", "N/A")
                result_content = fetch_result.get("content", {})

                # Extract from content.source
                if isinstance(result_content, dict):
                    source = result_content.get("source", {})
                    media_type = (
                        source.get("media_type", "N/A")
                        if isinstance(source, dict)
                        else "N/A"
                    )
                    data = (
                        source.get("data", "N/A") if isinstance(source, dict) else "N/A"
                    )
                    title = result_content.get("title", "N/A")
                else:
                    media_type = "N/A"
                    data = "N/A"
                    title = "N/A"
            else:
                url = fetch_result.url if hasattr(fetch_result, "url") else "N/A"
                retrieved_at = (
                    fetch_result.retrieved_at
                    if hasattr(fetch_result, "retrieved_at")
                    else "N/A"
                )

                # Extract from content.source
                result_content = (
                    fetch_result.content if hasattr(fetch_result, "content") else None
                )
                if result_content:
                    source = (
                        result_content.source
                        if hasattr(result_content, "source")
                        else None
                    )
                    media_type = (
                        source.media_type
                        if source and hasattr(source, "media_type")
                        else "N/A"
                    )
                    data = source.data if source and hasattr(source, "data") else "N/A"
                    title = (
                        result_content.title
                        if hasattr(result_content, "title")
                        else "N/A"
                    )
                else:
                    media_type = "N/A"
                    data = "N/A"
                    title = "N/A"

            content_preview = str(data)[:200] if data != "N/A" else "N/A"

            self.logger.info(
                f"Web fetch result received {mode_prefix} - Tool Use ID: {tool_use_id}, "
                f"URL: {url}, "
                f"Retrieved at: {retrieved_at}, "
                f"Title: {title}, "
                f"Media type: {media_type}, "
                f"Content preview: {content_preview}..."
            )

    def handle_response(
        self,
        response: Any,
        input_messages: List[Dict[str, Any]],
    ) -> None:
        """
        Processes a complete response from the model.
        Handles both text responses, tool use cases, MCP tool calls, thinking blocks,
        and pause_turn for long-running skill operations.

        Args:
            response: Complete response object from the model
            input_messages: Current conversation history to update
        """
        # Capture container ID for multi-turn conversations with skills
        if hasattr(response, "container") and response.container:
            self._container_id = response.container.id
            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(
                    f"[handle_response] Container ID captured: {self._container_id}"
                )

        # Handle pause_turn for long-running skill operations
        if response.stop_reason == "pause_turn":
            if self.logger.isEnabledFor(logging.INFO):
                self.logger.info(
                    "[handle_response] Received pause_turn - skill operation in progress"
                )

            # Add assistant's response to conversation and continue
            input_messages.append({"role": "assistant", "content": response.content})

            # Continue the conversation to let skill complete
            response = self.invoke_model(**{"input": input_messages, "stream": False})
            self.handle_response(response, input_messages)
            return

        if response.stop_reason == "tool_use":
            # Single-turn (plan D2): return the requested tool calls with the
            # signed replay state instead of executing tools in-handler.
            blocks: List[Dict[str, Any]] = []
            neutral_tool_calls: List[Dict[str, Any]] = []
            for content in response.content:
                if content.type == "thinking":
                    try:
                        thinking_text = (
                            content.thinking if hasattr(content, "thinking") else ""
                        )
                        thinking_signature = (
                            content.signature if hasattr(content, "signature") else None
                        )
                        if thinking_text:
                            thinking_block: Dict[str, Any] = {
                                "type": "thinking",
                                "thinking": thinking_text,
                            }
                            if thinking_signature:
                                thinking_block["signature"] = thinking_signature
                            blocks.append(thinking_block)
                            if self.final_output.get("reasoning_summary"):
                                self.final_output["reasoning_summary"] = (
                                    self.final_output["reasoning_summary"]
                                    + "\n"
                                    + thinking_text
                                )
                            else:
                                self.final_output["reasoning_summary"] = thinking_text
                            if self.logger.isEnabledFor(logging.DEBUG):
                                self.logger.debug(
                                    f"[handle_response] Captured thinking: {thinking_text[:100]}..."
                                )
                    except Exception as e:
                        if self.logger.isEnabledFor(logging.ERROR):
                            self.logger.error(f"Failed to process thinking block: {e}")
                        if not self.final_output.get("reasoning_summary"):
                            self.final_output["reasoning_summary"] = (
                                "Error processing thinking"
                            )
                elif content.type == "text":
                    blocks.append({"type": "text", "text": content.text})
                    self.assistant_messages.append(
                        {
                            "content": content.text,
                        }
                    )
                elif content.type == "tool_use":
                    blocks.append(
                        {
                            "type": "tool_use",
                            "id": content.id,
                            "name": content.name,
                            "input": content.input,
                        }
                    )
                    neutral_tool_calls.append(
                        {
                            "id": content.id,
                            "type": "function",
                            "function": {
                                "name": content.name,
                                "arguments": Serializer.json_dumps(
                                    content.input or {}
                                ),
                            },
                        }
                    )
            self.provider_state = self._provider_state(blocks)
            text_content = "".join(
                block["text"]
                for block in blocks
                if block.get("type") == "text" and block.get("text")
            )
            self.final_output.update(
                {
                    "message_id": response.id,
                    "role": response.role,
                    "content": text_content,
                    "tool_calls": neutral_tool_calls,
                    "provider_state": self.provider_state,
                    "finish_reason": "tool_calls",
                    "usage": self._map_usage(getattr(response, "usage", None)),
                }
            )
            return
        else:
            assistant_message_content = ""
            while self.assistant_messages:
                assistant_message = self.assistant_messages.pop()
                assistant_message_content += assistant_message["content"]

            final_content = ""
            output_files = []
            for content in response.content:
                # Handle thinking blocks
                if content.type == "thinking":
                    try:
                        thinking_text = (
                            content.thinking if hasattr(content, "thinking") else ""
                        )
                        if thinking_text:
                            # Store thinking summary in final_output
                            if self.final_output.get("reasoning_summary"):
                                self.final_output["reasoning_summary"] = (
                                    self.final_output["reasoning_summary"]
                                    + "\n"
                                    + thinking_text
                                )
                            else:
                                self.final_output["reasoning_summary"] = thinking_text

                            if self.logger.isEnabledFor(logging.DEBUG):
                                self.logger.debug(
                                    f"[handle_response] Captured thinking: {thinking_text[:100]}..."
                                )
                    except Exception as e:
                        if self.logger.isEnabledFor(logging.ERROR):
                            self.logger.error(f"Failed to process thinking block: {e}")
                        if not self.final_output.get("reasoning_summary"):
                            self.final_output["reasoning_summary"] = (
                                "Error processing thinking"
                            )
                    continue

                # Handle MCP tool use blocks
                elif content.type == "mcp_tool_use":
                    if self.logger.isEnabledFor(logging.INFO):
                        self.logger.info(
                            f"MCP tool call received - ID: {content.id}, Name: {content.name}, Server: {content.server_name}, Input: {content.input}."
                        )
                    continue
                    # Note: MCP tools are handled automatically by the API
                    # No need to manually execute them like regular tools
                # Handle MCP tool result blocks
                elif content.type == "mcp_tool_result":
                    if self.logger.isEnabledFor(logging.INFO):
                        self.logger.info(
                            f"MCP tool result received - Tool Use ID: {content.tool_use_id}, Content: {content.content}"
                        )
                    continue
                elif content.type == "code_execution_tool_result":
                    if content.content.type == "code_execution_result":
                        for file in content.content.content:
                            output_files.append({"file_id": file["file_id"]})
                    continue
                elif content.type == "BetaServerToolUseBlock":
                    continue
                elif content.type == "server_tool_use":
                    # Log server tool use (web_search and web_fetch)
                    # In non-streaming mode, input is already available in content.input
                    tool_input = (
                        content.input
                        if hasattr(content, "input")
                        else (content.get("input") if isinstance(content, dict) else {})
                    )
                    self._handle_server_tool_use(
                        content, tool_input=tool_input, is_streaming=False
                    )
                    continue
                elif content.type == "web_search_tool_result":
                    self._handle_web_search_tool_result(content, is_streaming=False)
                    continue
                elif content.type == "web_fetch_tool_result":
                    self._handle_web_fetch_tool_result(content, is_streaming=False)
                    continue
                elif not hasattr(content, "text"):
                    self.logger.error(
                        f"Unexpected content type in response: {content.type}"
                    )
                    continue

                # Log citations if present in text blocks
                if hasattr(content, "citations") and content.citations:
                    self._handle_citations(
                        content.citations, text_preview=content.text, is_streaming=False
                    )

                final_content += content.text

            if assistant_message_content:
                final_content = assistant_message_content + " " + final_content

            # Signed thinking must replay on the next turn (plan §5.3).
            signed_thinking = [
                {
                    "type": "thinking",
                    "thinking": getattr(content, "thinking", ""),
                    "signature": content.signature,
                }
                for content in response.content
                if getattr(content, "type", None) == "thinking"
                and getattr(content, "signature", None)
            ]
            state = (
                self._provider_state(signed_thinking) if signed_thinking else None
            )
            self.final_output.update(
                {
                    "message_id": response.id,
                    "role": response.role,
                    "content": final_content,
                    "output_files": output_files,
                    "usage": self._map_usage(getattr(response, "usage", None)),
                    "finish_reason": response.stop_reason or "stop",
                    "provider_state": state,
                }
            )

    def handle_stream(
        self,
        response_stream: Any,
        input_messages: List[Dict[str, Any]] = None,
        stream_event: threading.Event = None,
    ) -> None:
        """
        Processes streaming responses from the model chunk by chunk.
        Handles text content, tool use, MCP tool calls, and maintains state across chunks.

        Note: MCP tools are processed automatically by the MCP connector and don't require
        manual result handling or recursive calls.

        Args:
            response_stream: Iterator yielding response chunks
            input_messages: Optional conversation history to update
            stream_event: Optional event to signal stream completion
        """
        message_id = None
        json_input_parts = []
        stop_reason = None
        tool_use_data = None
        mcp_tool_use_data = None
        mcp_tool_result_data = None
        server_tool_use_data = None
        output_files = []
        self.accumulated_text = ""
        accumulated_partial_json = ""
        accumulated_partial_text = ""
        # Use cached output format type (performance optimization)
        output_format = self.output_format_type

        # Reasoning tracking variables (matching OpenAI and Ollama handler patterns)
        reasoning_no = 0
        reasoning_index = 0
        accumulated_reasoning_parts = []  # For display/logging (just text)
        accumulated_reasoning_blocks = []  # Complete blocks with signatures for API
        accumulated_partial_reasoning_text = ""
        reasoning_started = False
        current_reasoning_block_index = None
        current_reasoning_text_parts = []  # Accumulate text for current thinking block
        current_reasoning_signature = ""  # Track signature for current thinking block

        # Token usage capture (plan Q6 / §5.3: input from message_start,
        # output from the final message_delta).
        stream_usage_input = 0
        stream_usage_output = 0
        stream_usage_cache_read = 0
        stream_usage_cache_creation = 0

        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug(
                f"[handle_stream] Initialized stream state - "
                f"accumulated_partial_json: '{accumulated_partial_json}', "
                f"output_format: '{output_format}'"
            )

        index = 0

        if self.assistant_messages:
            index = self.assistant_messages[-1]["index"]
            self.send_data_to_stream(
                index=index,
                data_format=output_format,
                chunk_delta=" ",
            )
            index += 1

        for chunk in response_stream:
            # Stop streaming if the client disconnected mid-generation.
            if self.is_stream_cancelled():
                self.logger.info(
                    "Stream cancelled (client disconnected); stopping generation."
                )
                break
            # Handle message start event
            if chunk.type == "message_start":
                message_id = chunk.message.id
                _start_usage = getattr(chunk.message, "usage", None)
                if _start_usage is not None:
                    stream_usage_input = int(
                        getattr(_start_usage, "input_tokens", 0) or 0
                    )
                    _read = int(
                        getattr(_start_usage, "cache_read_input_tokens", 0) or 0
                    )
                    if _read:
                        stream_usage_cache_read = _read
                    _created = int(
                        getattr(
                            _start_usage, "cache_creation_input_tokens", 0
                        )
                        or 0
                    )
                    if _created:
                        stream_usage_cache_creation = _created
                # Capture container ID for multi-turn conversations with skills
                if hasattr(chunk.message, "container") and chunk.message.container:
                    self._container_id = chunk.message.container.id
                    if self.logger.isEnabledFor(logging.DEBUG):
                        self.logger.debug(
                            f"[handle_stream] Container ID captured: {self._container_id}"
                        )

            # Handle thinking/reasoning block start
            elif (
                chunk.type == "content_block_start"
                and hasattr(chunk.content_block, "type")
                and chunk.content_block.type == "thinking"
            ):
                # Start reasoning block
                reasoning_started = True
                current_reasoning_block_index = (
                    chunk.index if hasattr(chunk, "index") else 0
                )
                current_reasoning_text_parts = []  # Reset for new thinking block
                current_reasoning_signature = (
                    ""  # Reset signature for new thinking block
                )

                if self.enable_timeline_log and self.logger.isEnabledFor(logging.INFO):
                    elapsed = self._get_elapsed_time()
                    self.logger.info(f"[TIMELINE] T+{elapsed:.2f}ms: Reasoning started")

            # Handle thinking/reasoning text deltas
            elif (
                chunk.type == "content_block_delta"
                and hasattr(chunk.delta, "type")
                and chunk.delta.type == "thinking_delta"
            ):
                if not hasattr(chunk.delta, "thinking") or not chunk.delta.thinking:
                    continue

                reasoning_text = chunk.delta.thinking

                # Accumulate reasoning text for current block and for display
                current_reasoning_text_parts.append(reasoning_text)
                accumulated_reasoning_parts.append(reasoning_text)
                accumulated_partial_reasoning_text += reasoning_text

                # Process and send reasoning text
                reasoning_index, accumulated_partial_reasoning_text = (
                    self.process_text_content(
                        reasoning_index,
                        accumulated_partial_reasoning_text,
                        output_format,
                        suffix=f"rs#{reasoning_no}",
                    )
                )

            # Handle thinking signature deltas (encrypted signature for thinking blocks)
            elif (
                chunk.type == "content_block_delta"
                and hasattr(chunk.delta, "type")
                and chunk.delta.type == "signature_delta"
            ):
                if hasattr(chunk.delta, "signature"):
                    current_reasoning_signature += chunk.delta.signature

            # Handle content block start for text (to capture citations metadata)
            elif (
                chunk.type == "content_block_start"
                and hasattr(chunk.content_block, "type")
                and chunk.content_block.type == "text"
            ):
                # Synchronize index with reasoning_index when transitioning from reasoning to regular content
                # This ensures sequential ordering when reasoning blocks complete before text content starts
                if index == 0 and reasoning_index > 0:
                    index = reasoning_index + 1

                # Check if this text block has citations
                if (
                    hasattr(chunk.content_block, "citations")
                    and chunk.content_block.citations
                ):
                    self._handle_citations(
                        chunk.content_block.citations,
                        text_preview=None,
                        is_streaming=True,
                    )

            # Handle citations delta
            elif (
                chunk.type == "content_block_delta"
                and hasattr(chunk.delta, "type")
                and chunk.delta.type == "citations_delta"
            ):
                self._handle_citation_delta(chunk.delta)

            # Handle text content
            elif chunk.type == "content_block_delta" and hasattr(chunk.delta, "text"):
                if not chunk.delta.text:
                    continue

                if output_format in ["json_object", "json_schema"]:
                    accumulated_partial_json += chunk.delta.text
                    index, self.accumulated_text, accumulated_partial_json = (
                        self.process_and_send_json(
                            index,
                            self.accumulated_text,
                            accumulated_partial_json,
                            output_format,
                        )
                    )
                elif mcp_tool_result_data:
                    mcp_tool_result_data["content"] += chunk.delta.text
                else:
                    self.accumulated_text += chunk.delta.text
                    accumulated_partial_text += chunk.delta.text
                    # Check if text contains XML-style tags and update format
                    index, accumulated_partial_text = self.process_text_content(
                        index, accumulated_partial_text, output_format
                    )

            # Handle server tool use (web_search)
            elif (
                chunk.type == "content_block_start"
                and hasattr(chunk.content_block, "type")
                and chunk.content_block.type == "server_tool_use"
            ):
                server_tool_use_data = {
                    "id": chunk.content_block.id,
                    "name": (
                        chunk.content_block.name
                        if hasattr(chunk.content_block, "name")
                        else "N/A"
                    ),
                    "input": {},  # Will be populated from JSON deltas
                }
                # Log initial server tool use (before input is parsed)
                self._handle_server_tool_use(
                    server_tool_use_data, tool_input=None, is_streaming=True
                )

            # Handle regular tool use start
            elif (
                chunk.type == "content_block_start"
                and hasattr(chunk.content_block, "type")
                and chunk.content_block.type == "tool_use"
            ):
                tool_use_data = {
                    "id": chunk.content_block.id,
                    "type": chunk.content_block.type,
                    "name": chunk.content_block.name,
                    "input": {},  # Will be populated from JSON deltas
                }

            #! Handle MCP tool use start
            elif (
                chunk.type == "content_block_start"
                and hasattr(chunk.content_block, "type")
                and chunk.content_block.type == "mcp_tool_call"
            ):
                mcp_tool_use_data = {
                    "id": chunk.content_block.id,
                    "server_name": chunk.content_block.server_name,
                    "name": chunk.content_block.name,
                    "input": {},  # Will be populated from JSON deltas
                }

            #! Handle MCP tool result blocks
            elif (
                chunk.type == "content_block_start"
                and hasattr(chunk.content_block, "type")
                and chunk.content_block.type == "mcp_tool_result"
            ):
                mcp_tool_result_data = {
                    "tool_use_id": chunk.content_block.tool_use_id,
                    "is_error": chunk.content_block.is_error,
                    "content": (
                        chunk.content_block.content
                        if hasattr(chunk.content_block, "content")
                        else ""
                    ),
                }
                if chunk.content_block.is_error:
                    if self.logger.isEnabledFor(logging.INFO):
                        self.logger.info(
                            f"MCP tool result received - Tool Use ID: {mcp_tool_result_data['tool_use_id']}, Error: {mcp_tool_result_data['content']}"
                        )
                    raise Exception(
                        f"MCP tool error: {mcp_tool_result_data['content']}"
                    )
            elif (
                chunk.type == "content_block_start"
                and hasattr(chunk.content_block, "type")
                and chunk.content_block.type == "code_execution_tool_result"
            ):
                # Handle code execution results in streaming
                if hasattr(chunk.content_block, "content"):
                    # Look for files in the execution result
                    for file in chunk.content_block.content.contnet:
                        output_files.append({"file_id": file["file_id"]})

            elif (
                chunk.type == "content_block_start"
                and hasattr(chunk.content_block, "type")
                and chunk.content_block.type == "web_search_tool_result"
            ):
                # Handle web search results in streaming
                self._handle_web_search_tool_result(
                    chunk.content_block, is_streaming=True
                )
                continue

            elif (
                chunk.type == "content_block_start"
                and hasattr(chunk.content_block, "type")
                and chunk.content_block.type == "web_fetch_tool_result"
            ):
                # Handle web fetch results in streaming
                self._handle_web_fetch_tool_result(
                    chunk.content_block, is_streaming=True
                )
                continue

            # Handle tool input JSON parts (for both regular and MCP tools)
            elif (
                chunk.type == "content_block_delta"
                and hasattr(chunk.delta, "type")
                and chunk.delta.type == "input_json_delta"
            ):
                json_input_parts.append(chunk.delta.partial_json)

            # Handle content block stop - parse JSON for server tools here and close reasoning blocks
            elif chunk.type == "content_block_stop":
                # Check if we're closing a reasoning block
                if reasoning_started and (
                    current_reasoning_block_index is None
                    or (
                        hasattr(chunk, "index")
                        and chunk.index == current_reasoning_block_index
                    )
                ):
                    # Send any remaining reasoning text
                    if len(accumulated_partial_reasoning_text) > 0:
                        self.send_data_to_stream(
                            index=reasoning_index,
                            data_format=output_format,
                            chunk_delta=accumulated_partial_reasoning_text,
                            suffix=f"rs#{reasoning_no}",
                        )
                        accumulated_partial_reasoning_text = ""
                        # reasoning_index += 1

                    # Create complete thinking block with signature for API
                    current_thinking_text = "".join(current_reasoning_text_parts)
                    if current_thinking_text:
                        thinking_block = {
                            "type": "thinking",
                            "thinking": current_thinking_text,
                        }
                        # Add signature if present (required by API)
                        if current_reasoning_signature:
                            thinking_block["signature"] = current_reasoning_signature
                        accumulated_reasoning_blocks.append(thinking_block)

                    # # End reasoning block
                    # self.send_data_to_stream(
                    #     index=reasoning_index,
                    #     data_format=output_format,
                    #     chunk_delta=f"<ReasoningEnd Id={reasoning_no}/>",
                    #     suffix=f"rs#{reasoning_no}",
                    # )
                    # reasoning_no += 1
                    reasoning_started = False
                    current_reasoning_block_index = None

                    if self.enable_timeline_log and self.logger.isEnabledFor(
                        logging.INFO
                    ):
                        elapsed = self._get_elapsed_time()
                        self.logger.info(
                            f"[TIMELINE] T+{elapsed:.2f}ms: Reasoning ended"
                        )

                # If we have server tool use data and JSON parts, parse and log immediately
                if server_tool_use_data and json_input_parts:
                    try:
                        json_str = "".join(json_input_parts).strip()
                        if json_str:
                            parsed_input = Serializer.json_loads(json_str)
                            server_tool_use_data["input"] = parsed_input

                            # Log server tool with parsed input
                            self._handle_server_tool_use(
                                server_tool_use_data,
                                tool_input=parsed_input,
                                is_streaming=True,
                            )
                    except json.JSONDecodeError as e:
                        if self.logger.isEnabledFor(logging.ERROR):
                            self.logger.error(f"Error parsing server tool JSON: {e}")

                    # Clear the data and JSON parts for the next tool
                    server_tool_use_data = None
                    json_input_parts = []

                # Parse and clear JSON parts for regular tool_use when their blocks end
                # This prevents JSON accumulation across multiple tool calls
                elif tool_use_data and json_input_parts:
                    try:
                        json_str = "".join(json_input_parts).strip()
                        if json_str:
                            parsed_input = Serializer.json_loads(json_str)
                            tool_use_data["input"] = parsed_input
                    except json.JSONDecodeError as e:
                        if self.logger.isEnabledFor(logging.ERROR):
                            self.logger.error(f"Error parsing tool_use JSON: {e}")
                    # Clear JSON parts for next tool
                    json_input_parts = []

                # Parse and clear JSON parts for MCP tool_use when their blocks end
                # This prevents JSON accumulation across multiple tool calls
                elif mcp_tool_use_data and json_input_parts:
                    try:
                        json_str = "".join(json_input_parts).strip()
                        if json_str:
                            parsed_input = Serializer.json_loads(json_str)
                            mcp_tool_use_data["input"] = parsed_input
                    except json.JSONDecodeError as e:
                        if self.logger.isEnabledFor(logging.ERROR):
                            self.logger.error(f"Error parsing mcp_tool_use JSON: {e}")
                    # Clear JSON parts for next tool
                    json_input_parts = []

            # Handle message delta for stop reason
            elif chunk.type == "message_delta":
                stop_reason = chunk.delta.stop_reason
                _delta_usage = getattr(chunk.delta, "usage", None)
                if _delta_usage is not None:
                    stream_usage_output = int(
                        getattr(_delta_usage, "output_tokens", 0) or 0
                    )
                    _read = int(
                        getattr(_delta_usage, "cache_read_input_tokens", 0) or 0
                    )
                    if _read:
                        stream_usage_cache_read = _read
                    _created = int(
                        getattr(
                            _delta_usage, "cache_creation_input_tokens", 0
                        )
                        or 0
                    )
                    if _created:
                        stream_usage_cache_creation = _created

            elif chunk.type == "message_stop":
                if len(accumulated_partial_text) > 0:
                    self.send_data_to_stream(
                        index=index,
                        data_format=output_format,
                        chunk_delta=accumulated_partial_text,
                    )
                    accumulated_partial_text = ""
                    index += 1

        # Process JSON input if we have tool use (regular, MCP, or server)
        if (
            tool_use_data or mcp_tool_use_data or server_tool_use_data
        ) and json_input_parts:
            try:
                # Join JSON parts and parse
                json_str = "".join(json_input_parts).strip()

                # Only parse if we have actual content
                if json_str:
                    parsed_input = Serializer.json_loads(json_str)

                    if tool_use_data:
                        tool_use_data["input"] = parsed_input
                    elif mcp_tool_use_data:
                        mcp_tool_use_data["input"] = parsed_input
                    elif server_tool_use_data:
                        server_tool_use_data["input"] = parsed_input
                        # Log server tool with parsed input
                        self._handle_server_tool_use(
                            server_tool_use_data,
                            tool_input=parsed_input,
                            is_streaming=True,
                        )
                else:
                    # Empty JSON string, set empty dict as input
                    if self.logger.isEnabledFor(logging.WARNING):
                        self.logger.warning(
                            "Empty JSON input for tool call, using empty dict"
                        )
                    if tool_use_data:
                        tool_use_data["input"] = {}
                    elif mcp_tool_use_data:
                        mcp_tool_use_data["input"] = {}
                    elif server_tool_use_data:
                        server_tool_use_data["input"] = {}

            except json.JSONDecodeError as e:
                # Log the actual JSON string that failed to parse for debugging
                if self.logger.isEnabledFor(logging.ERROR):
                    self.logger.error(
                        f"Error parsing tool input JSON. JSON string: '{json_str}', Error: {e}"
                    )
                raise Exception(f"Error parsing tool input JSON: {e}")

        # Handle pause_turn for long-running skill operations
        if stop_reason == "pause_turn":
            if self.logger.isEnabledFor(logging.INFO):
                self.logger.info(
                    "[handle_stream] Received pause_turn - skill operation in progress"
                )

            # Build content from current state
            content = []
            if accumulated_reasoning_blocks:
                content.extend(accumulated_reasoning_blocks)
            if self.accumulated_text:
                content.append({"type": "text", "text": self.accumulated_text})

            # Add assistant's response to conversation and continue
            if content:
                input_messages.append({"role": "assistant", "content": content})

            # Continue the conversation to let skill complete
            response = self.invoke_model(
                **{"input": input_messages, "stream": bool(stream_event)}
            )
            self.handle_stream(
                response, input_messages=input_messages, stream_event=stream_event
            )
            return

        # Handle regular tool usage — single-turn: return the requested
        # tool calls with signed replay state instead of executing
        # in-handler (plan D2).
        if stop_reason == "tool_use" and tool_use_data:
            blocks: List[Dict[str, Any]] = []
            if accumulated_reasoning_blocks:
                blocks.extend(accumulated_reasoning_blocks)
            if self.accumulated_text:
                blocks.append({"type": "text", "text": self.accumulated_text})
                self.assistant_messages.append(
                    {
                        "content": self.accumulated_text,
                        "index": index,
                    }
                )
            blocks.append(dict(tool_use_data))
            neutral_tool_calls = [
                {
                    "id": tool_use_data.get("id") or "",
                    "type": "function",
                    "function": {
                        "name": tool_use_data.get("name") or "",
                        "arguments": Serializer.json_dumps(
                            tool_use_data.get("input") or {}
                        ),
                    },
                }
            ]
            self.provider_state = self._provider_state(blocks)
            self.send_data_to_stream(
                index=index,
                data_format=output_format,
                is_message_end=True,
            )
            self.final_output.update(
                {
                    "message_id": message_id,
                    "role": "assistant",
                    "content": self.accumulated_text or "",
                    "tool_calls": neutral_tool_calls,
                    "provider_state": self.provider_state,
                    "finish_reason": "tool_calls",
                    "usage": self._usage_from_parts(
                        stream_usage_input,
                        stream_usage_output,
                        stream_usage_cache_read,
                        stream_usage_cache_creation,
                    ),
                }
            )
            if stream_event:
                stream_event.set()
            return

        # Handle MCP tool usage - MCP tools are processed automatically by the connector
        # We just need to track them for logging/display purposes
        elif mcp_tool_use_data:
            # MCP tools are handled automatically, no recursive call needed
            if self.logger.isEnabledFor(logging.INFO):
                self.logger.info(
                    f"MCP tool '{mcp_tool_use_data.get('name')}' from server '{mcp_tool_use_data.get('server_name')}' was executed"
                )
        elif mcp_tool_result_data:
            # MCP tool results are handled automatically, no need to process them here
            if self.logger.isEnabledFor(logging.INFO):
                self.logger.info(
                    f"MCP tool result received - Tool Use ID: {mcp_tool_result_data.get('tool_use_id')}, Content: {mcp_tool_result_data.get('content')}"
                )

        self.send_data_to_stream(
            index=index,
            data_format=output_format,
            is_message_end=True,
        )

        assistant_message_content = ""
        while self.assistant_messages:
            assistant_message = self.assistant_messages.pop()
            assistant_message_content += assistant_message["content"]

        if assistant_message_content:
            self.accumulated_text = (
                assistant_message_content + " " + self.accumulated_text
            )

        # Store accumulated reasoning summary if present
        if accumulated_reasoning_parts:
            final_reasoning_text = "".join(accumulated_reasoning_parts)
            if self.final_output.get("reasoning_summary"):
                # Accumulate reasoning from multiple rounds (e.g., function calls)
                self.final_output["reasoning_summary"] = (
                    self.final_output["reasoning_summary"] + "\n" + final_reasoning_text
                )
            else:
                self.final_output["reasoning_summary"] = final_reasoning_text

            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(
                    f"[handle_stream] Stored reasoning summary: {final_reasoning_text[:100]}..."
                )

        signed_thinking = [
            block
            for block in (accumulated_reasoning_blocks or [])
            if block.get("signature")
        ]
        state = (
            self._provider_state(signed_thinking) if signed_thinking else None
        )
        self.final_output.update(
            {
                "message_id": message_id,
                "role": "assistant",
                "content": self.accumulated_text,
                "output_files": output_files,
                "usage": self._usage_from_parts(
                    stream_usage_input,
                    stream_usage_output,
                    stream_usage_cache_read,
                    stream_usage_cache_creation,
                ),
                "finish_reason": stop_reason or "stop",
                "provider_state": state,
            }
        )

        # Signal that streaming has finished
        if stream_event:
            stream_event.set()

    def insert_file(self, **kwargs: Dict[str, Any]) -> Any:
        if "encoded_content" in kwargs:
            encoded_content = kwargs["encoded_content"]
            # Decode the Base64 string
            decoded_content = base64.b64decode(encoded_content)

            # Save the decoded content into a BytesIO object
            content_io = BytesIO(decoded_content)

            # Assign a filename to the BytesIO object
            content_io.name = kwargs["filename"]
        elif "file_uri" in kwargs:
            # Pooled client injected by llm_engine (plan D7) — handlers build
            # no HTTP clients (Iron Rule #5).
            if self.ctx is None or getattr(self.ctx, "http_client", None) is None:
                raise Exception(
                    "insert_file requires ctx.http_client (pooled client)"
                )
            content_io = BytesIO(
                self.ctx.http_client.get(kwargs["file_uri"]).content
            )
            content_io.name = kwargs["filename"]
        else:
            raise Exception("No file content provided")

        file = self.client.beta.files.upload(
            file=(kwargs["filename"], content_io, kwargs["mime_type"]),
        )
        return file

    def get_file(self, **kwargs: Dict[str, Any]) -> Dict[str, Any]:
        file = self.client.beta.files.retrieve_metadata(kwargs["file_id"])
        uploaded_file = {
            "id": file.id,
            "type": file.type,
            "filename": file.filename,
            "size_bytes": file.size_bytes,
            "created_at": pendulum.from_timestamp(file.created_at, tz="UTC"),
            "mime_type": file.mime_type,
            "downloadable": file.downloadable,
        }
        if (
            "encoded_content" in kwargs
            and kwargs["encoded_content"]
            and file.downloadable
        ):
            response = self.client.beta.files.download(kwargs["file_id"])
            content = response.content  # Get the actual bytes data)
            # Convert the content to a Base64-encoded string
            uploaded_file["encoded_content"] = base64.b64encode(content).decode("utf-8")

        return uploaded_file

    # ----------------------------
    # Skill Management Methods
    # ----------------------------

    def list_skills(
        self, source: Optional[str] = None, limit: int = 20, after: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        List all available skills.

        Args:
            source: Filter by skill source - "anthropic" for pre-built skills,
                   "custom" for user-uploaded skills, or None for all
            limit: Maximum number of skills to return (default 20)
            after: Cursor for pagination

        Returns:
            Dictionary containing:
                - data: List of skill objects
                - has_more: Boolean indicating if more results exist
                - first_id: ID of first skill in results
                - last_id: ID of last skill in results
        """
        params = {"betas": [AnthropicBetaVersion.SKILLS.value], "limit": limit}
        if source:
            params["source"] = source
        if after:
            params["after"] = after

        skills = self.client.beta.skills.list(**params)

        return {
            "data": [
                {
                    "id": skill.id,
                    "display_title": skill.display_title,
                    "source": skill.source,
                    "latest_version": skill.latest_version,
                    "created_at": skill.created_at,
                }
                for skill in skills.data
            ],
            "has_more": skills.has_more,
            "first_id": skills.first_id,
            "last_id": skills.last_id,
        }

    def get_skill(self, skill_id: str) -> Dict[str, Any]:
        """
        Retrieve details about a specific skill.

        Args:
            skill_id: The ID of the skill to retrieve

        Returns:
            Dictionary containing skill details
        """
        skill = self.client.beta.skills.retrieve(
            skill_id=skill_id, betas=[AnthropicBetaVersion.SKILLS.value]
        )

        return {
            "id": skill.id,
            "display_title": skill.display_title,
            "source": skill.source,
            "latest_version": skill.latest_version,
            "created_at": skill.created_at,
        }

    def create_skill(
        self, display_title: str, files: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Create a new custom skill.

        Args:
            display_title: Display name for the skill
            files: List of file dictionaries with keys:
                - filename: Name of the file (must include path, e.g., "skill_dir/SKILL.md")
                - content: File content as bytes or file-like object
                - mime_type: Optional MIME type

        Returns:
            Dictionary containing created skill details

        Note:
            - Must include a SKILL.md file at the top level
            - All files must specify a common root directory in their paths
            - Total upload size must be under 8MB
        """
        # Convert file dictionaries to tuples for the API
        file_tuples = []
        for file_dict in files:
            if "encoded_content" in file_dict:
                content = base64.b64decode(file_dict["encoded_content"])
            else:
                content = file_dict.get("content", b"")

            file_tuple = (
                file_dict["filename"],
                BytesIO(content) if isinstance(content, bytes) else content,
            )
            if "mime_type" in file_dict:
                file_tuple = file_tuple + (file_dict["mime_type"],)
            file_tuples.append(file_tuple)

        skill = self.client.beta.skills.create(
            display_title=display_title,
            files=file_tuples,
            betas=[AnthropicBetaVersion.SKILLS.value],
        )

        return {
            "id": skill.id,
            "display_title": skill.display_title,
            "source": skill.source,
            "latest_version": skill.latest_version,
            "created_at": skill.created_at,
        }

    def delete_skill(self, skill_id: str) -> Dict[str, Any]:
        """
        Delete a custom skill.

        Args:
            skill_id: The ID of the skill to delete

        Returns:
            Dictionary confirming deletion

        Note:
            All versions of the skill must be deleted before the skill can be deleted.
        """
        # First delete all versions
        versions = self.client.beta.skills.versions.list(
            skill_id=skill_id, betas=[AnthropicBetaVersion.SKILLS.value]
        )

        for version in versions.data:
            self.client.beta.skills.versions.delete(
                skill_id=skill_id,
                version=version.version,
                betas=[AnthropicBetaVersion.SKILLS.value],
            )

        # Then delete the skill
        self.client.beta.skills.delete(
            skill_id=skill_id, betas=[AnthropicBetaVersion.SKILLS.value]
        )

        return {"id": skill_id, "deleted": True}

    def list_skill_versions(
        self, skill_id: str, limit: int = 20, after: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        List all versions of a skill.

        Args:
            skill_id: The ID of the skill
            limit: Maximum number of versions to return (default 20)
            after: Cursor for pagination

        Returns:
            Dictionary containing:
                - data: List of version objects
                - has_more: Boolean indicating if more results exist
        """
        params = {"betas": [AnthropicBetaVersion.SKILLS.value], "limit": limit}
        if after:
            params["after"] = after

        versions = self.client.beta.skills.versions.list(skill_id=skill_id, **params)

        return {
            "data": [
                {
                    "version": version.version,
                    "created_at": version.created_at,
                }
                for version in versions.data
            ],
            "has_more": versions.has_more,
        }

    def create_skill_version(
        self, skill_id: str, files: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Create a new version of an existing skill.

        Args:
            skill_id: The ID of the skill to update
            files: List of file dictionaries with keys:
                - filename: Name of the file
                - content: File content as bytes or file-like object
                - mime_type: Optional MIME type

        Returns:
            Dictionary containing created version details
        """
        # Convert file dictionaries to tuples for the API
        file_tuples = []
        for file_dict in files:
            if "encoded_content" in file_dict:
                content = base64.b64decode(file_dict["encoded_content"])
            else:
                content = file_dict.get("content", b"")

            file_tuple = (
                file_dict["filename"],
                BytesIO(content) if isinstance(content, bytes) else content,
            )
            if "mime_type" in file_dict:
                file_tuple = file_tuple + (file_dict["mime_type"],)
            file_tuples.append(file_tuple)

        version = self.client.beta.skills.versions.create(
            skill_id=skill_id,
            files=file_tuples,
            betas=[AnthropicBetaVersion.SKILLS.value],
        )

        return {
            "version": version.version,
            "created_at": version.created_at,
        }

    def delete_skill_version(self, skill_id: str, version: str) -> Dict[str, Any]:
        """
        Delete a specific version of a skill.

        Args:
            skill_id: The ID of the skill
            version: The version to delete

        Returns:
            Dictionary confirming deletion
        """
        self.client.beta.skills.versions.delete(
            skill_id=skill_id,
            version=version,
            betas=[AnthropicBetaVersion.SKILLS.value],
        )

        return {"skill_id": skill_id, "version": version, "deleted": True}

    def reset_container(self) -> None:
        """
        Reset the container ID, useful when starting a new conversation
        that should not share container state with previous conversations.
        """
        self._container_id = None
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug("[reset_container] Container ID reset")
