"""Unit tests for inspect_model.py."""

import pytest

from inspect_history.inspect_model import (
    InspectEntry,
    PartDetail,
    UsageDetail,
    build_inspect_entries,
    classify_tool,
)


class TestClassifyTool:
    """Tests for classify_tool()."""

    def test_write_tools(self):
        assert classify_tool("create_file") == "[W]"
        assert classify_tool("edit_file") == "[W]"
        assert classify_tool("replace_in_file") == "[W]"
        assert classify_tool("delete_file") == "[W]"
        assert classify_tool("delete_snippet") == "[W]"

    def test_shell_tool(self):
        assert classify_tool("agent_run_shell_command") == "[!]"

    def test_browser_tools(self):
        assert classify_tool("browser_navigate") == "[B]"
        assert classify_tool("browser_click") == "[B]"

    def test_terminal_tools(self):
        assert classify_tool("terminal_run") == "[T]"
        assert classify_tool("terminal_write") == "[T]"

    def test_other_tools(self):
        assert classify_tool("read_file") == "[.]"
        assert classify_tool("list_files") == "[.]"
        assert classify_tool("grep") == "[.]"


class TestPartDetail:
    """Tests for PartDetail dataclass."""

    def test_basic_creation(self):
        part = PartDetail(part_kind="text", content="Hello world")
        assert part.part_kind == "text"
        assert part.content == "Hello world"
        assert part.tool_name is None
        assert part.tool_call_id is None

    def test_tool_call_part(self):
        part = PartDetail(
            part_kind="tool-call",
            content="read_file(...)",
            tool_name="read_file",
            tool_call_id="tc_123",
            tool_args={"file_path": "/test.py"},
        )
        assert part.tool_name == "read_file"
        assert part.tool_call_id == "tc_123"
        assert part.tool_args == {"file_path": "/test.py"}

    def test_role_color(self):
        assert PartDetail(part_kind="text", content="").role_color == "fg:ansiblue"
        assert (
            PartDetail(part_kind="user-prompt", content="").role_color == "fg:ansigreen"
        )
        assert (
            PartDetail(part_kind="system-prompt", content="").role_color
            == "bold fg:ansicyan"
        )
        assert (
            PartDetail(part_kind="tool-call", content="").role_color == "fg:ansiyellow"
        )
        assert (
            PartDetail(part_kind="thinking", content="").role_color == "fg:ansimagenta"
        )
        assert (
            PartDetail(part_kind="unknown", content="").role_color
            == "fg:ansibrightblack"
        )


class TestUsageDetail:
    """Tests for UsageDetail dataclass."""

    def test_default_values(self):
        usage = UsageDetail()
        assert usage.input_tokens == 0
        assert usage.output_tokens == 0
        assert usage.total_tokens == 0
        assert usage.has_cache is False
        assert usage.has_audio is False

    def test_total_tokens(self):
        usage = UsageDetail(input_tokens=100, output_tokens=50)
        assert usage.total_tokens == 150

    def test_has_cache(self):
        assert UsageDetail(cache_write_tokens=10).has_cache is True
        assert UsageDetail(cache_read_tokens=10).has_cache is True
        assert UsageDetail().has_cache is False

    def test_has_audio(self):
        assert UsageDetail(input_audio_tokens=10).has_audio is True
        assert UsageDetail(output_audio_tokens=10).has_audio is True
        assert UsageDetail(cache_audio_read_tokens=10).has_audio is True
        assert UsageDetail().has_audio is False


class TestInspectEntry:
    """Tests for InspectEntry dataclass."""

    def test_basic_creation(self):
        entry = InspectEntry(
            history_index=0,
            kind="request",
            role="user",
            parts=[PartDetail(part_kind="user-prompt", content="Hello")],
            preview="Hello",
        )
        assert entry.history_index == 0
        assert entry.kind == "request"
        assert entry.role == "user"
        assert len(entry.parts) == 1

    def test_role_color(self):
        assert (
            InspectEntry(history_index=0, kind="request", role="user").role_color
            == "fg:ansigreen"
        )
        assert (
            InspectEntry(history_index=0, kind="response", role="assistant").role_color
            == "fg:ansiblue"
        )
        assert (
            InspectEntry(history_index=0, kind="request", role="system").role_color
            == "bold fg:ansicyan"
        )
        assert (
            InspectEntry(history_index=0, kind="request", role="tool-return").role_color
            == "fg:ansiyellow"
        )

    def test_role_short(self):
        assert (
            InspectEntry(history_index=0, kind="request", role="user").role_short
            == "USR"
        )
        assert (
            InspectEntry(history_index=0, kind="response", role="assistant").role_short
            == "AST"
        )
        assert (
            InspectEntry(history_index=0, kind="request", role="system").role_short
            == "SYS"
        )
        assert (
            InspectEntry(history_index=0, kind="request", role="tool-return").role_short
            == "TRT"
        )
        assert (
            InspectEntry(history_index=0, kind="request", role="mixed").role_short
            == "MIX"
        )
        assert (
            InspectEntry(history_index=0, kind="request", role="unknown").role_short
            == "???"
        )


class TestBuildInspectEntriesWithMocks:
    """Tests using mock pydantic-ai message objects."""

    def test_empty_history(self):
        result = build_inspect_entries([])
        assert result == []

    def test_unknown_message_type(self):
        """Test handling of unknown/non-pydantic-ai objects."""
        result = build_inspect_entries(["just a string", 12345, None])
        assert len(result) == 3
        for i, entry in enumerate(result):
            assert entry.history_index == i
            assert entry.kind == "unknown"
            assert entry.role == "unknown"


class TestBuildInspectEntriesWithRealMessages:
    """Tests using real pydantic-ai message objects."""

    @pytest.fixture
    def make_user_request(self):
        """Factory for creating UserPromptPart requests."""
        from pydantic_ai.messages import ModelRequest, UserPromptPart

        def _make(content: str) -> ModelRequest:
            return ModelRequest(parts=[UserPromptPart(content=content)])

        return _make

    @pytest.fixture
    def make_system_request(self):
        """Factory for creating SystemPromptPart requests."""
        from pydantic_ai.messages import ModelRequest, SystemPromptPart

        def _make(content: str) -> ModelRequest:
            return ModelRequest(parts=[SystemPromptPart(content=content)])

        return _make

    @pytest.fixture
    def make_text_response(self):
        """Factory for creating TextPart responses."""
        from pydantic_ai.messages import ModelResponse, TextPart

        def _make(content: str) -> ModelResponse:
            return ModelResponse(parts=[TextPart(content=content)])

        return _make

    def test_user_prompt(self, make_user_request):
        msg = make_user_request("Hello, world!")
        result = build_inspect_entries([msg])

        assert len(result) == 1
        entry = result[0]
        assert entry.history_index == 0
        assert entry.kind == "request"
        assert entry.role == "user"
        assert len(entry.parts) == 1
        assert entry.parts[0].part_kind == "user-prompt"
        assert entry.parts[0].content == "Hello, world!"
        assert entry.preview == "Hello, world!"

    def test_system_prompt(self, make_system_request):
        msg = make_system_request("You are a helpful assistant.")
        result = build_inspect_entries([msg])

        assert len(result) == 1
        entry = result[0]
        assert entry.kind == "request"
        assert entry.role == "system"
        assert entry.has_system_prompt is True
        assert entry.parts[0].part_kind == "system-prompt"

    def test_bundled_system_and_user(self):
        """Test first message with system + user prompt bundled."""
        from pydantic_ai.messages import (
            ModelRequest,
            SystemPromptPart,
            UserPromptPart,
        )

        msg = ModelRequest(
            parts=[
                SystemPromptPart(content="System instructions"),
                UserPromptPart(content="User question"),
            ]
        )
        result = build_inspect_entries([msg])

        assert len(result) == 1
        entry = result[0]
        assert entry.role == "system"  # System takes precedence
        assert entry.has_system_prompt is True
        assert len(entry.parts) == 2
        assert entry.parts[0].part_kind == "system-prompt"
        assert entry.parts[1].part_kind == "user-prompt"

    def test_text_response(self, make_text_response):
        msg = make_text_response("Here is my response.")
        result = build_inspect_entries([msg])

        assert len(result) == 1
        entry = result[0]
        assert entry.kind == "response"
        assert entry.role == "assistant"
        assert entry.parts[0].part_kind == "text"
        assert entry.parts[0].content == "Here is my response."

    def test_tool_call_response(self):
        """Test response with tool calls."""
        from pydantic_ai.messages import ModelResponse, ToolCallPart

        msg = ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name="read_file",
                    args={"file_path": "/test.py"},
                    tool_call_id="tc_001",
                )
            ]
        )
        result = build_inspect_entries([msg])

        assert len(result) == 1
        entry = result[0]
        assert entry.kind == "response"
        assert entry.role == "assistant"
        assert entry.tool_call_count == 1
        assert entry.parts[0].part_kind == "tool-call"
        assert entry.parts[0].tool_name == "read_file"
        assert entry.parts[0].tool_call_id == "tc_001"
        assert entry.parts[0].tool_args == {"file_path": "/test.py"}
        assert "<1 tool call: read_file>" in entry.preview

    def test_tool_return_request(self):
        """Test request with tool return."""
        from pydantic_ai.messages import ModelRequest, ToolReturnPart

        msg = ModelRequest(
            parts=[
                ToolReturnPart(
                    tool_name="read_file",
                    content="file contents here",
                    tool_call_id="tc_001",
                )
            ]
        )
        result = build_inspect_entries([msg])

        assert len(result) == 1
        entry = result[0]
        assert entry.kind == "request"
        assert entry.role == "tool-return"
        assert entry.parts[0].part_kind == "tool-return"
        assert entry.parts[0].tool_name == "read_file"
        assert entry.parts[0].content == "file contents here"

    def test_thinking_part(self):
        """Test response with thinking part."""
        try:
            from pydantic_ai.messages import ModelResponse, ThinkingPart
        except ImportError:
            pytest.skip("ThinkingPart not available in this pydantic-ai version")

        msg = ModelResponse(
            parts=[
                ThinkingPart(content="Let me think about this..."),
            ]
        )
        result = build_inspect_entries([msg])

        assert len(result) == 1
        entry = result[0]
        assert entry.thinking_block_count == 1
        assert entry.parts[0].part_kind == "thinking"
        assert entry.parts[0].content == "Let me think about this..."

    def test_mixed_response(self):
        """Test response with multiple part types."""
        try:
            from pydantic_ai.messages import (
                ModelResponse,
                TextPart,
                ThinkingPart,
                ToolCallPart,
            )
        except ImportError:
            pytest.skip("ThinkingPart not available")

        msg = ModelResponse(
            parts=[
                ThinkingPart(content="Thinking..."),
                TextPart(content="Here's my answer"),
                ToolCallPart(tool_name="read_file", args={}, tool_call_id="tc_1"),
                ToolCallPart(tool_name="grep", args={}, tool_call_id="tc_2"),
            ]
        )
        result = build_inspect_entries([msg])

        assert len(result) == 1
        entry = result[0]
        assert entry.role == "assistant"
        assert entry.tool_call_count == 2
        assert entry.thinking_block_count == 1
        assert len(entry.parts) == 4
        # Preview should prefer text content
        assert entry.preview == "Here's my answer"

    def test_usage_extraction(self):
        """Test that usage details are extracted correctly."""
        from pydantic_ai.messages import ModelResponse, RequestUsage, TextPart

        msg = ModelResponse(
            parts=[TextPart(content="Response")],
            usage=RequestUsage(
                input_tokens=100,
                output_tokens=50,
                cache_write_tokens=10,
                cache_read_tokens=5,
            ),
        )
        result = build_inspect_entries([msg])

        assert len(result) == 1
        entry = result[0]
        assert entry.usage is not None
        assert entry.usage.input_tokens == 100
        assert entry.usage.output_tokens == 50
        assert entry.usage.total_tokens == 150
        assert entry.usage.cache_write_tokens == 10
        assert entry.usage.cache_read_tokens == 5
        assert entry.usage.has_cache is True

    def test_response_metadata(self):
        """Test that response metadata is extracted."""
        from pydantic_ai.messages import ModelResponse, TextPart

        msg = ModelResponse(
            parts=[TextPart(content="Hi")],
            model_name="gpt-4o",
            provider_name="openai",
            finish_reason="stop",
        )
        result = build_inspect_entries([msg])

        assert len(result) == 1
        entry = result[0]
        assert entry.model_name == "gpt-4o"
        assert entry.provider_name == "openai"
        assert entry.finish_reason == "stop"

    def test_full_conversation(self, make_user_request, make_text_response):
        """Test a multi-turn conversation."""
        history = [
            make_user_request("What is 2+2?"),
            make_text_response("2+2 equals 4."),
            make_user_request("And 3+3?"),
            make_text_response("3+3 equals 6."),
        ]
        result = build_inspect_entries(history)

        assert len(result) == 4
        assert [e.history_index for e in result] == [0, 1, 2, 3]
        assert [e.role for e in result] == ["user", "assistant", "user", "assistant"]
        assert [e.kind for e in result] == [
            "request",
            "response",
            "request",
            "response",
        ]

    def test_retry_prompt_part(self):
        """Test request with retry prompt (validation error)."""
        try:
            from pydantic_ai.messages import ModelRequest, RetryPromptPart
        except ImportError:
            pytest.skip("RetryPromptPart not available")

        msg = ModelRequest(
            parts=[
                RetryPromptPart(
                    content="Validation error: missing required field",
                    tool_name="create_file",
                    tool_call_id="tc_001",
                )
            ]
        )
        result = build_inspect_entries([msg])

        assert len(result) == 1
        entry = result[0]
        assert entry.role == "tool-return"  # Retry is treated as tool response
        assert entry.parts[0].part_kind == "retry-prompt"
        assert entry.parts[0].tool_name == "create_file"

    def test_preview_truncation(self, make_user_request):
        """Test that previews are truncated but content is not."""
        long_content = "x" * 200
        msg = make_user_request(long_content)
        result = build_inspect_entries([msg])

        assert len(result) == 1
        entry = result[0]
        # Content should be full
        assert entry.parts[0].content == long_content
        assert len(entry.parts[0].content) == 200
        # Preview should be truncated
        assert len(entry.preview) <= 80
        assert entry.preview.endswith("...")


class TestSystemPromptInstructions:
    """Regression tests for inspect-049.

    code-puppy stores the system prompt in ``ModelRequest.instructions`` (not
    as a ``SystemPromptPart``), and pydantic-ai stamps ``instructions`` onto
    every replayed request. Without normalization the system prompt leaks into
    tool-return (TRT) and follow-up user messages.
    """

    SYS = "You are a helpful assistant. Follow the rules."

    def _build(self):
        from pydantic_ai.messages import (
            ModelRequest,
            ModelResponse,
            ToolCallPart,
            ToolReturnPart,
            UserPromptPart,
        )

        history = [
            ModelRequest(parts=[UserPromptPart(content="hi")], instructions=self.SYS),
            ModelResponse(
                parts=[ToolCallPart(tool_name="read_file", args={}, tool_call_id="tc1")]
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name="read_file",
                        content="FILE CONTENTS",
                        tool_call_id="tc1",
                    )
                ],
                instructions=self.SYS,
            ),
            ModelRequest(
                parts=[UserPromptPart(content="thanks")], instructions=self.SYS
            ),
        ]
        return build_inspect_entries(history)

    def test_system_prompt_surfaced_once_as_system_message(self):
        entries = self._build()
        system_entries = [e for e in entries if e.role == "system"]
        assert len(system_entries) == 1
        sys_entry = system_entries[0]
        assert sys_entry.history_index == 0
        assert sys_entry.has_system_prompt is True
        assert sys_entry.parts[0].part_kind == "system-prompt"
        assert sys_entry.parts[0].content == self.SYS
        # The user prompt that shared the first request is still present.
        assert any(p.part_kind == "user-prompt" for p in sys_entry.parts)

    def test_tool_return_has_no_system_prompt(self):
        entries = self._build()
        trt = entries[2]
        assert trt.role == "tool-return"
        # Instructions must be stripped so the detail pane never shows the
        # system prompt on a tool-return message.
        assert trt.instructions is None
        assert all(p.part_kind != "system-prompt" for p in trt.parts)
        assert trt.parts[0].part_kind == "tool-return"
        assert trt.parts[0].content == "FILE CONTENTS"

    def test_followup_user_has_no_system_prompt(self):
        entries = self._build()
        followup = entries[3]
        assert followup.role == "user"
        assert followup.instructions is None
        assert all(p.part_kind != "system-prompt" for p in followup.parts)

    def test_detail_text_excludes_system_prompt_on_trt(self):
        from inspect_history.inspect_render import detail_to_plain_text

        entries = self._build()
        trt_text = detail_to_plain_text(entries[2])
        assert self.SYS not in trt_text
        assert "instructions:" not in trt_text
        assert "FILE CONTENTS" in trt_text

    def test_changed_instructions_surface_new_system_message(self):
        """A mid-conversation system prompt change is surfaced again."""
        from pydantic_ai.messages import ModelRequest, UserPromptPart

        history = [
            ModelRequest(
                parts=[UserPromptPart(content="a")], instructions="PROMPT ONE"
            ),
            ModelRequest(
                parts=[UserPromptPart(content="b")], instructions="PROMPT ONE"
            ),
            ModelRequest(
                parts=[UserPromptPart(content="c")], instructions="PROMPT TWO"
            ),
        ]
        entries = build_inspect_entries(history)
        system_entries = [e for e in entries if e.role == "system"]
        assert len(system_entries) == 2
        assert system_entries[0].parts[0].content == "PROMPT ONE"
        assert system_entries[1].parts[0].content == "PROMPT TWO"
        # The middle request reused PROMPT ONE — stripped, stays a user message.
        assert entries[1].role == "user"
        assert entries[1].instructions is None

    def test_request_without_instructions_unchanged(self):
        from pydantic_ai.messages import ModelRequest, UserPromptPart

        history = [ModelRequest(parts=[UserPromptPart(content="hi")])]
        entries = build_inspect_entries(history)
        assert entries[0].role == "user"
        assert entries[0].has_system_prompt is False
        assert all(p.part_kind != "system-prompt" for p in entries[0].parts)


class TestPreviewGeneration:
    """Tests for preview generation edge cases."""

    def test_empty_parts(self):
        """Entry with no parts should have fallback preview."""
        entry = InspectEntry(
            history_index=0,
            kind="request",
            role="unknown",
            parts=[],
            preview="<empty>",
        )
        assert entry.preview == "<empty>"

    def test_multiline_content_flattened(self):
        """Multi-line content should be flattened in preview."""
        from pydantic_ai.messages import ModelRequest, UserPromptPart

        msg = ModelRequest(parts=[UserPromptPart(content="Line 1\nLine 2\nLine 3")])
        result = build_inspect_entries([msg])

        # Preview should be single line
        assert "\n" not in result[0].preview
        assert "Line 1" in result[0].preview
