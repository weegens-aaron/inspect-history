"""Tests for register_callbacks.py command handler."""

from unittest.mock import MagicMock, patch


from inspect_history import register_callbacks


class TestHandleInspectCommand:
    """Tests for handle_inspect_command()."""

    def test_no_agent_emits_error(self):
        """If no agent is available, emit an error and return True."""
        with patch.object(register_callbacks, "get_current_agent", return_value=None):
            with patch.object(register_callbacks, "emit_error") as mock_error:
                result = register_callbacks.handle_inspect_command("/inspect")
                assert result is True
                mock_error.assert_called_once()
                assert "No active agent" in mock_error.call_args[0][0]

    def test_empty_history_emits_error(self):
        """If history is empty, emit an error and return True."""
        mock_agent = MagicMock()
        mock_agent.get_message_history.return_value = []

        with patch.object(
            register_callbacks, "get_current_agent", return_value=mock_agent
        ):
            with patch.object(register_callbacks, "emit_error") as mock_error:
                result = register_callbacks.handle_inspect_command("/inspect")
                assert result is True
                mock_error.assert_called_once()
                assert "empty" in mock_error.call_args[0][0].lower()

    def test_no_valid_entries_emits_error(self):
        """If build_inspect_entries returns empty, emit an error."""
        mock_agent = MagicMock()
        mock_agent.get_message_history.return_value = ["something"]

        with patch.object(
            register_callbacks, "get_current_agent", return_value=mock_agent
        ):
            with patch.object(
                register_callbacks, "build_inspect_entries", return_value=[]
            ):
                with patch.object(register_callbacks, "emit_error") as mock_error:
                    result = register_callbacks.handle_inspect_command("/inspect")
                    assert result is True
                    mock_error.assert_called_once()
                    assert "No valid messages" in mock_error.call_args[0][0]

    def test_launches_tui_with_entries(self):
        """If history has valid entries, launch the InspectMenu."""
        mock_agent = MagicMock()
        mock_agent.get_message_history.return_value = ["msg1", "msg2"]

        mock_entries = [MagicMock(), MagicMock()]
        mock_menu_instance = MagicMock()

        with patch.object(
            register_callbacks, "get_current_agent", return_value=mock_agent
        ):
            with patch.object(
                register_callbacks, "build_inspect_entries", return_value=mock_entries
            ):
                with patch.object(
                    register_callbacks, "InspectMenu", return_value=mock_menu_instance
                ) as mock_menu_cls:
                    result = register_callbacks.handle_inspect_command("/inspect")
                    assert result is True
                    mock_menu_cls.assert_called_once_with(mock_entries)
                    mock_menu_instance.run.assert_called_once()

    def test_command_argument_unused(self):
        """The command argument is unused — any string works."""
        mock_agent = MagicMock()
        mock_agent.get_message_history.return_value = []

        with patch.object(
            register_callbacks, "get_current_agent", return_value=mock_agent
        ):
            with patch.object(register_callbacks, "emit_error"):
                # These should all behave the same
                result1 = register_callbacks.handle_inspect_command("/inspect")
                result2 = register_callbacks.handle_inspect_command("/i")
                result3 = register_callbacks.handle_inspect_command("")
                assert result1 is True
                assert result2 is True
                assert result3 is True


class TestCommandRegistration:
    """Tests for @register_command decorator settings."""

    def test_handler_exists(self):
        """The handler function should exist."""
        assert callable(register_callbacks.handle_inspect_command)

    def test_module_imports_code_puppy_apis(self):
        """Verify the module imports from public APIs."""
        # These should not raise ImportError
        from code_puppy.agents.agent_manager import get_current_agent
        from code_puppy.command_line.command_registry import register_command
        from code_puppy.messaging import emit_error

        assert callable(get_current_agent)
        assert callable(register_command)
        assert callable(emit_error)
