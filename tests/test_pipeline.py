# test_pipeline.py
import json
import logging
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import ANY, MagicMock, call, mock_open, patch

# Add the project root to sys.path to allow imports from pipeline, config, etc.
# This assumes test_pipeline.py is in a 'tests' subdirectory or similar.
# Adjust if your structure is different.
current_dir = Path(__file__).resolve().parent
project_root = (
    current_dir.parent
)  # Assuming tests are in a 'tests' folder one level down
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Now import the module to be tested and its dependencies
try:
    from llama_index.core import (  # Assuming Document is used, ensure BaseNode is also imported if needed
        Document,
    )
    from llama_index.core.base.response.schema import Response as LlamaResponse
    from llama_index.core.llms import (  # Import MessageRole if used directly in pipeline
        MessageRole,
    )
    from llama_index.core.schema import BaseNode  # Added for spec in mocks
    from llama_index.core.schema import NodeWithScore

    import config
    import constants
    import pipeline  # This is your main script, formerly pipeline_logic.py
    from constants import LogMessages as LM
except ImportError as e:
    print(f"Error importing modules for testing: {e}")
    print(
        "Ensure pipeline.py, config.py, and constants.py are in the project root or accessible via PYTHONPATH."
    )
    print(f"Current sys.path: {sys.path}")
    raise

# Remove global logging.disable(logging.CRITICAL)
# self.assertLogs will handle logger levels temporarily.
# If other library logs are too noisy, they can be selectively silenced:
# logging.getLogger("another_library").setLevel(logging.WARNING)


class TestPipelineUtils(unittest.TestCase):

    def test_save_to_jsonl_success(self):
        data = [{"id": 1, "text": "hello"}, {"id": 2, "text": "world"}]
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "test.jsonl"
            result = pipeline.save_to_jsonl(data, file_path)
            self.assertTrue(result)
            self.assertTrue(file_path.exists())
            with open(file_path, "r") as f:
                lines = f.readlines()
                self.assertEqual(len(lines), 2)
                self.assertEqual(json.loads(lines[0]), data[0])
                self.assertEqual(json.loads(lines[1]), data[1])

    def test_save_to_jsonl_empty_list(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "empty.jsonl"
            result = pipeline.save_to_jsonl([], file_path)
            self.assertTrue(result)
            self.assertTrue(file_path.exists())
            with open(file_path, "r") as f:
                self.assertEqual(f.read(), "")

    @patch("builtins.open", side_effect=IOError("Test error"))
    def test_save_to_jsonl_failure(self, mock_file_open):
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "fail.jsonl"
            # Ensure parent directory exists for the initial mkdir call
            file_path.parent.mkdir(parents=True, exist_ok=True)
            result = pipeline.save_to_jsonl([{"id": 1}], file_path)
            self.assertFalse(result)

    def test_load_from_jsonl_success(self):
        data = [{"id": 1, "text": "hello"}, {"id": 2, "text": "world"}]
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "test.jsonl"
            with open(file_path, "w") as f:
                for item in data:
                    f.write(json.dumps(item) + "\n")

            loaded_data = pipeline.load_from_jsonl(file_path)
            self.assertEqual(loaded_data, data)

    def test_load_from_jsonl_non_existent_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "non_existent.jsonl"
            loaded_data = pipeline.load_from_jsonl(file_path)
            self.assertEqual(loaded_data, [])

    def test_load_from_jsonl_malformed_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "malformed.jsonl"
            with open(file_path, "w") as f:
                f.write('{"id": 1, "text": "hello"}\n')
                f.write("this is not json\n")
                f.write('{"id": 2, "text": "world"}\n')

            # Ensure the logger 'pipeline' (from pipeline.py) is used
            with self.assertLogs(logging.getLogger("pipeline"), level="WARNING") as cm:
                loaded_data = pipeline.load_from_jsonl(file_path)
            self.assertEqual(len(loaded_data), 2)
            self.assertEqual(loaded_data[0], {"id": 1, "text": "hello"})
            self.assertEqual(loaded_data[1], {"id": 2, "text": "world"})
            self.assertTrue(
                any(
                    "Invalid JSON on line 2" in record.getMessage()
                    for record in cm.records
                )
            )

    @patch("pipeline.time.sleep")
    @patch("pipeline.random.uniform", return_value=1.5)
    def test_human_sleep(self, mock_uniform, mock_sleep):
        # Ensure the logger 'pipeline' is used
        with self.assertLogs(logging.getLogger("pipeline"), level="INFO") as cm:
            pipeline.human_sleep(min_sec=1, max_sec=2)
        mock_uniform.assert_called_once_with(1, 2)
        mock_sleep.assert_called_once_with(1.5)
        self.assertTrue(
            any(
                LM.HUMAN_SLEEP_DELAY.format(delay=1.5) in record.getMessage()
                for record in cm.records
            )
        )


class TestModelSetup(unittest.TestCase):

    @patch("pipeline.HuggingFaceEmbedding")
    @patch("pipeline.GoogleGenAI")
    def test_setup_llm_and_embed_models_success(
        self, MockGoogleGenAI, MockHuggingFaceEmbedding
    ):
        mock_embed = MockHuggingFaceEmbedding.return_value
        mock_llm = MockGoogleGenAI.return_value

        # Patch config.GOOGLE_API_KEY directly for this test's scope
        # OR explicitly pass google_api_key to the function call
        with patch.object(
            config, "GOOGLE_API_KEY", "fake_key_for_test"
        ):  # Ensure this patch is effective
            embed_model, llm = pipeline.setup_llm_and_embed_models(
                embed_model_name="test_embed_model",
                llm_model_name="test_llm_model",
                google_api_key="fake_key_for_test",  # Explicitly pass to override default argument issue
            )

        # Check if HuggingFaceEmbedding was called correctly (based on nomic logic in pipeline.py)
        if "nomic" in "test_embed_model":
            MockHuggingFaceEmbedding.assert_called_once_with(
                model_name="test_embed_model",
                query_instruction=config.NOMIC_QUERY_INSTRUCTION,
                text_instruction=config.NOMIC_DOCUMENT_INSTRUCTION,
                model_kwargs={"trust_remote_code": True},
            )
        else:
            MockHuggingFaceEmbedding.assert_called_once_with(
                model_name="test_embed_model"
            )

        MockGoogleGenAI.assert_called_once_with(
            model_name="test_llm_model", api_key="fake_key_for_test"
        )
        self.assertEqual(embed_model, mock_embed)
        self.assertEqual(llm, mock_llm)

    @patch("pipeline.HuggingFaceEmbedding")
    @patch("pipeline.GoogleGenAI")
    def test_setup_llm_and_embed_models_no_api_key(
        self, MockGoogleGenAI, MockHuggingFaceEmbedding
    ):
        mock_embed = MockHuggingFaceEmbedding.return_value

        # Ensure the logger 'pipeline' is used
        with self.assertLogs(logging.getLogger("pipeline"), level="ERROR") as cm:
            # Explicitly pass None for google_api_key to trigger the error path
            embed_model, llm = pipeline.setup_llm_and_embed_models(google_api_key=None)

        self.assertEqual(embed_model, mock_embed)
        self.assertIsNone(llm)
        MockGoogleGenAI.assert_not_called()
        self.assertTrue(
            any(
                LM.GOOGLE_API_KEY_NOT_FOUND in record.getMessage()
                for record in cm.records
            )
        )

    @patch("pipeline.HuggingFaceEmbedding", side_effect=Exception("Embedding error"))
    def test_setup_llm_and_embed_models_embed_failure(self, MockHuggingFaceEmbedding):
        # Ensure the logger 'pipeline' is used
        with self.assertLogs(logging.getLogger("pipeline"), level="ERROR") as cm:
            embed_model, llm = pipeline.setup_llm_and_embed_models(
                google_api_key="fake_key"
            )  # Provide a key so it attempts LLM

        self.assertIsNone(embed_model)  # Because HuggingFaceEmbedding raised an error
        # Depending on exact logic, llm might be None or a mock if embed fails first.
        # In the current pipeline.py, if embed_model init fails, it returns (None, None) before llm init.
        self.assertIsNone(llm)
        self.assertTrue(
            any(
                LM.FAILED_CONFIGURE_LLM.format(error=Exception("Embedding error"))
                in record.getMessage()
                for record in cm.records
            )
        )


class TestLLMSummary(unittest.TestCase):

    def setUp(self):
        self.mock_llm = MagicMock(spec=pipeline.GoogleGenAI)
        self.mock_chat_response = MagicMock()
        self.mock_chat_response.message = MagicMock()
        self.mock_chat_response.message.content = "This is a test summary."
        self.mock_llm.chat.return_value = self.mock_chat_response

    @patch("pipeline.human_sleep")
    def test_call_llm_for_summary_success(self, mock_human_sleep):
        code = "def hello():\n  print('world')"
        summary = pipeline.call_llm_for_summary(code, self.mock_llm, "test_file.py")

        self.assertEqual(summary, "This is a test summary.")
        self.mock_llm.chat.assert_called_once()
        args, _ = self.mock_llm.chat.call_args
        messages = args[0]
        self.assertEqual(
            messages[0].role, pipeline.MessageRole.SYSTEM
        )  # Assuming MessageRole is accessible via pipeline
        self.assertEqual(messages[1].role, pipeline.MessageRole.USER)
        self.assertIn(code, messages[1].content)
        mock_human_sleep.assert_called_once()

    @patch("pipeline.human_sleep")
    def test_call_llm_for_summary_empty_code(self, mock_human_sleep):
        summary = pipeline.call_llm_for_summary("  \n  ", self.mock_llm)
        self.assertEqual(summary, LM.NO_CODE_FOR_SUMMARY)
        self.mock_llm.chat.assert_not_called()
        mock_human_sleep.assert_not_called()

    @patch("pipeline.human_sleep")
    def test_call_llm_for_summary_llm_error(self, mock_human_sleep):
        self.mock_llm.chat.side_effect = Exception("LLM API Error")
        code = "def error_func(): pass"

        # Ensure the logger 'pipeline' is used
        with self.assertLogs(logging.getLogger("pipeline"), level="ERROR") as cm:
            summary = pipeline.call_llm_for_summary(code, self.mock_llm)

        self.assertEqual(
            summary, LM.SUMMARY_UNAVAILABLE_ERROR.format(error="LLM API Error")
        )
        # Check if the specific log message is present
        self.assertTrue(
            any(
                LM.FAILED_GENERATE_SUMMARY.format(
                    file_path="Unknown", error=Exception("LLM API Error")
                )
                in record.getMessage()
                for record in cm.records
            )
        )
        mock_human_sleep.assert_not_called()


class TestCodeExtraction(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.repo_path = Path(self.temp_dir.name)

        (self.repo_path / "file1.py").write_text("def func1():\n  pass")
        (self.repo_path / "file2.py").write_text(
            "class MyClass:\n  def method(self):\n    pass"
        )
        (self.repo_path / "file.txt").write_text("some text")
        (self.repo_path / "empty.py").write_text("")
        (self.repo_path / "subdir").mkdir()
        (self.repo_path / "subdir" / "file3.java").write_text("public class Main { }")

    def tearDown(self):
        self.temp_dir.cleanup()

    @patch("pipeline.CodeSplitter")
    @patch("pipeline.config")  # Patch the whole config module
    @patch("pipeline.constants")  # Patch the whole constants module
    def test_extract_code_chunks_basic(
        self, mock_constants_module, mock_config_module, MockCodeSplitter
    ):
        # Configure the mocks for config and constants
        mock_config_module.SUPPORTED_FILE_SUFFIXES = (".py", ".java")
        mock_config_module.FILE_EXTENSION_TO_LANGUAGE_MAP = {
            ".py": "python",
            ".java": "java",
        }
        mock_config_module.CODE_SPLITTER_CHUNK_LINES = 100
        mock_config_module.CODE_SPLITTER_CHUNK_LINES_OVERLAP = 10
        mock_config_module.CODE_SPLITTER_MAX_CHARS = 1000

        mock_constants_module.UTF8 = "utf-8"
        mock_constants_module.HYBRID_CHUNK_DATA_KEYS = {
            "FILE_PATH_METADATA_KEY": "file_path"
        }
        mock_constants_module.CHUNK_DATA_KEYS = {
            "FILE": "file",
            "CODE": "code",
            "METADATA": "metadata",
        }

        mock_splitter_instance = MockCodeSplitter.return_value

        def mock_get_nodes(docs):
            # Using str(docs[0].metadata['file_path']) directly as it's simpler
            doc_file_path = str(
                docs[0].metadata[
                    mock_constants_module.HYBRID_CHUNK_DATA_KEYS[
                        "FILE_PATH_METADATA_KEY"
                    ]
                ]
            )

            if "file1.py" in doc_file_path:
                node1 = MagicMock(spec=BaseNode)  # Use imported BaseNode
                node1.text = "def func1():\n  pass"
                node1.metadata = {
                    mock_constants_module.HYBRID_CHUNK_DATA_KEYS[
                        "FILE_PATH_METADATA_KEY"
                    ]: str(self.repo_path / "file1.py")
                }
                return [node1]
            elif "file2.py" in doc_file_path:
                node2 = MagicMock(spec=BaseNode)
                node2.text = "class MyClass:\n  def method(self):\n    pass"
                node2.metadata = {
                    mock_constants_module.HYBRID_CHUNK_DATA_KEYS[
                        "FILE_PATH_METADATA_KEY"
                    ]: str(self.repo_path / "file2.py")
                }
                return [node2]
            elif "file3.java" in doc_file_path:
                node3 = MagicMock(spec=BaseNode)
                node3.text = "public class Main { }"
                node3.metadata = {
                    mock_constants_module.HYBRID_CHUNK_DATA_KEYS[
                        "FILE_PATH_METADATA_KEY"
                    ]: str(self.repo_path / "subdir" / "file3.java")
                }
                return [node3]
            return []

        mock_splitter_instance.get_nodes_from_documents.side_effect = mock_get_nodes

        chunks = pipeline.extract_code_chunks(self.repo_path)

        self.assertEqual(len(chunks), 3)
        self.assertEqual(MockCodeSplitter.call_count, 3)

        actual_languages_used = sorted(
            [c.kwargs["language"] for c in MockCodeSplitter.call_args_list]
        )
        expected_languages = sorted(
            ["python", "python", "java"]
        )  # Order might vary due to rglob
        self.assertEqual(actual_languages_used, expected_languages)

        self.assertTrue(
            any(
                c[mock_constants_module.CHUNK_DATA_KEYS["CODE"]]
                == "def func1():\n  pass"
                for c in chunks
            )
        )
        self.assertTrue(
            any(
                c[mock_constants_module.CHUNK_DATA_KEYS["CODE"]]
                == "public class Main { }"
                for c in chunks
            )
        )

    @patch("pipeline.load_from_jsonl")
    def test_extract_code_chunks_uses_cache(self, mock_load_from_jsonl):
        cached_data = [{"file": "cached.py", "code": "cached_code", "metadata": {}}]
        mock_load_from_jsonl.return_value = cached_data

        cache_file_path = self.repo_path / "cache.jsonl"

        with patch.object(Path, "exists", return_value=True):
            chunks = pipeline.extract_code_chunks(
                self.repo_path, output_file=cache_file_path, force_refresh=False
            )

        self.assertEqual(chunks, cached_data)
        mock_load_from_jsonl.assert_called_once_with(cache_file_path)


if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)
