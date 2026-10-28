"""UI smoke checks, using Streamlit's local testing API—not external hosting."""
import importlib.util
import os
from pathlib import Path
import unittest
from unittest.mock import patch

APP = Path(__file__).resolve().parents[1]/"streamlit_app.py"


@unittest.skipUnless(importlib.util.find_spec("streamlit"),"Run in the isolated app environment after installing requirements-app.txt")
class StreamlitSmokeTests(unittest.TestCase):
    def test_viewer_starts_and_upload_is_opt_in(self):
        from streamlit.testing.v1 import AppTest
        with patch.dict(os.environ,{"STROKEPOINT_ENABLE_LOCAL_UPLOAD":"0"}):
            app=AppTest.from_file(str(APP)).run(timeout=40)
            self.assertEqual(len(app.exception),0)
            self.assertEqual(app.title[0].value,"StrokePoint++")
            self.assertEqual(len(app.tabs),5)
            self.assertEqual(len(app.get("file_uploader")),0)
            # If real reviewed aggregates are packaged, exercise the second
            # cohort without reading any original project artifact or labels.
            if app.sidebar.radio:
                radio=app.sidebar.radio[0]
                if len(radio.options)>1:
                    radio.set_value("IXI_HH_FINAL").run(timeout=40)
                    self.assertEqual(len(app.exception),0)

    def test_local_upload_requires_second_user_consent(self):
        from streamlit.testing.v1 import AppTest
        with patch.dict(os.environ,{"STROKEPOINT_ENABLE_LOCAL_UPLOAD":"1","TMPDIR":"/dev/shm"}):
            app=AppTest.from_file(str(APP)).run(timeout=40)
            self.assertEqual(len(app.exception),0)
            self.assertEqual(len(app.get("file_uploader")),0)
            self.assertFalse(app.checkbox[0].value)
            app.checkbox[0].check().run(timeout=40)
            self.assertEqual(len(app.exception),0)
            self.assertEqual(len(app.get("file_uploader")),1)

    def test_local_upload_requires_ram_temporary_directory(self):
        from streamlit.testing.v1 import AppTest
        with patch.dict(os.environ,{"STROKEPOINT_ENABLE_LOCAL_UPLOAD":"1","TMPDIR":"/tmp"}):
            app=AppTest.from_file(str(APP)).run(timeout=40)
            self.assertEqual(len(app.exception),0)
            self.assertEqual(len(app.get("file_uploader")),0)
            self.assertTrue(any("RAM-backed" in warning.value for warning in app.warning))


if __name__ == "__main__":unittest.main()
