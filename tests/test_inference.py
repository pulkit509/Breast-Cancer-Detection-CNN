import importlib.util
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np
from inference import CLASS_NAMES, MAX_UPLOAD_BYTES, preprocess_image, validate_model, validate_probabilities


class OutputTests(unittest.TestCase):
    def test_class_order_and_probabilities(self):
        probabilities = validate_probabilities([[0.1, 0.7, 0.2]])
        self.assertEqual(CLASS_NAMES[int(np.argmax(probabilities))], "benign")
        np.testing.assert_array_equal(probabilities, [0.1, 0.7, 0.2])

    def test_invalid_predictions(self):
        for output in ([0.1, 0.2, 0.7], [[0.1, 0.9]], [[float("nan"), 0, 1]],
                       [[-0.1, 0.2, 0.9]], [[0.2, 0.2, 0.2]], [[0, 0, float("inf")]]):
            with self.subTest(output=output), self.assertRaises(ValueError):
                validate_probabilities(output)

    def test_model_contract(self):
        def softmax():
            pass
        model = SimpleNamespace(input_shape=(None, 128, 128, 3), output_shape=(None, 3),
                                layers=[SimpleNamespace(activation=softmax)])
        validate_model(model)
        for field, value in [("input_shape", (None, 224, 224, 3)),
                             ("output_shape", (None, 2)), ("layers", [SimpleNamespace()])]:
            old = getattr(model, field)
            setattr(model, field, value)
            with self.assertRaises(ValueError):
                validate_model(model)
            setattr(model, field, old)


@unittest.skipUnless(importlib.util.find_spec("cv2"), "OpenCV is not installed")
class PreprocessingTests(unittest.TestCase):
    def test_matches_notebook(self):
        import cv2
        rng = np.random.default_rng(42)
        # Color, grayscale and alpha images, including a non-square resize.
        for shape in [(31, 47, 3), (31, 47), (31, 47, 4)]:
            image = rng.integers(0, 256, size=shape, dtype=np.uint8)
            ok, encoded = cv2.imencode(".png", image)
            self.assertTrue(ok)
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "scan.png"
                path.write_bytes(encoded.tobytes())
                expected = cv2.resize(cv2.imread(str(path)), (128, 128))
            _, actual = preprocess_image(encoded.tobytes())
            self.assertEqual(actual.shape, (1, 128, 128, 3))
            self.assertEqual(actual.dtype, np.uint8)
            np.testing.assert_array_equal(actual[0], expected)

    def test_invalid_uploads(self):
        for content in (b"", b"not an image", b"x" * (MAX_UPLOAD_BYTES + 1)):
            with self.assertRaises(ValueError):
                preprocess_image(content)


@unittest.skipUnless(importlib.util.find_spec("streamlit"), "Streamlit is not installed")
class DashboardTests(unittest.TestCase):
    def test_missing_model_renders_and_disables_prediction(self):
        import os
        from unittest.mock import patch
        from streamlit.testing.v1 import AppTest
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(os.environ, {"MODEL_PATH": str(Path(directory) / "missing.keras")}):
                app = AppTest.from_file("streamlit_app.py").run()
                self.assertFalse(app.exception)
                self.assertTrue(app.button[0].disabled)
                self.assertIn("Model not configured", app.warning[0].value)


if __name__ == "__main__":
    unittest.main()
