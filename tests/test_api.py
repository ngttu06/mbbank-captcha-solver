import base64
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from mbbank.CONFIG import LettersInt
from mbbank.MAIN import CrackLettesInt4
from mbbank.server import app, decode_base64_image


def image_bytes(color='white', size=(160, 50)):
    buffer = BytesIO()
    Image.new('RGB', size, color).save(buffer, format='PNG')
    return buffer.getvalue()


class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        app.config['TESTING'] = True
        cls.image = image_bytes()
        cls.encoded = base64.b64encode(cls.image).decode('ascii')

    def setUp(self):
        self.client = app.test_client()

    def assert_prediction(self, response):
        self.assertEqual(response.status_code, 200)
        prediction = response.get_json()['prediction']
        self.assertEqual(len(prediction), LettersInt.PIC_NAME_LEN)
        self.assertTrue(set(prediction).issubset(LettersInt.content_range))
        return prediction

    def test_health(self):
        response = self.client.get('/health')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json(), {'status': 'ok'})

    def test_index(self):
        self.assertEqual(self.client.get('/').get_json()['predict'], '/predict')

    def test_prediction_matches_original_model(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'sample.png'
            path.write_bytes(self.image)
            expected = CrackLettesInt4().predict_one(str(path))
        response = self.client.post('/predict', json={'image_base64': self.encoded})
        self.assertEqual(self.assert_prediction(response), expected)

    def test_upload_and_base64_match(self):
        uploaded = self.client.post(
            '/predict', data={'image': (BytesIO(self.image), 'captcha.png')}
        )
        encoded = self.client.post('/predict', json={'image_base64': self.encoded})
        self.assertEqual(self.assert_prediction(uploaded), self.assert_prediction(encoded))

    def test_data_uri(self):
        response = self.client.post(
            '/predict', json={'image_base64': 'data:image/png;base64,' + self.encoded}
        )
        self.assert_prediction(response)

    def test_invalid_inputs(self):
        cases = [
            {}, [], None, {'image_base64': None}, {'image_base64': ''},
            {'image_base64': 123}, {'image_base64': 'not-base64!'},
            {'image_base64': 'data:text/plain;base64,' + self.encoded},
            {'image_base64': 'data:image/png,' + self.encoded},
            {'image_base64': base64.b64encode(b'not an image').decode('ascii')},
        ]
        for payload in cases:
            with self.subTest(payload=payload):
                response = self.client.post('/predict', json=payload)
                self.assertEqual(response.status_code, 400)
                self.assertIn('error', response.get_json())

    def test_invalid_content_type(self):
        response = self.client.post('/predict', data='image=hello')
        self.assertEqual(response.status_code, 400)

    def test_malformed_json(self):
        response = self.client.post('/predict', data='{', content_type='application/json')
        self.assertEqual(response.status_code, 400)

    def test_empty_upload(self):
        response = self.client.post('/predict', data={'image': (BytesIO(), 'empty.png')})
        self.assertEqual(response.status_code, 400)

    def test_pixel_limit(self):
        response = self.client.post(
            '/predict', data={'image': (BytesIO(image_bytes(size=(1001, 1000))), 'large.png')}
        )
        self.assertEqual(response.status_code, 400)

    def test_request_limit(self):
        response = self.client.post(
            '/predict', json={'image_base64': 'A' * (2 * 1024 * 1024)}
        )
        self.assertEqual(response.status_code, 413)
        self.assertIn('error', response.get_json())

    def test_json_http_errors(self):
        self.assertEqual(self.client.get('/predict').status_code, 405)
        response = self.client.get('/missing')
        self.assertEqual(response.status_code, 404)
        self.assertIn('error', response.get_json())

    def test_concurrent_distinct_images(self):
        samples = [image_bytes(color) for color in ['white', 'black', 'red', 'green']]

        def submit(data):
            with app.test_client() as client:
                return self.assert_prediction(client.post(
                    '/predict', data={'image': (BytesIO(data), 'captcha.png')}
                ))

        expected = [submit(sample) for sample in samples]
        with ThreadPoolExecutor(max_workers=4) as pool:
            actual = list(pool.map(submit, samples * 3))
        self.assertEqual(actual, expected * 3)

    def test_base64_decoder(self):
        self.assertEqual(decode_base64_image(self.encoded), self.image)


if __name__ == '__main__':
    unittest.main()