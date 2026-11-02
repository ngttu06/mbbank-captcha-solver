import base64
import binascii
from io import BytesIO
from threading import Lock

import torch
from flask import Flask, request, jsonify
from PIL import Image, UnidentifiedImageError
from torchvision.transforms import ToTensor
from werkzeug.exceptions import HTTPException

try:
    from .CONFIG import LettersInt
    from .MAIN import CNN
except ImportError:
    from CONFIG import LettersInt
    from MAIN import CNN


class CaptchaPredictor:
    def __init__(self):
        self.model = CNN()
        self.model.load_state_dict(
            torch.load(LettersInt.model_path, map_location='cpu', weights_only=True)
        )
        self.model.eval()
        self.transform = ToTensor()
        self.lock = Lock()

    def predict(self, image_data):
        with Image.open(BytesIO(image_data)) as image:
            if image.width * image.height > 1_000_000:
                raise ValueError('Image must contain at most 1,000,000 pixels.')
            image = image.convert('RGB')
            image = image.resize((LettersInt.IMG_WIDTH, LettersInt.IMG_HEIGHT))
            tensor = self.transform(image).unsqueeze(0)

        # Bound simultaneous inference memory use while HTTP requests remain threaded.
        with self.lock, torch.inference_mode():
            output = self.model(tensor)
            indices = output.reshape(
                LettersInt.PIC_NAME_LEN, LettersInt.range_len
            ).argmax(dim=1).tolist()
        return ''.join(LettersInt.content_range[index] for index in indices)


def decode_base64_image(value):
    if not isinstance(value, str) or not value:
        raise ValueError('image_base64 must be a non-empty string.')
    if value.startswith('data:'):
        header, separator, value = value.partition(',')
        if not separator or not header.startswith('data:image/') or not header.endswith(';base64'):
            raise ValueError('Invalid image data URI.')
    try:
        image_data = base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError) as error:
        raise ValueError('Invalid Base64 image.') from error
    if not image_data:
        raise ValueError('Image is empty.')
    return image_data


app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024
predictor = CaptchaPredictor()


@app.errorhandler(HTTPException)
def handle_http_error(error):
    return jsonify({'error': error.description}), error.code


@app.get('/')
def index():
    return jsonify({
        'service': 'mbcaptcha',
        'health': '/health',
        'predict': '/predict',
        'inputs': ['multipart: image', 'json: image_base64'],
    })


@app.get('/health')
def health():
    return jsonify({'status': 'ok'})


@app.post('/predict')
def predict():
    try:
        if 'image' in request.files:
            image_data = request.files['image'].read()
            if not image_data:
                raise ValueError('Image is empty.')
        else:
            payload = request.get_json(silent=True)
            if not isinstance(payload, dict) or 'image_base64' not in payload:
                raise ValueError('Provide multipart image or JSON image_base64.')
            image_data = decode_base64_image(payload['image_base64'])

        prediction = predictor.predict(image_data)
        return jsonify({'prediction': prediction})
    except (ValueError, OSError, UnidentifiedImageError, Image.DecompressionBombError) as error:
        return jsonify({'error': str(error)}), 400
    except HTTPException:
        raise
    except Exception:
        app.logger.exception('Prediction failed')
        return jsonify({'error': 'Prediction failed.'}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=False)