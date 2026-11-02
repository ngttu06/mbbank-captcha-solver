"""Exercise a running API over HTTP, including concurrent uploads."""

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path

from PIL import Image
import requests


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('url', help='API base URL, without /predict')
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--expected', help='Expected prediction for the supplied image')
    parser.add_argument(
        '--check-size-limit', action='store_true',
        help='Send a 2 MiB body; use against localhost to avoid proxy early-rejection upload timeouts',
    )
    args = parser.parse_args()
    url = args.url.rstrip('/')
    image = args.image.read_bytes()

    response = requests.get(url + '/health', timeout=15)
    response.raise_for_status()
    assert response.json() == {'status': 'ok'}, response.text

    def upload(data):
        response = requests.post(
            url + '/predict', files={'image': ('captcha.png', data, 'image/png')}, timeout=30
        )
        response.raise_for_status()
        prediction = response.json()['prediction']
        assert isinstance(prediction, str) and len(prediction) == 6, response.text
        return prediction

    expected = upload(image)
    if args.expected:
        assert expected == args.expected, (expected, args.expected)
    encoded = base64.b64encode(image).decode('ascii')
    for value in [encoded, 'data:image/png;base64,' + encoded]:
        response = requests.post(url + '/predict', json={'image_base64': value}, timeout=30)
        response.raise_for_status()
        assert response.json() == {'prediction': expected}, response.text

    for payload in [{}, [], {'image_base64': '!'}, {'image_base64': None}]:
        response = requests.post(url + '/predict', json=payload, timeout=15)
        assert response.status_code == 400, (response.status_code, response.text)
        assert 'error' in response.json(), response.text

    response = requests.post(
        url + '/predict', files={'image': ('invalid.png', b'not an image')}, timeout=15
    )
    assert response.status_code == 400, (response.status_code, response.text)
    if args.check_size_limit:
        response = requests.post(
            url + '/predict', json={'image_base64': 'A' * (2 * 1024 * 1024)}, timeout=30
        )
        assert response.status_code == 413, (response.status_code, response.text[:300])
    assert requests.get(url + '/predict', timeout=15).status_code == 405

    samples = [image]
    for color in ['white', 'black', 'red']:
        buffer = BytesIO()
        Image.new('RGB', (160, 50), color).save(buffer, format='PNG')
        samples.append(buffer.getvalue())
    sequential = [upload(sample) for sample in samples]
    with ThreadPoolExecutor(max_workers=4) as pool:
        concurrent = list(pool.map(upload, samples * 3))
    assert concurrent == sequential * 3, (concurrent, sequential)
    size_check = ', size limit' if args.check_size_limit else ''
    print(f'PASS {url}: health, upload, Base64, data URI, invalid inputs{size_check}, 12 concurrent predictions')
    print(f'Sample prediction: {expected}')


if __name__ == '__main__':
    main()