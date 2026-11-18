"""Reuse fal measurement-sheet edits and resume their recorded queue requests."""

import fcntl
import hashlib
import io
import json
import os
import tempfile
import subprocess
import time
from pathlib import Path

from PIL import Image


def request_images(input_png, fields, cache):
    def command(arguments):
        result = subprocess.run(
            ["genmedia", *arguments, "--json"],
            check=True,
            capture_output=True,
            text=True,
        )
        return json.loads(result.stdout)

    endpoint = fields["model"]
    pending = cache / "request.json"
    submitted = not pending.exists()
    if pending.exists():
        request = json.loads(pending.read_text())
    else:
        upload = command(["upload", str(input_png.resolve())])
        width, height = map(int, fields["size"].split("x"))
        params = dict(
            prompt=fields["prompt"],
            image_urls=[upload["cdn_url"]],
            image_size=dict(width=width, height=height),
            quality=fields["quality"],
            num_images=int(fields["n"]),
            background="opaque",
            output_format="png",
        )
        args = ["run", endpoint, "--async"]
        for name, value in params.items():
            args.extend(
                [
                    "--" + name,
                    json.dumps(value)
                    if isinstance(value, (dict, list))
                    else str(value),
                ]
            )
        request = command(args)
        request.update(endpoint=endpoint, parameters=params)
        pending.write_text(json.dumps(request, indent=2))
    download = cache / "download"
    download.mkdir(exist_ok=True)
    while True:
        result = command(["status", endpoint, request["request_id"]])
        state = result.get("status", "").lower()
        if state in ("completed", "complete"):
            break
        if state in ("failed", "error", "cancelled", "canceled"):
            raise RuntimeError(f"Measurement image request failed: {pending}")
        time.sleep(10)
    result = command(
        ["status", endpoint, request["request_id"], "--download", str(download) + "/"]
    )
    files = [Path(item["path"]) for item in result["downloaded_files"]]
    expected = tuple(map(int, fields["size"].split("x")))
    for file in files:
        with Image.open(file) as image:
            if image.size != expected:
                raise ValueError(
                    f"Unexpected native image dimensions {image.size}, expected {expected}"
                )
    return [file.read_bytes() for file in files], dict(
        provider="fal",
        endpoint=endpoint,
        request_id=request["request_id"],
        api_calls=int(submitted),
        response=result,
    )


def sheets(folder, input_png, fields, refresh=False):
    with Image.open(input_png) as source:
        rgb = source.convert("RGB")
        pixels = hashlib.sha256(rgb.tobytes()).hexdigest()
    guide = json.loads((folder / "guidance.json").read_text())
    signature = dict(
        source=guide["source_hash"], dimensions=rgb.size, pixels=pixels, fields=fields
    )
    fingerprint = hashlib.sha256(
        json.dumps(signature, sort_keys=True).encode()
    ).hexdigest()
    cache = folder / "views" / "measurement-cache" / fingerprint
    cache.mkdir(parents=True, exist_ok=True)
    receipt = cache / "receipt.json"
    with (cache / "lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if receipt.exists() and not refresh:
            saved = json.loads(receipt.read_text())
            hashes = saved["outputs"]
            result = [(cache / (h + ".png")).read_bytes() for h in hashes]
            if len(result) != int(fields["n"]) or any(
                hashlib.sha256(v).hexdigest() != h for v, h in zip(result, hashes)
            ):
                raise ValueError(f"Damaged cached measurement: {receipt}")
            return result, dict(cache_hit=True, api_calls=0, request=fingerprint, usage=saved.get("usage"))
        pending = cache / "request.json"
        if refresh and receipt.exists() and pending.exists():
            previous = (
                json.loads(receipt.read_text()).get("usage", {}).get("request_id")
            )
            current = json.loads(pending.read_text())["request_id"]
            if previous == current:
                pending.rename(cache / ("completed-" + current + ".json"))
                (cache / "download").rename(cache / ("download-" + current))
        images, usage = request_images(input_png, fields, cache)
        if len(images) != int(fields["n"]):
            raise ValueError("Incomplete measurement sheet response")
        hashes = []
        for raw in images:
            with Image.open(io.BytesIO(raw)) as im:
                im.verify()
            h = hashlib.sha256(raw).hexdigest()
            (cache / (h + ".png")).write_bytes(raw)
            hashes.append(h)
        with tempfile.NamedTemporaryFile("w", dir=cache, delete=False) as tmp:
            json.dump(dict(signature=signature, outputs=hashes, usage=usage), tmp)
        os.replace(tmp.name, receipt)
        return images, dict(
            cache_hit=False, api_calls=usage["api_calls"], request=fingerprint, usage=usage
        )
