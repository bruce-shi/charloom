"""Resolve local glTF buffers and pack companion images for Blender and GLB export."""

import base64
import mimetypes
import struct
from pathlib import Path
from urllib.parse import unquote, unquote_to_bytes, urlsplit


def resource(base, uri):
    if uri.startswith("data:"):
        header, encoded = uri[5:].split(",", 1)
        return (
            base64.b64decode(encoded, validate=True)
            if header.endswith(";base64")
            else unquote_to_bytes(encoded)
        ), header.split(";")[0]
    address = urlsplit(uri)
    if address.scheme or address.netloc:
        raise ValueError(f"Expected a local glTF resource: {uri}")
    path = base / unquote(address.path)
    return path.read_bytes(), mimetypes.guess_type(path)[0]


def pack(document, base, binary):
    packed = bytearray()
    offsets = []
    for i, desc in enumerate(document.get("buffers", [])):
        if "uri" in desc:
            data = resource(base, desc["uri"])[0]
        elif i == 0:
            data = binary
        else:
            raise ValueError("Missing URI for external glTF buffer")
        length = desc["byteLength"]
        if len(data) < length:
            raise ValueError(f"Truncated glTF buffer {i}")
        packed.extend(bytes(-len(packed) % 4))
        offsets.append(len(packed))
        packed.extend(data[:length])
    views = document.setdefault("bufferViews", [])
    for view in views:
        offset = offsets[view.get("buffer", 0)] + view.get("byteOffset", 0)
        view.update(buffer=0, byteOffset=offset)
    for image in document.get("images", []):
        if "uri" in image:
            data, mime = resource(base, image.pop("uri"))
            packed.extend(bytes(-len(packed) % 4))
            image["bufferView"] = len(views)
            image["mimeType"] = image.get("mimeType", mime)
            views.append(dict(buffer=0, byteOffset=len(packed), byteLength=len(data)))
            packed.extend(data)
    document["buffers"] = [dict(byteLength=len(packed))]
    return packed


def chunks(data):
    if len(data) < 20 or struct.unpack_from("<III", data) != (0x46546C67, 2, len(data)):
        raise ValueError("Invalid glTF 2 binary header")
    result = {}
    at = 12
    while at < len(data):
        size, kind = struct.unpack_from("<II", data, at)
        at += 8
        if at + size > len(data):
            raise ValueError("Truncated GLB chunk")
        result[kind] = data[at : at + size]
        at += size
    return result[0x4E4F534A], result.get(0x004E4942, b"")
