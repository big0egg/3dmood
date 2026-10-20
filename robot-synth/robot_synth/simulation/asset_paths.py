"""Resolve author-machine MJCF asset paths without modifying original HDF5."""

from pathlib import Path
import xml.etree.ElementTree as ET


def resolve_asset(value, roots):
    value = value.replace("\\", "/")
    path = Path(value)
    candidates = []
    # Recognize the namespace before testing existence: avoid accepting arbitrary
    # same-named files from the working directory as substitutes.
    if "/robosuite/" in value and "/assets/" in value:
        relative = value.split("/assets/", 1)[1]
        candidates = [Path(roots["robosuite"]) / relative]
    elif "/assets/" in value:
        relative = value.split("/assets/", 1)[1]
        candidates = [Path(roots["libero"]) / relative]
    elif path.is_absolute():
        candidates = [path]
    else:
        candidates = [Path(root) / path for root in roots.values()]
    valid = sorted({p.resolve() for p in candidates if p.is_file()})
    if len(valid) != 1:
        raise FileNotFoundError(f"Cannot uniquely resolve asset {value}; checked: {', '.join(map(str, candidates))}")
    return valid[0]


def resolve_xml(xml, roots):
    root = ET.fromstring(xml)
    compiler = root.find("compiler")
    if compiler is not None:
        compiler.attrib.pop("meshdir", None)
        compiler.attrib.pop("texturedir", None)
    files = {}
    for node in root.iter():
        if node.tag == "include":
            raise ValueError("Unexpanded MJCF includes are not accepted in recorded scene XML")
        if "file" in node.attrib:
            original = node.get("file")
            resolved = resolve_asset(original, roots)
            files[original] = str(resolved)
            node.set("file", str(resolved))
    return ET.tostring(root, encoding="unicode"), files
