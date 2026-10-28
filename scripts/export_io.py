"""Stable publication writes; content transformations belong in the exporters."""
import io
import json
import zipfile


def write_bytes(path, data):
    """Do not touch an unchanged output (including its timestamp)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.read_bytes() != data:
        path.write_bytes(data)


def write_text(path, text):
    data = text.replace('\r\n', '\n').encode('utf8')
    # A checkout's CRLF conversion alone is not a publication change.
    if path.exists() and path.read_bytes().replace(b'\r\n', b'\n') == data:
        return
    write_bytes(path, data)


def write_json(path, data):
    write_text(path, json.dumps(data, ensure_ascii=False, indent=2) + '\n')


def merge_courses(previous, current, selected, key):
    """Replace selected courses in their existing order; retain all other records."""
    result = []
    for item in previous:
        course = item[key]
        if course not in selected:
            result.append(item)
        elif course not in {record[key] for record in result}:
            result.extend(record for record in current if record[key] == course)
    result.extend(record for record in current if record[key] not in {item[key] for item in previous})
    return result


def write_bundle(path, members):
    """Reuse identical archives; rebuild changed archives deterministically."""
    if path.exists():
        with zipfile.ZipFile(path) as old:
            if set(old.namelist()) == set(members) and all(old.read(name) == raw for name, raw in members.items()):
                return
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name, raw in sorted(members.items()):
            info = zipfile.ZipInfo(name, date_time=(2026, 10, 16, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, raw)
    write_bytes(path, buffer.getvalue())
