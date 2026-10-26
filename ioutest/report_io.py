"""Small report writers shared with the training package's evaluation helpers."""
import csv
import io
from .common import atomic_text

def write_csv(path, header, rows):
    stream = io.StringIO(); writer = csv.writer(stream)
    writer.writerow(header); writer.writerows(rows)
    atomic_text(path, stream.getvalue())

def write_dict_csv(path, rows):
    names = list(dict.fromkeys(k for r in rows for k in r))
    write_csv(path, names, [[r.get(n) for n in names] for r in rows])
