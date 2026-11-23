#!/usr/bin/env python3
"""Export the first sheet of instruction-timing-table.xlsx to instruction-timing-table.csv.

Standard library only (zipfile + ElementTree), so the build needs no openpyxl.

The sheet groups rows visually: a unit (MEM, VXM, ...) or an instruction name is written once
and left blank on the rows below it. LatencyTable::load_csv (src/lpusim.cpp) matches rows by
(unit, instruction, datatype), so blank unit / instruction cells are filled down here.
Whole-number floats ("1.0") are written as integers ("1"), because load_csv only accepts
integers; anything else (blank, "n/a", "20 per plane") is copied as is and load_csv keeps the
default for that cell.

    python3 planning/xlsx_to_csv.py [in.xlsx] [out.csv]
"""
import csv
import os
import re
import sys
import xml.etree.ElementTree as ET
import zipfile

NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
HERE = os.path.dirname(os.path.abspath(__file__))


def col_index(ref):
    n = 0
    for ch in re.match(r"[A-Z]+", ref).group():
        n = n * 26 + ord(ch) - 64
    return n - 1


def read_rows(path):
    z = zipfile.ZipFile(path)
    shared = []
    if "xl/sharedStrings.xml" in z.namelist():
        for si in ET.fromstring(z.read("xl/sharedStrings.xml")).findall(NS + "si"):
            shared.append("".join(t.text or "" for t in si.iter(NS + "t")))
    sheet = ET.fromstring(z.read("xl/worksheets/sheet1.xml"))
    rows = []
    for row in sheet.find(NS + "sheetData"):
        cells = {}
        for c in row:
            v = c.find(NS + "v")
            if v is None:
                inline = c.find(NS + "is")
                val = "".join(t.text or "" for t in inline.iter(NS + "t")) if inline is not None else ""
            elif c.get("t") == "s":
                val = shared[int(v.text)]
            else:
                val = v.text or ""
            cells[col_index(c.get("r"))] = val
        rows.append([cells.get(i, "") for i in range(max(cells) + 1)] if cells else [])
    return rows


def clean(v):
    if re.fullmatch(r"-?\d+\.0+", v):
        return v.split(".")[0]
    return v


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "instruction-timing-table.xlsx")
    dst = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "instruction-timing-table.csv")
    rows = read_rows(src)
    header, body = rows[0], rows[1:]
    width = len(header)
    unit = instr = ""
    out = []
    for r in body:
        r = [clean(x) for x in (r + [""] * width)[:width]]
        if not any(r):
            continue
        if r[0]:
            unit, instr = r[0], ""
        else:
            r[0] = unit
        if r[1]:
            instr = r[1]
        else:
            r[1] = instr
        out.append(r)
    with open(dst, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(header)
        w.writerows(out)
    print(f"{os.path.relpath(src)} -> {os.path.relpath(dst)}: {len(out)} rows")


if __name__ == "__main__":
    main()
