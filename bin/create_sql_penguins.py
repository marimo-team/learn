#!/usr/bin/env python

import csv
import sqlite3
import sys


SCHEMA = """
CREATE TABLE penguins (
    species text,
    island text,
    bill_length_mm real,
    bill_depth_mm real,
    flipper_length_mm real,
    body_mass_g real,
    sex text
);
"""

def main():
    infile = sys.argv[1]
    outfile = sys.argv[2]

    con = sqlite3.connect(outfile)
    con.execute(SCHEMA)

    with open(infile, newline="") as f:
        reader = csv.DictReader(f)
        rows = [
            (
                row["species"],
                row["island"],
                float(row["bill_length_mm"]) if row["bill_length_mm"] else None,
                float(row["bill_depth_mm"]) if row["bill_depth_mm"] else None,
                float(row["flipper_length_mm"]) if row["flipper_length_mm"] else None,
                float(row["body_mass_g"]) if row["body_mass_g"] else None,
                row["sex"] if row["sex"] else None,
            )
            for row in reader
        ]

    con.executemany(
        "INSERT INTO penguins VALUES (?, ?, ?, ?, ?, ?, ?)", rows
    )
    con.commit()
    con.close()


if __name__ == "__main__":
    main()
