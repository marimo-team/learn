#!/usr/bin/env python

import datetime
import faker
import itertools
import random
import sqlite3
import sys


LOCALE = "es"

NUM_PERSONS = 6

DATE_START = datetime.date(2025, 9, 1)
DATE_END = datetime.date(2025, 12, 31)
DATE_DURATION = 7

NUM_MACHINES = 5

CREATE_PERSONS = """\
create table person(
    person_id text not null primary key,
    personal text not null,
    family text not null,
    supervisor_id text,
    foreign key(supervisor_id) references person(person_id)
);
"""
INSERT_PERSONS = """\
insert into person values (:person_id, :personal, :family, :supervisor_id);
"""

CREATE_SURVEYS = """\
create table survey(
    survey_id text not null primary key,
    person_id text not null,
    start_date text,
    end_date text,
    foreign key(person_id) references person(person_id)
);
"""
INSERT_SURVEYS = """\
insert into survey values(:survey_id, :person_id, :start, :end);
"""

CREATE_MACHINES = """\
create table machine(
    machine_id text not null primary key,
    machine_type text not null
);
"""
INSERT_MACHINES = """\
insert into machine values(:machine_id, :machine_type);
"""

CREATE_RATINGS = """\
create table rating(
    person_id text not null,
    machine_id text not null,
    level integer,
    foreign key(person_id) references person(person_id),
    foreign key(machine_id) references machine(machine_id)
);
"""
INSERT_RATINGS = """\
insert into rating values(:person_id, :machine_id, :level);
"""

def main():
    db_name = sys.argv[1]
    seed = int(sys.argv[2])
    random.seed(seed)

    persons_counter = itertools.count()
    next(persons_counter)
    persons = gen_persons(NUM_PERSONS, persons_counter)

    supers = gen_persons(int(NUM_PERSONS / 2), persons_counter)
    for p in persons:
        p["supervisor_id"] = random.choice(supers)["person_id"]
    if len(supers) > 1:
        supers[0]["supervisor_id"] = supers[-1]["person_id"]

    surveys = gen_surveys(persons + supers[0:int(len(supers)/2)])
    surveys[int(len(surveys)/2)]["start"] = None

    cnx = sqlite3.connect(db_name)
    cur = cnx.cursor()

    everyone = persons + supers
    random.shuffle(everyone)
    cur.execute(CREATE_PERSONS)
    cur.executemany(INSERT_PERSONS, everyone)

    cur.execute(CREATE_SURVEYS)
    cur.executemany(INSERT_SURVEYS, surveys)

    machines = gen_machines()
    cur.execute(CREATE_MACHINES)
    cur.executemany(INSERT_MACHINES, machines)

    ratings = gen_ratings(everyone, machines)
    cur.execute(CREATE_RATINGS)
    cur.executemany(INSERT_RATINGS, ratings)

    cnx.commit()
    cnx.close()


def gen_machines():
    adjectives = "hydraulic rotary modular industrial automated".split()
    nouns = "press conveyor generator actuator compressor".split()
    machines = set()
    while len(machines) < NUM_MACHINES:
        candidate = f"{random.choice(adjectives)} {random.choice(nouns)}"
        if candidate not in machines:
            machines.add(candidate)
    counter = itertools.count()
    next(counter)
    return [
        {"machine_id": f"M{next(counter):04d}", "machine_type": m}
        for m in machines
    ]


def gen_persons(num, counter):
    fake = faker.Faker(LOCALE)
    fake.seed_instance(random.randint(0, 1_000_000))
    return [
        {
            "person_id": f"P{next(counter):03d}",
            "personal": fake.first_name(),
            "family": fake.last_name(),
            "supervisor_id": None,
        }
        for _ in range(num)
    ]


def gen_ratings(persons, machines):
    temp = {}
    while len(temp) < int(len(persons) * len(machines) / 4):
        p = random.choice(persons)["person_id"]
        m = random.choice(machines)["machine_id"]
        if (p, m) in temp:
            continue
        temp[(p, m)] = random.choice([None, 1, 2, 3])
    return [
        {"person_id": p, "machine_id": m, "level": v}
        for ((p, m), v) in temp.items()
    ]

def gen_surveys(persons):
    surveys = []
    counter = itertools.count()
    next(counter)
    for person in persons:
        person_id = person["person_id"]
        start = DATE_START
        while start <= DATE_END:
            survey_id = f"S{next(counter):04d}"
            end = start + datetime.timedelta(days=random.randint(1, DATE_DURATION))
            surveys.append({
                "survey_id": survey_id,
                "person_id": person_id,
                "start": start.isoformat(),
                "end": end.isoformat() if end <= DATE_END else None
            })
            start = end + datetime.timedelta(days=random.randint(1, DATE_DURATION))
    return surveys


if __name__ == "__main__":
    main()
