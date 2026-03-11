create table job (
    name text not null,
    credits real not null
);

create table work (
    person text not null,
    job text not null
);

insert into job values
('calibrate', 1.5),
('clean', 0.5);

insert into work values
('Amal', 'calibrate'),
('Amal', 'clean'),
('Amal', 'complain'),
('Gita', 'clean'),
('Gita', 'clean'),
('Gita', 'complain'),
('Madhi', 'complain');
