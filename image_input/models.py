from django.db import models
from django.db.models import CASCADE

YEAR_CHOICES = [
    ('2016, 1s', '2016, 1s'),
    ('2016, 2s', '2016, 2s'),
    ('2017, 1s', '2017, 1s'),
    ('2017, 2s', '2017, 2s'),
    ('2018, 1s', '2018, 1s'),
    ('2018, 2s', '2018, 2s'),
    ('2019, 1s', '2019, 1s'),
    ('2019, 2s', '2019, 2s'),
    ('2020, 1s', '2020, 1s'),
    ('2020, 2s', '2020, 2s'),
    ('2021, 1s', '2021, 1s'),
    ('2021, 2s', '2021, 2s')
]

DAY_CHOICES = [
    ('MON', 'Monday'),
    ('TUE', 'Tuesday'),
    ('WED', 'Wednesday'),
    ('THU', 'Thursday'),
    ('FRI', 'Friday'),
    ('SAT', 'Saturday'),
    ('SUN', 'Sunday'),
]

TIME_CHOICES = [
    ('8:30-10:00', '8:30-10:00'),
    ('10:10-11:40', '10:10-11:40'),
    ('11:50-13:20', '11:50-13:20'),
    ('13:35-15:05', '13:35-15:05'),
    ('13:40-15:10', '13:40-15:10'),
    ('14:00-15:30', '14:00-15:30'),
    ('15:20-16:50', '15:20-16:50'),
    ('15:40-17:10', '15:40-17:10'),
    ('17:00-18:30', '17:00-18:30'),
    ('17:20-18:50', '17:20-18:50'),
    ('18:40-20:10', '18:40-20:10'),
    ('19:00-20:30', '19:00-20:30'),
]

DISCIPLINE_CHOICES = [
    ('LEC', 'Лекция'),
    ('PRC', 'Практика')
]

WEEKDAY_MAP = {
    0: 'MON',
    1: 'TUE',
    2: 'WED',
    3: 'THU',
    4: 'FRI',
    5: 'SAT',
    6: 'SUN'
}


class Discipline(models.Model):
    name = models.CharField(max_length=128, verbose_name='Название')
    type = models.CharField(max_length=3, choices=DISCIPLINE_CHOICES, verbose_name='Тип занятия')

    def __str__(self):
        return '{} - {}'.format(self.name, self.type)

    class Meta:
        ordering = ['name']
        verbose_name = 'Дисциплина'
        verbose_name_plural = 'Дисциплины'


class Timetable(models.Model):
    year = models.CharField(max_length=32, choices=YEAR_CHOICES, verbose_name='Год')

    def __str__(self):
        return self.year

    class Meta:
        verbose_name = 'Расписание'
        verbose_name_plural = 'Расписания'


class Entry(models.Model):
    timetable = models.ForeignKey(Timetable, on_delete=CASCADE, verbose_name='Расписание')
    disciplines = models.ManyToManyField(Discipline, verbose_name='Дисциплины')
    day = models.CharField(max_length=3, choices=DAY_CHOICES, verbose_name='День')
    time = models.CharField(max_length=32, choices=TIME_CHOICES, verbose_name='Время')

    def __str__(self):
        return '{} {} {}'.format(self.timetable, self.day, self.time)

    class Meta:
        verbose_name = 'Запись'
        verbose_name_plural = 'Записи'
