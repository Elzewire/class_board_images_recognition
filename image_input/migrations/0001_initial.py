# Generated by Django 3.2.4 on 2021-06-15 19:59

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Discipline',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=128, verbose_name='Название')),
            ],
            options={
                'verbose_name': 'Дисциплина',
                'verbose_name_plural': 'Дисциплины',
            },
        ),
        migrations.CreateModel(
            name='Timetable',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('year', models.IntegerField(choices=[(2016, '2016'), (2017, '2017'), (2018, '2018'), (2019, '2019'), (2020, '2020'), (2021, '2021')], verbose_name='Год')),
            ],
            options={
                'verbose_name': 'Расписание',
                'verbose_name_plural': 'Расписания',
            },
        ),
        migrations.CreateModel(
            name='Entry',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('day', models.CharField(choices=[('MON', 'Monday'), ('TUE', 'Tuesday'), ('WED', 'Wednesday'), ('THU', 'Thursday'), ('Fri', 'Friday'), ('SAT', 'Saturday'), ('SUN', 'Sunday')], max_length=3, verbose_name='День')),
                ('time', models.CharField(choices=[('8:30-10:00', '8:30-10:00'), ('10:10-11:40', '10:10-11:40'), ('11:50-13:20', '11:50-13:20'), ('13:35-15:05', '13:35-15:05'), ('13:40-15:10', '13:40-15:10'), ('14:00-15:30', '14:00-15:30'), ('15:20-16:50', '15:20-16:50'), ('15:40-17:10', '15:40-17:10'), ('17:00-18:30', '17:00-18:30'), ('17:20-18:50', '17:20-18:50'), ('18:40-20:10', '18:40-20:10'), ('19:00-20:30', '19:00-20:30')], max_length=32, verbose_name='Время')),
                ('discipline', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='image_input.discipline', verbose_name='Дисциплина')),
                ('timetable', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='image_input.timetable', verbose_name='Расписание')),
            ],
            options={
                'verbose_name': 'Запись',
                'verbose_name_plural': 'Записи',
            },
        ),
    ]