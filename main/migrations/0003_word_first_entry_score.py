# Generated by Django 4.0.3 on 2022-04-22 04:31

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('main', '0002_word_date'),
    ]

    operations = [
        migrations.AddField(
            model_name='word',
            name='first_entry_score',
            field=models.FloatField(blank=True, null=True),
        ),
    ]
