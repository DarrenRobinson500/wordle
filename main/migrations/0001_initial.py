# Generated by Django 4.0.3 on 2022-04-03 06:20

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Word',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('word', models.CharField(blank=True, max_length=255, null=True)),
                ('guess1', models.CharField(blank=True, max_length=255, null=True)),
                ('guess2', models.CharField(blank=True, max_length=255, null=True)),
                ('guess3', models.CharField(blank=True, max_length=255, null=True)),
                ('guess4', models.CharField(blank=True, max_length=255, null=True)),
                ('guess5', models.CharField(blank=True, max_length=255, null=True)),
                ('guess6', models.CharField(blank=True, max_length=255, null=True)),
                ('outcome1', models.TextField(blank=True, null=True)),
                ('outcome2', models.TextField(blank=True, null=True)),
                ('outcome3', models.TextField(blank=True, null=True)),
                ('outcome4', models.TextField(blank=True, null=True)),
                ('outcome5', models.TextField(blank=True, null=True)),
                ('outcome6', models.TextField(blank=True, null=True)),
                ('attempts', models.IntegerField(blank=True, null=True)),
            ],
        ),
    ]