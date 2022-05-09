from django.db import models
import datetime

class Word(models.Model):
    word = models.CharField(max_length=255, null=True, blank=True)

    guess1 = models.CharField(max_length=255, null=True, blank=True)
    guess2 = models.CharField(max_length=255, null=True, blank=True)
    guess3 = models.CharField(max_length=255, null=True, blank=True)
    guess4 = models.CharField(max_length=255, null=True, blank=True)
    guess5 = models.CharField(max_length=255, null=True, blank=True)
    guess6 = models.CharField(max_length=255, null=True, blank=True)

    outcome1 = models.TextField(null=True, blank=True)
    outcome2 = models.TextField(null=True, blank=True)
    outcome3 = models.TextField(null=True, blank=True)
    outcome4 = models.TextField(null=True, blank=True)
    outcome5 = models.TextField(null=True, blank=True)
    outcome6 = models.TextField(null=True, blank=True)

    attempts = models.IntegerField(null=True, blank=True)
    attempts_brain = models.IntegerField(null=True, blank=True)

    first_entry_score = models.FloatField(null=True, blank=True)

    date = models.DateField(auto_now=False, null=True)

    def __str__(self):
        if self.date < datetime.date.today():
            return self.word + " (" + self.date.strftime('%d %b %Y') + ")"
        else:
            return self.word

    def capitals(self):
        return self.word.upper()