from django.urls import path
from django.contrib.auth import views as auth_views
from .views import *

urlpatterns = [
    path('', home, name='home'),
    path('home', home, name='home'),
    path('sandpit', sandpit, name='sandpit'),
    path('home/<entry>', home, name='home'),
    path('initial_load', initial_load, name='initial_load'),
    path('load_dates', load_dates, name='load_dates'),
    path('clear', clear, name='clear'),
    path('upcoming', upcoming, name='upcoming'),
    path('summary', summary, name='summary'),
    path('initial_word_distribution', initial_word_distribution, name='initial_word_distribution'),
    path('summary/<outcome1>', summary, name='summary'),
    path('solve_all/<attempts>', solve_all, name='solve_all'),

    path('test/<word>', test, name='test'),

]