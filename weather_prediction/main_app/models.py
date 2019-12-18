from django.db import models

# Create your models here.
class CurrentWeather(models.Model):
	weather_id = models.CharField(max_length=10)
	temp = models.FloatField()
	temp_min = models.FloatField()
	temp_max = models.FloatField()
	pressure = models.IntegerField(default=0.0)
	humidity = models.IntegerField(default=0)
	speed = models.FloatField(default=0.0)
	deg = models.IntegerField(default=0)
	year = models.IntegerField()
	month = models.IntegerField()
	day = models.IntegerField()
	hour = models.IntegerField()
	minute = models.IntegerField()
	description = models.CharField(max_length=20)
	clouds = models.CharField(max_length=10)

	class Meta:
		app_label = 'main_app'
		db_table = "currentweather"
