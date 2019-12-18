import json
import time
import statistics as st
from datetime import date

import numpy as np
from sklearn import datasets
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from django.shortcuts import render
from django.http import HttpResponse
from .models import CurrentWeather

import requests
from django.forms import ModelForm, TextInput


from django.db import models

class City(models.Model):
    name = models.CharField(max_length=25)

    def __str__(self):
        return self.name

    class Meta:
        verbose_name_plural = 'cities'


class CityForm(ModelForm):
    class Meta:
        model = City 
        fields = ['name']
        widgets = {'name' : TextInput(attrs={'class' : 'input', 'placeholder' : 'City Name'})}


def checkIfNull(val):
	if val is None:
		return 0.0	############### Check if value is None
	else:
		return val

def checkIfMissing(val):
	if val.find('*') != -1:
		return 0.0	############### Check if value is missing
	else:
		return val

def fahrenheitToCelsius(val):
	return float("{:.2f}".format((int(val) - 32) / 1.8))

def unixToLocalTime(unix_timestamp):
    local_time = time.localtime(unix_timestamp)		############convert unixtime to local 
    return time.strftime("%Y-%m-%d %H:%M", local_time)

def getDateSeries(date_today):			####### get next date, day & month of forecast from today's
	month = date_today.strftime("%m")
	day = date_today.strftime("%d")
	month = int(month)
	day = int(day)
	M = []
	D = []
	for x in range(5):
		if month == 2 and (day + x + 1) > 28:
			m = month + 1
			d = day + x + 1 - 28
			M.append(m)
			D.append(d)
		elif month == 1 or month == 3 or month == 5 or month == 7 or month == 8 or month == 10 or month == 12:
			if day + x + 1 > 31:
				m = month + 1
				d = day + x + 1 - 31
				M.append(m)
				D.append(d)
			else:
				M.append(month)
				D.append(day + x + 1)
		elif month == 4 or month == 6 or month == 9 or month == 11:
			if day + x + 1 > 30:
				m = month + 1
				d = day + x + 1 - 30
				M.append(m)
				D.append(d)
			else:
				M.append(month)
				D.append(day + x + 1)
		else:
			M.append(month)
			D.append(day + x + 1)
	date_series = {
	'month' : M,
	'day' : D
	}
	return date_series

#print(getDateSeries(date.today()))


def makeDataSet(month, day):		############# making suitable dataset to train model
	searched_data = CurrentWeather.objects.filter(month=month, day=day)
	merged_row_data = []
	merged_row_target =[]
	for x in range(len(searched_data)):
		single_row_data = []
		merged_row_target.append(searched_data[x].temp)
		single_row_data.append(searched_data[x].pressure)
		single_row_data.append(searched_data[x].humidity)
		single_row_data.append(searched_data[x].speed)
		single_row_data.append(searched_data[x].deg)
		single_row_data.append(searched_data[x].temp_min)
		single_row_data.append(searched_data[x].temp_max)
		merged_row_data.append(single_row_data)
	dataset_processed ={
	'data' : merged_row_data,
	'target' : merged_row_target
	}
	return dataset_processed
#print(makeDataSet(10, 28))

def GNB(dataset, test_data):		##################### GNB #########################
	data = np.array(dataset['data'])
	target = np.array(dataset['target'])
	target = np.around(target)
	gnb = GaussianNB().fit(data, target)
	gnb_predictions = gnb.predict(test_data) 
	return gnb_predictions

def decisionTreeClassifier(dataset, test_data):		################# Decision Tree Classifier ###############
	data = np.array(dataset['data'])
	target = np.array(dataset['target'])
	target = np.around(target)
	dtree_model = DecisionTreeClassifier(max_depth = 2).fit(data, target)
	dtree_predictions = dtree_model.predict(test_data)
	return dtree_predictions

def SVMC(dataset, test_data):
	data = np.array(dataset['data'])
	target = np.array(dataset['target'])
	target = np.around(target)
	svm_model_linear = SVC(kernel = 'linear', C = 1).fit(data, target) 
	svm_predictions = svm_model_linear.predict(test_data) 
	return svm_predictions

def KNN(dataset, test_data):
	data = np.array(dataset['data'])
	target = np.array(dataset['target'])
	target = np.around(target)
	knn = KNeighborsClassifier(n_neighbors = 7).fit(data, target)
	knn_predictions = knn.predict(test_data)

def fiveDayForecastCustom(date_today, weather_today):
	pressure = weather_today['pressure']
	humidity = weather_today['humidity']
	speed = weather_today['speed']
	deg = weather_today['deg']
	temp_min = weather_today['temp_min']
	temp_max = weather_today['temp_max']	#####################################
											#									#
	test_data = []							#									#
	test_data.append(pressure)				# ******making test data************#
	test_data.append(humidity)				#									#
	test_data.append(speed)					#####################################
	test_data.append(deg)
	test_data.append(temp_min)
	test_data.append(temp_max)
	test_data = [test_data]

	date_series = getDateSeries(date_today) #####################################
	month_series = date_series['month']		##*********collecting 5 days date***#
	day_series = date_series['day']			#####################################

	gnb_output = []
	dtree_output = []
	svmc_output = []
	knn_output = []
	for x in range(5):														 #####################################
		dataset = makeDataSet(month_series[x], day_series[x])				 #									 #
		gnb_output.append(int(GNB(dataset, test_data)))						 #*******applying algorithm**********#
		dtree_output.append(int(decisionTreeClassifier(dataset, test_data))) #####################################
		svmc_output.append(int(SVMC(dataset, test_data)))
		#knn_output.append(KNN(dataset, test_data))

	custom_output = {
	'month_series' : month_series,
	'day_series' : day_series,
	'gnb_output' : gnb_output,
	'dtree_output' : dtree_output,
	'svmc_output' : svmc_output,
	}
	#print(custom_output)
	return custom_output

def accuracyCalculation(api_forecast, our_forecast):
	api_data = []
	our_data = []
	gnb_accuracy = []
	dtree_accuracy = []
	svmc_accuracy = []
	avg_accuracy = []

	api_data.append(api_forecast['temp'][7])
	api_data.append(api_forecast['temp'][15])
	api_data.append(api_forecast['temp'][23])
	api_data.append(api_forecast['temp'][31])
	api_data.append(api_forecast['temp'][39])

	our_data.append(our_forecast['gnb_output'])
	our_data.append(our_forecast['dtree_output'])
	our_data.append(our_forecast['svmc_output'])

	for x in range(5):
		error = (abs(api_data[x] - our_data[0][x]) / api_data[x]) * 100
		gnb_accuracy.append(float("{:.2f}".format(100 - error))) # rouding upto two decimal point

		error = (abs(api_data[x] - our_data[1][x]) / api_data[x]) * 100
		dtree_accuracy.append(float("{:.2f}".format(100 - error)))

		error = (abs(api_data[x] - our_data[2][x]) / api_data[x]) * 100
		svmc_accuracy.append(float("{:.2f}".format(100 - error)))

	avg_accuracy.append("{:.2f}".format(st.mean(gnb_accuracy)))
	avg_accuracy.append("{:.2f}".format(st.mean(dtree_accuracy)))
	avg_accuracy.append("{:.2f}".format(st.mean(svmc_accuracy)))

	accuracy = {
	'gnb' : gnb_accuracy,
	'dtree' : dtree_accuracy,
	'svmc' : svmc_accuracy,
	'avg' : avg_accuracy,
	}

	#print(gnb_accuracy, dtree_accuracy, svmc_accuracy, avg_accuracy)
	return accuracy


def addCurrentWeatherFromJSON(weather_current):
	weather_id = weather_current['id']
	temp = weather_current['temp']
	temp_min = weather_current['temp_min']
	temp_max = weather_current['temp_max']
	pressure = weather_current['pressure']
	humidity = weather_current['humidity']
	speed = weather_current['speed']
	deg = weather_current['deg']
	year = weather_current['year']
	month = weather_current['month']
	day = weather_current['day']
	hour = weather_current['hour']
	minute = weather_current['minute']
	description = weather_current['description']
	clouds = weather_current['clouds']
	current_weather = CurrentWeather(weather_id=weather_id,temp=temp,temp_min=temp_min,temp_max=temp_max,
									 pressure=pressure,humidity=humidity,speed=speed,deg=deg,year=year,month=month,
									 day=day,hour=hour,minute=minute,description=description,clouds=clouds)
	current_weather.save()
	
def collectHistoricalData():
	with open(r'F:\Django\#Data\hourly.json') as json_file:
	    data = json.load(json_file)
	    data = data['data']
	    for x in range(len(data)):

	    	_date_ = checkIfNull(data[x]['time_local'])
	    	year = _date_[0] + _date_[1] + _date_[2] + _date_[3]
	    	month = _date_[5] + _date_[6]
	    	day = _date_[8] + _date_[9]
	    	hour = _date_[11] + _date_[12]
	    	minute = _date_[14] + _date_[15]

	    	weather_history = {
	    	'id' : "null",
	    	'description' : "null",
	    	'temp' : checkIfNull(data[x]['temperature']),
	    	'pressure' : checkIfNull(data[x]['pressure']),
	    	'humidity' : checkIfNull(data[x]['humidity']),
	    	'temp_min' : checkIfNull(data[x]['temperature']),
	    	'temp_max' : checkIfNull(data[x]['temperature']),
	    	'speed' : checkIfNull(data[x]['windspeed']),
	    	'deg' : checkIfNull(data[x]['winddirection']),
	    	'clouds' : 0,
	    	'year' : year,
	    	'month' : month,
	    	'day' : day,
	    	'hour' : hour,
	    	'minute' : minute,
	    	}
	    	addCurrentWeatherFromJSON(weather_history)
	    	print("Data Row:" , x)

########################################collectHistoricalData(#)########################

def collectHistoricalDataNCDC():
	f = open(r'F:\Django\#Data\NCDC\hourly\NCDC.txt', "r")
	data = f.readline()

	for x in range(61609):
		data = f.readline()
		data = data.split()

		_date_ = checkIfMissing(data[2])
		year = _date_[0] + _date_[1] + _date_[2] + _date_[3]
		month = _date_[4] + _date_[5]
		day = _date_[6] + _date_[7]
		hour = _date_[8] + _date_[9]
		minute = _date_[10] + _date_[11]

		weather_history = {
		'id' : "null",
		'description' : checkIfMissing(data[7]),
		'temp' : fahrenheitToCelsius(checkIfMissing(data[21])),
		'pressure' : float(checkIfMissing(data[23])),
		'humidity' : 0,
		'temp_min' : fahrenheitToCelsius(checkIfMissing(data[21])),
		'temp_max' : fahrenheitToCelsius(checkIfMissing(data[21])),
		'speed' : float(checkIfMissing(data[4])),
		'deg' : float(checkIfMissing(data[3])),
		'clouds' : float(checkIfMissing(data[6])),
		'year' : year,
		'month' : month,
		'day' : day,
		'hour' : hour,
		'minute' : minute,
		}
		#######addCurrentWeatherFromJSON(weather_history)
		print("Data Row:" , x)

########################################collectHistoricalDataNCDC()#############################

def homeView(request):
	num1 = 10
	num2 = 20
	print("The Sum of", num1, "and", num2, "is:", num1+num2)
	#return HttpResponse("<div style='width:200px; height: 200px; font-size: 40px; color:red; background:green;'>Farhad Raihan</div>")
	return render(request, "html/home.html", {"sum" : num1 + num2})


def index(request):
	#API KEY for 5 days forecast: 9753f805347655df4767d49e4476c913
    url_current = 'http://api.openweathermap.org/data/2.5/weather?q={},{}&units=metric&appid=9753f805347655df4767d49e4476c913'
    url_forecast = 'http://api.openweathermap.org/data/2.5/forecast?q={},{}&units=metric&appid=9753f805347655df4767d49e4476c913'
    city = 'Dhaka'
    country = 'BD'
    data_current = requests.get(url_current.format(city,country)).json()
    data_forecast = requests.get(url_forecast.format(city,country)).json()

    _date_ = data_current['dt']
    _date_ = unixToLocalTime(_date_)
    year = _date_[0] + _date_[1] + _date_[2] + _date_[3]
    month = _date_[5] + _date_[6]
    day = _date_[8] + _date_[9]
    hour = _date_[11] + _date_[12]
    minute = _date_[14] + _date_[15]

    weather_current = {
    	'lon' : data_current['coord']['lon'],
    	'lat' : data_current['coord']['lat'],
    	'id' : data_current['weather'][0]['id'],
    	'main' : data_current['weather'][0]['main'],
    	'description' : data_current['weather'][0]['description'],
    	'icon' : data_current['weather'][0]['icon'],
    	'temp' : data_current['main']['temp'],
    	'pressure' : data_current['main']['pressure'],
    	'humidity' : data_current['main']['humidity'],
    	'temp_min' : data_current['main']['temp_min'],
    	'temp_max' : data_current['main']['temp_max'],
    	#'sea_level' : data_current['main']['sea_level'],
    	#'grnd_level' : data_current['main']['grnd_level'],
    	'speed' : data_current['wind']['speed'],
    	'deg' : data_current['wind']['deg'],
    	'clouds' : data_current['clouds']['all'],
    	'year' : year,
    	'month' : month,
    	'day' : day,
    	'hour' : hour,
    	'minute' : minute,
    	'country' : data_current['name'],
    }
    from datetime import date
    #print(fiveDayForecastCustom(date.today(), weather_current))
    custom_output = fiveDayForecastCustom(date.today(), weather_current)

    date = []
    temp = []
    temp_min = []
    temp_max = []
    pressure = []
    sea_level =[]
    grnd_level = []
    humidity = []
    description = []
    icon = []
    speed = []
    deg = []
    rain = []

    for x in range(40):
    	date.append(data_forecast['list'][x]['dt'])
    	temp.append(data_forecast['list'][x]['main']['temp'])
    	temp_min.append(data_forecast['list'][x]['main']['temp_min'])
    	temp_max.append(data_forecast['list'][x]['main']['temp_max'])
    	pressure.append(data_forecast['list'][x]['main']['pressure'])
    	sea_level.append(data_forecast['list'][x]['main']['sea_level'])
    	grnd_level.append(data_forecast['list'][x]['main']['grnd_level'])
    	humidity.append(data_forecast['list'][x]['main']['humidity'])
    	description.append(data_forecast['list'][x]['weather'][0]['description'])
    	icon.append(data_forecast['list'][x]['weather'][0]['icon'])
    	speed.append(data_forecast['list'][x]['wind']['speed'])
    	deg.append(data_forecast['list'][x]['wind']['deg'])

    for x in range(36,40):
    	#rain.append(data_forecast['list'][35]['rain']['3h'])
    	x

    weather_forecast = {
    'date' : date,
    'temp' : temp,
    'temp_min' : temp_min,
    'temp_max' : temp_max,
    'pressure' : pressure,
    'sea_level' : sea_level,
    'grnd_level' : grnd_level,
    'humidity' : humidity,
    'description' : description,
    'icon' : icon,
    'speed' : speed,
    'deg' : deg,
    'rain' : rain,
    }
    accuracy = accuracyCalculation(weather_forecast, custom_output)

    #addCurrentWeatherFromJSON(weather_current)

    """
    city = r['city']['name']
    temp = r['list'][6]['dt_txt']

    print(r['list'][39]['main']['temp'])
    """
    return render(request, "html/home.html", {"weather_current" :  weather_current, "weather_forecast" : weather_forecast, "custom_output" : custom_output, "accuracy" : accuracy})
