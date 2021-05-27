# Copyright 2021 The KubeEdge Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder,OneHotEncoder

data = pd.read_csv('ashrae_db2.01.csv')
data.drop(['Database','Publication (Citation)'],axis=1,inplace=True)

labelencoder = LabelEncoder()
Koppen_climateclassifications = labelencoder.fit_transform(data['Koppen climate classification'])
Climates = labelencoder.fit_transform(data['Climate'])
Countrys = labelencoder.fit_transform(data['Country'])

data['Koppen climate classification'] = Koppen_climateclassifications
data['Climate'] = Climates
data['Country'] = Countrys


city_class_dict = {"Tokyo":1, "Texas":2, 'Berkeley':3, 'Chennai':4, 'Hyderabad':5, 'Ilam':6,
'San Francisco':7, 'Alameda':8, 'Philadelphia':9, 'Guangzhou':10,
'Changsha':11, 'Yueyang':12, 'Harbin':13, 'Beijing':14, 'Chaozhou':15, 'Nanyang':16,
'Makati':17, 'Sydney':18, 'Jaipur':19, 'Kota Kinabalu':20, 'Kuala Lumpur':21,
'Beverly Hills':22, 'Putra Jaya':23, 'Kinarut':24, 'Kuching':25, 'Bedong':26,
'Bratislava':27, 'Elsinore':28, 'Gabes':29, 'Gafsa':30, 'El Kef':31, 'Sfax':32,
'Tunis':33, 'Midland':34, 'London':35, 'Lyon':36, 'Gothenburg':37, 'Malmo':38,
'Porto':39, 'Halmstad':40, 'Athens':41, 'Lisbon':42, 'Florianopolis':43,
'BrasÌ_lia':44, 'Recife':45, 'Maceio':46, 'Seoul':47, 'Tsukuba':48, 'Lodi':49,
'Varese':50, 'Imola':51, 'Shanghai':52, 'Liege':53, 'Mexicali':54, 'Hermosillo':55,
'Colima':56, 'Culiacan ':57, 'MÌ©rida':58, 'Tezpur':59, 'Imphal':60, 'Shilong':61,
'Ahmedabad':62, 'Bangalore':63, 'Delhi':64, 'Shimla':65, 'Bandar Abbas':66,
'Karlsruhe':67, 'Bauchi':68, 'Stuttgart':69, 'Hampshire':70, 'Wollongong':71,
'Goulburn':72, 'Singapore':73, 'Cardiff':74, 'Bangkok':75, 'Jakarta':76,
'Montreal':77, 'Brisbane':78, 'Darwin':79, 'Melbourne':80, 'Ottawa':81, 'Karachi':82,
'Multan':83, 'Peshawar':84, 'Quetta':85, 'Saidu Sharif':86, 'Oxford':87,
'San Ramon':88, 'Palo Alto':89, 'Walnut Creek':90, 'Townsville':91,
'Liverpool':92, 'St Helens':93, 'Chester':94, 'Grand Rapids':95, 'Auburn':96,
'Kalgoorlie':97, 'Honolulu':98}
Building_type_class_dict = {'Classroom':1, 'Office':2, 'Others':3, 'Multifamily housing':4,'Senior center':5}
Cooling_startegy_building_level_class_dict = {'Air Conditioned':1, 'Naturally Ventilated':2, 'Mixed Mode':3,'Mechanically Ventilated':4}
Cooling_startegy_operation_mode_for_MM_buildings_class_dict = {'Air Conditioned':1, 'Naturally Ventilated':2, 'Unknown':3}
Heating_strategy_building_level_class_dict = {'Mechanical Heating':1}
Sex_class_dict = {'Female':2, 'Male':1}
Thermal_preference_class_dict = {'warmer':1, 'no change':2, 'cooler':3}
Air_movement_preference_class_dict = {'no change':1, 'more':2, 'less':3}
Humidity_preference_class_dict = {'drier':1, 'no change':2, 'more humid':3}


data['City'] = data['City'].map(city_class_dict)
data['Building type'] = data['Building type'].map(Building_type_class_dict)
data['Cooling startegy_building level'] = data['Cooling startegy_building level'].map(Cooling_startegy_building_level_class_dict)
data['Cooling startegy_operation mode for MM buildings'] = data['Cooling startegy_operation mode for MM buildings'].map(Cooling_startegy_operation_mode_for_MM_buildings_class_dict)
data['Heating strategy_building level'] = data['Heating strategy_building level'].map(Heating_strategy_building_level_class_dict)
data['Sex'] = data['Sex'].map(Sex_class_dict)
data['Thermal preference'] = data['Thermal preference'].map(Thermal_preference_class_dict)
data['Air movement preference'] = data['Air movement preference'].map(Air_movement_preference_class_dict)
data['Humidity preference'] = data['Humidity preference'].map(Humidity_preference_class_dict)

data['Year'].fillna(2002,inplace=True)
data['City'].fillna(0,inplace=True)
data['Building type'].fillna(0,inplace=True)
data['Cooling startegy_building level'].fillna(1,inplace=True)
data['Cooling startegy_operation mode for MM buildings'].fillna(1,inplace=True)
data['Heating strategy_building level'].fillna(0,inplace=True)
data['Age'].fillna(13.189256388091055,inplace=True)
data['Sex'].fillna(0,inplace=True)
data['Thermal sensation'].fillna(0.16300883968656754,inplace=True)
data['Thermal sensation acceptability'].fillna(1,inplace=True)
data['Thermal preference'].fillna(0,inplace=True)
data['Air movement acceptability'].fillna(1,inplace=True)
data['Air movement preference'].fillna(0,inplace=True)
data['PMV'].fillna(1.0862311565319103,inplace=True)
data['PPD'].fillna(20.962094284773425,inplace=True)
data['SET'].fillna(25.769629,inplace=True)
data['Clo'].fillna(0.675876,inplace=True)
data['Met'].fillna(1.206626,inplace=True)
data['activity_10'].fillna(1.194218,inplace=True)
data['activity_20'].fillna(1.257274,inplace=True)
data['activity_30'].fillna(1.264003,inplace=True)
data['activity_60'].fillna(1.319214,inplace=True)
data['Air temperature (C)'].fillna(24.496358,inplace=True)
data['Air temperature (F)'].fillna(76.090540,inplace=True)
data['Ta_h (C)'].fillna(24.569258,inplace=True)
data['Ta_h (F)'].fillna(76.223719,inplace=True)
data['Ta_m (C)'].fillna(24.220964,inplace=True)
data['Ta_l (C)'].fillna(23.450124,inplace=True)
data['Ta_l (F)'].fillna(74.207647,inplace=True)
data['Operative temperature (C)'].fillna(24.504233,inplace=True)
data['Operative temperature (F)'].fillna(76.105627,inplace=True)
data['Radiant temperature (C)'].fillna(24.602735,inplace=True)
data['Radiant temperature (F)'].fillna(76.283592,inplace=True)
data['Globe temperature (C)'].fillna(24.621170,inplace=True)
data['Globe temperature (F)'].fillna(76.316978,inplace=True)
data['Tg_h (C)'].fillna(24.796730,inplace=True)
data['Tg_h (F)'].fillna(76.631297,inplace=True)
data['Tg_m (C)'].fillna(24.375786,inplace=True)
data['Tg_m (F)'].fillna(75.874689,inplace=True)
data['Tg_l (C)'].fillna(22.970135,inplace=True)
data['Tg_l (F)'].fillna(73.341419,inplace=True)
data['Relative humidity (%)'].fillna(47.548293,inplace=True)
data['Humidity preference'].fillna(0,inplace=True)
data['Humidity sensation'].fillna(11.470175,inplace=True)
data['Air velocity (m/s)'].fillna(0.847680,inplace=True)
data['Air velocity (fpm)'].fillna(34.932351,inplace=True)
data['Outdoor monthly air temperature (C)'].fillna(17.446746,inplace=True)
data['Outdoor monthly air temperature (F)'].fillna(63.383538,inplace=True)

data['Thermal comfort'].fillna(0,inplace=True)
data.replace([5.0, 6, 4.0, 1, 3, 2, 4.5, 5.2, 2.5, 2.7, 2.2, 2.3, 2.8, 4.3, 2.4,
       4.2, 3.5, 0, 1.8, 'Na', '2.6', '5.1', '1.7', '3.4', '0.9', '4.3',
       '6', '5', '4', '2', '3', '1', 1.3, 1.5, ' '],[5.0, 6, 4.0, 1, 3, 2, 4.5, 5.2, 2.5, 2.7, 2.2, 2.3, 2.8, 4.3, 2.4,
       4.2, 3.5, 0, 1.8, 0, '2.6', '5.1', '1.7', '3.4', '0.9', '4.3',
       '6', '5', '4', '2', '3', '1', 1.3, 1.5, 0],inplace=True)
data['Thermal comfort'] = pd.to_numeric(data['Thermal comfort'], errors='coerce')

data.drop(columns=['Subject«s height (cm)','Subject«s weight (kg)','Blind (curtain)','Fan','Window','Door','Heater'],axis=1,inplace=True)
data.drop(columns=["Velocity_h (m/s)", "Velocity_h (fpm)",	"Velocity_m (m/s)",	"Velocity_m (fpm)",	"Velocity_l (m/s)",	"Velocity_l (fpm)"], axis=1, inplace=True)
data.drop(['Data contributor','Ta_m (F)'],axis=1,inplace=True)

data.replace(['Summer', 'Autumn', 'Winter', 'Spring', 'nan'], [1, 2, 3,4,5], inplace=True)
data['Season'].fillna(2,inplace=True)

data.to_csv("./new.csv")
