# ProyectoGen
# Descripción del codigo
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 15:45:29 2023
@author: Gandhi Alexis Sinhue Contreras Torres
UAV/UAS/Drones Matching Plot Methodology 
"""

import seaborn as sns
import numpy as np
import math as math
import matplotlib.pyplot as plt
import matplotlib
sns.set_style('whitegrid')

# =============================================================================
# Plot Style Available
# =============================================================================
"""
MATPLOTLIB NICE STYLES
seaborn-darkgrid
fivethirtyeight
seaborn-whitegrid
classic
fast
ggplot
seaborn
Solarize_Light2
seaborn-paper
bmh
dark_background
"""

# MATCHING PLOT
# =============================================================================
# Input Parameters
# =============================================================================

# Mass of avionics components in (kg)
masses = {'Controller Kit':0.5,
          'Servo_m':0.015,
          'Servo_n' :14,
          'ESC' :0.125,
          'ESC_n': 2,
          'Pitot_m':0.04,
          'AIProcessor_m':0.25,
          'AICamera_m':0.05,
          'Payload_m':0,
          'Batery_m':0.36,
          'Batery_n':2,
          'motor_m': 0.225}

battery_data = {
    'Battery Capacity (mAh)': 3300,
    'Battery Voltage (V)': 22.2,
    'Time energy consumption (h)':1
}

engines ={'efficiency': 0.81,
          'number of engines':2
}

flight_performance = {'Range (m)': 2010*6,
                      'Stall Speed (m/s)':19,
                      'takeoff run speed (m/s)':19*1.2,
                      'Cruise Velocity (m/s)':27,
                      'Max Velocity (m/s)': 27*1.3,
                      'Alttitude (m)': 2400,
                      'Takeoff run (m)': 30,
                      'ROC (m/s)': 1.5,
                      'Absolute ceiling (m)':120,
                      'Cruise ceiling (m)':90,
}

aerodynamic_param = {'CL': 0.2, #0.22,
                     'CD': 0.013,#0.014,
                     'CD0': 0.025, #0.025,
                     'CLmax': 0.48, #0.3,
                     'CDmax': 0.028, #0.024
}

ehekatl_param = {'AR': 4.6, #3.855,
                 'Area (m^2)':1.3
}

Matching_plot_point_designed = {'Wing Loading (prop)':68,
                                'Wing Loading (jet)': 52,
                                'Power Loading (prop)':0.03,
                                'Thrust-Weight-ratio (jet)':0.91}

linspace_n = 10000
reference_area_range = np.linspace(0.5,6,linspace_n)

# =============================================================================
# Matching Plot - UAV/UAS/Drones
# =============================================================================

class ISA:
    "ISA in Troposphere"
    def __init__(self,z):   
        "fixed parameters in sea level"
        self.pressure_0 = 101325 #Pa
        self.density_0 = 1.225 #kg/m^3
        self.gravity = 9.81 #m/s^2
        self.R = 
        # =============================================================================
# Matching Plot - UAV/UAS/Drones
# =============================================================================
# Esta sección del código define la clase ISA, conversion y Matching_Plot, que se utilizarán para crear la metodología de trazado de gráficos para UAV/UAS/Drones.
class ISA:
    "ISA in Troposphere"
    def __init__(self,z):
        ...


# La clase ISA representa la atmósfera estándar internacional (ISA) en la troposfera.
# Se inicializa con el parámetro z, que representa la altitud en metros. 
# Los métodos dentro de la clase calculan la temperatura, presión y densidad atmosférica en función de la altitud utilizando las fórmulas de la ISA.

class conversion:
    def __init__(self,c):
        ...

# La clase "conversion" se utiliza para realizar conversiones de unidades. 
# Se inicializa con el parámetro c, que representa el valor a convertir. 
# Los métodos dentro de la clase realizan diferentes conversiones de unidades, como velocidad, densidad, distancia, masa, fuerza, aceleración, área, temperatura, carga alar, potencia y presión.

  class Matching_Plot:
    def __init__(self,masses,battery_data, engines,flight_performance, aerodynamic_param, ehekatl_param,reference_area_range):
        ...

  # La clase Matching_Plot se encarga de realizar el trazado de gráficos para UAV/UAS/Drones. 
  # Se inicializa con varios parámetros relacionados con las masas, datos de la batería, motores, rendimiento en vuelo, parámetros aerodinámicos, parámetros de Ehekatl y rango de área de referencia.

  # Los valores de los parámetros se asignan a las variables correspondientes. Luego se realizan cálculos y cálculos intermedios para obtener los datos necesarios para el trazado de gráficos.

  
