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
#matplotlib.style.use('seaborn-whitegrid')
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
"Mass of avionics components in (kg)"

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
        self.R = 287 # m^2/(s^2 K)
        self.T_0 = 15 # Celsius
        self.lambd = -6.5E-3
        self.T_0k = self.T_0 + 273.15 # Kelvin
        "Define altitude in meters"
        self.z = z
        "ISA functions"
        def Temperature(z):
            return self.T_0k - self.z*(1/1000)*6.5
        self.Tem_ISA = Temperature(z)
        def Pressure(z):
            return (self.pressure_0*(1 + ((self.lambd*self.z)/self.T_0k))**((-self.gravity/(self.R*self.lambd))))
        self.Press_ISA = Pressure(z)
        def Density(z):
            return (self.density_0*(1 + ((self.lambd*self.z)/self.T_0k))**((-self.gravity/(self.R*self.lambd))-1))
        self.Den_ISA = Density(z)
        "Fixed Value"
        self.T_ISA = self.T_0k - z*(1/1000)*6.5
        self.P_ISA = self.pressure_0*(1 + ((self.lambd*z)/self.T_0k))**((-self.gravity/(self.R*self.lambd)))
        self.D_ISA = self.density_0*(1 + ((self.lambd*z)/self.T_0k))**((-self.gravity/(self.R*self.lambd))-1)        

class conversion:
    def __init__(self,c):
        "Velocity"
        self.ms_kt = c*1.94384
        self.ms_fts = c*3.2808398950131
        self.ms_mph = c*2.23694
        self.ms_kmh = c*3.6
        self.ms_fpm = c*196.85039
        "Density"
        self.kgm3_slugft3 = c*0.0019403203259304
        self.kgm3_lbft3 = c*0.062428
        self.kgm3_lbin3 = c*0.0000361273
        "Distance"
        self.m_ft = c*3.28084
        self.m_in = c*39.3701
        "Mass"
        self.kg_lb = c*2.20462
        self.kg_slug = (c*2.20462)*0.031056
        self.kg_ton = c*0.001
        "Force"
        self.kgf_N = c*9.80665
        self.kgf_lbf = c*2.20462
        self.N_lbf = c*0.224809
        "Acceleration"
        self.ms2_fts2 = c*3.28084
        "Area"
        self.m2_ft2 = c*10.7639
        self.m2_in2 = c*1550
        "Temperature"
        self.C_K = c+273.15
        self.C_F = 1.8*c+32
        self.C_R = 1.8*(c+273.15)
        "Load wing"
        self.Nm2_lbft2 = c*0.02088547
        "Power"
        self.W_hp = c*0.001340
        self.lbW_lbhp = c/0.00134102
        "Pressure"
        self.Pa_PSI = c**0.0001450377

class Matching_Plot:
    def __init__(self,masses,battery_data, engines,flight_performance, aerodynamic_param, ehekatl_param,reference_area_range):
        "Matching Plot Design Points"
        self.Wing_loading_JET = Matching_plot_point_designed.get('Wing Loading (jet)')
        self.Thrust_Weight_Ratio = Matching_plot_point_designed.get('Thrust-Weight-ratio (jet)')
        self.Wing_loading_PROP = Matching_plot_point_designed.get('Wing Loading (prop)')
        self.Power_loading_PROP = Matching_plot_point_designed.get('Power Loading (prop)')
        "Gravity"
        self.g = 9.81
        "Ehekatl param"
        AR = ehekatl_param.get('AR')
        S = ehekatl_param.get('Area (m^2)')
        "Oswald wing efficiency factor"
        self.e = 0.9
        "Induced Drag factor"
        self.K = 1/ (np.pi*self.e*AR)
        "Flight performance"
        Range = flight_performance.get('Range (m)')
        Cruise_speed = flight_performance.get('Cruise Velocity (m/s)')
        Max_speed = flight_performance.get('Max Velocity (m/s)')
        z = flight_performance.get('Alttitude (m)')
        Stall_speed = flight_performance.get('Stall Speed (m/s)')
        TO_speed = flight_performance.get('takeoff run speed (m/s)')
        ceiling = flight_performance.get('Absolute ceiling (m)')
        ROC = flight_performance.get('ROC (m/s)')
        "Aerodynamics"
        CL = aerodynamic_param.get('CL')
        CD = aerodynamic_param.get('CD')
        CD0 = aerodynamic_param.get('CD0')
        CLmax = aerodynamic_param.get('CLmax')
        CDmax = aerodynamic_param.get('CDmax')
        self.CLCD = CL/CD
        self.R = 287
        "Engines"
        eng_eff = engines.get('efficiency')
        eng_n = engines.get('number of engines')
        "--------------Avionics, Payload and Energy Masses------------------"
        self.masses = masses
        "Avionics"
        av_m1 = masses.get('Controller Kit')
        av_m2 = masses.get('Servo_m')*masses.get('Servo_n')
        av_m3 = masses.get('ESC')*masses.get('ESC_n')
        av_m4 = masses.get('Pitot_m')
        av_m5 = masses.get('AIProcessor_m')
        av_m6 = masses.get('AICamera_m')
        av_m7 = masses.get('motor_m')
        self.av_tm = av_m1 + av_m2 + av_m3 + av_m4 
        "Payload"
        self.pl_m = masses.get('Payload_m') + av_m5 + av_m6
        "Energy"
        self.b_m = masses.get('Batery_m')*masses.get('Batery_n')
        "Avionics Weight"
        self.av_w = self.av_tm*self.g
        self.pl_w = self.pl_m*self.g
        self.b_w = self.b_m*self.g
        "----------------Battery Analysis-----------------------------------"
        b_mAh = battery_data.get('Battery Capacity (mAh)')
        b_V = battery_data.get('Battery Voltage (V)')
        b_ec = battery_data.get('Time energy consumption (h)')
        battery_output_power = (masses.get('Batery_n')*b_mAh*b_V)/1000 
        self.Battery_energy_density = (battery_output_power*b_ec)/(self.b_m) 
        WB_W_TO = 1.05*((self.g/(eng_eff*self.Battery_energy_density*3600))*(Range/self.CLCD))
        "-------------Empty Weight Calculus---------------------------------"
        a = -4.6e-5 #-4.6e-5
        b = 0.78  #0.68
        bg = (1-WB_W_TO-b)
        c = -(self.pl_m + self.av_tm)
        fg = (bg**2)-(4*a*c)
        if fg<0:
          print('no existe solucion')
        else:
            x1=(-bg+math.sqrt(fg))/(2*a)
            x2=(-bg-math.sqrt(fg))/(2*a)
            if x1 > x2:
                self.MTO = x2
            elif x2 > x1:
                self.MTO = x1
        "---------Maximum Take-Off Weight Estimation-----------------------"
        self.MTOW = 9 #self.MTO
        omega = (ISA(z).D_ISA) / (ISA(0).D_ISA)
        "--------------------WING AND ENGINE SIZING-------------------------"
        cruise_time = Range/Cruise_speed
        Drag = 0.5*ISA(z).D_ISA*(Cruise_speed**2)*S*CD
        Lift = 0.5*ISA(z).D_ISA*(Cruise_speed**2)*S*CL
        Dragmax = 0.5*ISA(z).D_ISA*(Max_speed**2)*S*CDmax
        Liftmax = 0.5*ISA(z).D_ISA*(Max_speed**2)*S*CLmax
        "---Wing Loading & Power loading ---"
        WS = np.linspace(10,100,linspace_n)
        "---Stall Speed---"
        Vs = (1/2)*ISA(z).D_ISA*(Stall_speed**2)*CLmax
        Vsr = np.linspace(Vs, Vs,linspace_n)
        Vsrr = np.linspace(0, 0,linspace_n)
        "---Take-Off Run Parameters---"
        CL_flapTO = 0
        self.CLTO = CL + CL_flapTO
        CD_LG = 0
        self.T_SLmax = (0.5*ISA(0).D_ISA*(Max_speed**2)*S*CDmax)*eng_n
        "---Velocity of sound--------"
        self.sound_speed = np.sqrt(1.4*self.R*(ISA(z).T_ISA))
        Mach = Cruise_speed/self.sound_speed
        Machmax = Max_speed/self.sound_speed
        "-------Zero Lift-Drag Coefficient (CD0)---------------------------"
        self.CD0C = (2*self.T_SLmax-((4*self.K*conversion(self.MTOW).kgf_N)/((ISA(z).D_ISA)*omega*(Max_speed**2)*S)))/(ISA(0).D_ISA*(Max_speed**2)*S)
        "---Maximum Speed---"
        def WPSL_Vmax_Jet(WS):
            return (((ISA(0).D_ISA)*(Max_speed**2)*self.CD0C)*(1/(2*WS)))+(((2*self.K)/(omega*(ISA(z).D_ISA)*(Max_speed**2)))*WS)
        WPSL_Vmax_Jet = WPSL_Vmax_Jet(WS)
        def WPSL_Vmax_Prop(WS):
            return (eng_eff)/(((0.5*(ISA(0).D_ISA))*(Max_speed**3)*self.CD0C*(1/(WS)))+(((2*self.K)/(omega*(ISA(z).D_ISA)*Max_speed))*WS))
        WPSL_Vmax__Prop = WPSL_Vmax_Prop(WS)
        "---Take-off Run---"
        VTO = TO_speed
        Mu = 0.04
        CD_G = 0.025 #6930
        CL_R = CLmax/(1.1**2)
        self.STO = flight_performance.get('Takeoff run (m)')
        def TakeOffRun_Jet(WS):
            return ((Mu - (Mu+(CD_G/CL_R))*(np.exp(0.6*(ISA(z).D_ISA)*self.g*CD_G*self.STO*(1/WS))))/(1-(np.exp(0.6*(ISA(z).D_ISA)*self.g*CD_G*self.STO*(1/WS)))))
        TakeOffRun_Jet =  TakeOffRun_Jet(WS)
        def TakeOffRun_Prop(WS):
            return ((1 - np.exp(0.6*ISA(z).D_ISA*self.g*CD_G*self.STO*(1/WS)))/(Mu-(Mu+(CD_G/CL_R)*(np.exp(0.6*ISA(z).D_ISA*self.g*CD_G*self.STO*(1/WS))))))*(eng_eff/VTO)
        TakeOffRun_Prop = TakeOffRun_Prop(WS)
        "---Rate of Climb---"
        ROC_max = flight_performance.get('ROC (m/s)')
        def ROC_Jet(WS):
            return ((ROC_max/(np.sqrt((2/(ISA(0).D_ISA)*(np.sqrt((self.CD0C)/self.K))))*WS)))+(1/(Lift/Drag))
        ROC_Jet =  ROC_Jet(WS)  
        def ROC_Prop(WS):
            return 1 / ((ROC_max/eng_eff)+(np.sqrt((2/(ISA(0).D_ISA*np.sqrt((3*self.CD0C)/self.K)*WS)))*(1.155/((Lift/Drag)*eng_eff))))
        ROC_Prop = ROC_Prop(WS)
        "---Ceiling---"
        def Ceiling_Jet(WS,ROCC):
            #return ((ROCC/((ISA(z+ceiling).D_ISA/ISA(0).D_ISA)*np.sqrt((2/(ISA(z).D_ISA)*(np.sqrt((self.CD0C)/self.K))))*WS)))+(1/(ISA(z+ceiling).D_ISA) / ((ISA(0).D_ISA)*(Lift/Drag)))
            return ((ROCC/((ISA(z+ceiling).D_ISA/ISA(0).D_ISA)*np.sqrt((2/(ISA(z).D_ISA)*(np.sqrt((self.CD0C)/self.K)))*WS))))+(1/(ISA(z+ceiling).D_ISA) / ((ISA(z+ceiling).D_ISA/ISA(0).D_ISA)*(Liftmax/Dragmax)))
        Ceiling_Jet = Ceiling_Jet(WS, ROC)
        def Ceiling_Prop(WS,ROCC):
            return (ISA(z+ceiling).D_ISA/ISA(0).D_ISA) / ((ROCC/eng_eff)+(np.sqrt((2/(ISA(z + ceiling).D_ISA*np.sqrt((3*self.CD0C)/self.K)*WS)))*(1.155/((Lift/Drag)*eng_eff))))
        Ceiling_Prop = Ceiling_Prop(WS,ROC)
        "Matching plot point designed"
        "---RESULTS---"
        self.WSd_prop = Matching_plot_point_designed.get('Wing Loading (prop)')  #45 #45
        self.WSd_jet = Matching_plot_point_designed.get('Wing Loading (jet)') #45  #55
        self.WPd = Matching_plot_point_designed.get('Power Loading (prop)') #0.022 #0.06 
        self.WTd = Matching_plot_point_designed.get('Thrust-Weight-ratio (jet)') #1.35 #0.8
        Sd_prop = (self.MTOW*self.g)/self.WSd_prop
        Sd_jet = (self.MTOW*self.g)/self.WSd_jet
        Pd = (self.MTOW*self.g)/self.WPd
        Td = (self.MTOW*self.g*self.WTd)
        "-----------------------MTOW - Data Visualization-------------------"
        print('--- MTOW Estimation: '.upper(),"{0:.2f}".format(self.MTOW),"(kg) |","{0:.2f}".format(conversion(self.MTOW).kgf_N),"(N) |","{0:.2f}".format(conversion(self.MTOW).kg_lb),"(lb) ---")
        print('Avionic Mass:',"{0:.2f}".format(self.av_tm),"(kg) |","{0:.2f}".format(conversion(self.av_tm).kgf_N),"(N) |","{0:.2f}".format(conversion(self.av_tm).kg_lb),"(lb)")
        print('Battery Mass:',"{0:.2f}".format(self.b_m),"(kg) |","{0:.2f}".format(conversion(self.b_m).kgf_N),"(N) |","{0:.2f}".format(conversion(self.b_m).kg_lb),"(lb)")
        print('Payload Mass:',"{0:.2f}".format(self.pl_m),"(kg) |","{0:.2f}".format(conversion(self.pl_m).kgf_N),"(N) |","{0:.2f}".format(conversion(self.pl_m).kg_lb),"(lb)")
        self.empty_w = self.MTOW-(self.av_tm+self.b_m+self.pl_m)
        print('Empty Weight Mass:',"{0:.2f}".format(self.empty_w),"(kg) |","{0:.2f}".format(conversion(self.empty_w).kgf_N),"(N) |","{0:.2f}".format(conversion(self.empty_w).kg_lb),"(lb) ---")
        print('---BATERRY ANALYSIS---')
        print("energy-density:","{0:.2f}".format(self.Battery_energy_density),"(Wh/kg)")
        print("Battery Output Power:", "{0:.2f}".format(battery_output_power),"Wh")
        print("WB/WTO:","{0:.4f}".format(WB_W_TO))
        print('---AERODYNAMIC FORCES---')
        print("Lift:","{0:.2f}".format(Lift),"(N) |","{0:.2f}".format(conversion(Lift).N_lbf),"(lbf)")
        print("Drag:","{0:.2f}".format(Drag),"(N) |","{0:.2f}".format(conversion(Drag).N_lbf),"(lbf)")
        print("L/D:","{0:.2f}".format(Lift/Drag))
        print("Liftmax:","{0:.2f}".format(Liftmax),"(N) |","{0:.2f}".format(conversion(Liftmax).N_lbf),"(lbf)")
        print("Dragmax:","{0:.2f}".format(Dragmax),"(N) |","{0:.2f}".format(conversion(Dragmax).N_lbf),"(lbf)")
        print("L/Dmax:","{0:.2f}".format(Liftmax/Dragmax))
        print("Mach Cruise","{0:.3f}".format(Mach),"|","Mach Max","{0:.3f}".format(Machmax))
        print("Induced drag factor","{0:.3f}".format( self.K))
        print("zero-lift drag coefficient","{0:.4f}".format( self.CD0C) )
        print('----Matching Plot Parameters---')
        print("Stall Speed","{0:.2f}".format(Vs), "(N/m2) |","{0:.2f}".format(conversion(Vs).Nm2_lbft2),"(lbf/ft2)" )
        print("Thrust in Sea Level", "{0:.2f}".format(self.T_SLmax), "(N)")
        print("---MATCHING PLOT RESULTS---")
        print('Wing Size')
        print("Prop  S:","{0:.3f}".format(Sd_prop),"m2 | ","Jet S:","{0:.3f}".format(Sd_jet),"m2")
        print('Power Size')
        print( "Prop  P:","{0:.2f}".format(Pd),"W | " "Jet T:","{0:.2f}".format(Td),"N")
        "---------------------------------------------------------------------"
        "Figures"
        limit_upper = 4
        limit_lower = 0
        
        "Matching Plot Jet"
        fig, MP = plt.subplots(figsize=(9,5))
        MP.plot(Vsr,np.linspace(limit_lower, limit_upper,linspace_n),'r-',linewidth=1,label='Vstall')
        MP.fill_betweenx(np.linspace(limit_lower, limit_upper,linspace_n),Vsrr, Vsr, color='red', alpha=0.1)
        MP.plot(WS,WPSL_Vmax_Jet,'b-',linewidth=1,label='Vmax')
        MP.fill_between(WS,WPSL_Vmax_Jet,3, color='blue', alpha=0.1)
        MP.plot(WS,TakeOffRun_Jet,'g-',linewidth=1,label='TakeOffRun')
        MP.fill_between(WS,TakeOffRun_Jet,3, color='green', alpha=0.1)
        MP.plot(WS,ROC_Jet,'y-',linewidth=1,label='ROC')
        MP.fill_between(WS,ROC_Jet,3, color='yellow', alpha=0.1)
        MP.plot(WS,Ceiling_Jet,'k-',linewidth=1,label='Ceiling')
        MP.fill_between(WS,ROC_Jet,3, color='k', alpha=0.1)
        MP.scatter(self.Wing_loading_JET, self.Thrust_Weight_Ratio, color="red", label='Design Point')
        MP.set_title('EHEKATL BWB Matching Plot Jet')   
        MP.set_ylabel('Thrust to Weight ratio')
        MP.set_xlabel('W/S (N/m^2)')
        MP.legend(loc = "best")
        
        "Matching Plot Jet BS"
        fig, MPB = plt.subplots(figsize=(9,5))
        MPB.plot(conversion(Vsr).Nm2_lbft2,np.linspace(limit_lower, limit_upper,linspace_n),'r-',linewidth=1,label='Vstall')
        MPB.fill_betweenx(np.linspace(limit_lower, limit_upper,linspace_n),conversion(Vsrr).Nm2_lbft2, conversion(Vsr).Nm2_lbft2, color='red', alpha=0.1)
        MPB.plot( conversion(WS).Nm2_lbft2,WPSL_Vmax_Jet,'b-',linewidth=1,label='Vmax')
        MPB.fill_between( conversion(WS).Nm2_lbft2,WPSL_Vmax_Jet,3, color='blue', alpha=0.1)
        MPB.plot( conversion(WS).Nm2_lbft2,TakeOffRun_Jet,'g-',linewidth=1,label='TakeOffRun')
        MPB.fill_between( conversion(WS).Nm2_lbft2,TakeOffRun_Jet,3, color='green', alpha=0.1)
        MPB.plot(conversion(WS).Nm2_lbft2,ROC_Jet,'y-',linewidth=1,label='ROC')
        MPB.fill_between(conversion(WS).Nm2_lbft2,ROC_Jet,3, color='yellow', alpha=0.1)
        MPB.plot(conversion(WS).Nm2_lbft2,Ceiling_Jet,'k-',linewidth=1,label='Ceiling')
        MPB.fill_between(conversion(WS).Nm2_lbft2,ROC_Jet,3, color='k', alpha=0.1)
        MPB.scatter(conversion(self.Wing_loading_JET).Nm2_lbft2, self.Thrust_Weight_Ratio, color="red", label='Design Point')
        MPB.set_title('EHEKATL BWB Matching Plot Jet British Units')   
        MPB.set_ylabel('Thrust to Weight ratio')
        MPB.set_xlabel('W/S (lb/ft^2)')
        MPB.legend(loc = "best")
        
        limit_upper = 0.5
        limit_lower = 0
        "Matching Plot Prop"
        fig, MPP = plt.subplots(figsize=(9,5))
        MPP.plot(Vsr,np.linspace(limit_lower, limit_upper,linspace_n),'r-',linewidth=1,label='Vstall')
        MPP.fill_betweenx(np.linspace(limit_lower, limit_upper,linspace_n),Vsrr, Vsr, color='red', alpha=0.1)
        MPP.plot(WS,WPSL_Vmax__Prop,'b-',linewidth=1,label='Vmax')
        MPP.fill_between(WS,WPSL_Vmax__Prop, color='blue', alpha=0.1)
        MPP.plot(WS,TakeOffRun_Prop,'g-',linewidth=1,label='TakeOffRun')
        MPP.fill_between(WS,TakeOffRun_Prop, color='green', alpha=0.1)
        MPP.plot(WS,ROC_Prop,'y-',linewidth=1,label='ROC')
        MPP.fill_between(WS,ROC_Prop, color='yellow', alpha=0.1)
        MPP.plot(WS,Ceiling_Prop,'k-',linewidth=1,label='Ceiling')
        MPP.fill_between(WS,Ceiling_Prop, color='k', alpha=0.1)
        MPP.scatter(self.Wing_loading_PROP, self.Power_loading_PROP, color="red", label='Design Point')
        MPP.set_title('EHEKATL BWB Matching Plot Prop')   
        MPP.set_ylabel('W/P (N/W)')
        MPP.set_xlabel('W/S (N/m^2)') 
        MPP.legend(loc = "best") 
        
        
        limit_upper = 80
        limit_lower = 0
        "Matching Plot Prop B"
        WPSL_Vmax__Prop_lb = conversion(WPSL_Vmax__Prop).N_lbf
        WPSL_Vmax__Prop_hp = conversion(WPSL_Vmax__Prop_lb).lbW_lbhp
        WPSL_TakeOffRun__Prop_lb = conversion(TakeOffRun_Prop).N_lbf
        WPSL_TakeOffRun__Prop_hp = conversion(WPSL_TakeOffRun__Prop_lb).lbW_lbhp
        WPSL_ROC__Prop_lb = conversion(ROC_Prop).N_lbf
        WPSL_ROC_hp = conversion(WPSL_ROC__Prop_lb).lbW_lbhp
        WPSL_Ceiling__Prop_lb = conversion(Ceiling_Prop).N_lbf
        WPSL_Ceiling_hp = conversion(WPSL_Ceiling__Prop_lb).lbW_lbhp
        WPSL_PWL__Prop_lb = conversion(self.Power_loading_PROP).N_lbf
        WPSL_PWL__Prop_hp =  conversion(WPSL_PWL__Prop_lb).lbW_lbhp
        fig, MPPB = plt.subplots(figsize=(9,5))
        MPPB.plot(conversion(Vsr).Nm2_lbft2,np.linspace(limit_lower, limit_upper,linspace_n),'r-',linewidth=1,label='Vstall')
        MPPB.fill_betweenx(np.linspace(limit_lower, limit_upper,linspace_n),conversion(Vsrr).Nm2_lbft2, conversion(Vsr).Nm2_lbft2, color='red', alpha=0.1)
        MPPB.plot(conversion(WS).Nm2_lbft2,WPSL_Vmax__Prop_hp,'b-',linewidth=1,label='Vmax')
        MPPB.fill_between(conversion(WS).Nm2_lbft2,WPSL_Vmax__Prop_hp, color='blue', alpha=0.1)
        MPPB.plot(conversion(WS).Nm2_lbft2,WPSL_TakeOffRun__Prop_hp,'g-',linewidth=1,label='TakeOffRun')
        MPPB.fill_between(conversion(WS).Nm2_lbft2,WPSL_TakeOffRun__Prop_hp, color='green', alpha=0.1)
        MPPB.plot(conversion(WS).Nm2_lbft2,WPSL_ROC_hp,'y-',linewidth=1,label='ROC')
        MPPB.fill_between(conversion(WS).Nm2_lbft2,WPSL_ROC_hp, color='yellow', alpha=0.1)
        MPPB.plot(conversion(WS).Nm2_lbft2,WPSL_Ceiling_hp,'k-',linewidth=1,label='Ceiling')
        MPPB.fill_between(conversion(WS).Nm2_lbft2,WPSL_Ceiling_hp, color='k', alpha=0.1)
        MPPB.scatter(conversion(self.Wing_loading_PROP).Nm2_lbft2 , WPSL_PWL__Prop_hp, color="red", label='Design Point')
        MPPB.set_title('EHEKATL BWB Matching Plot Prop British Units')   
        MPPB.set_ylabel('W/P (lb/hp)')
        MPPB.set_xlabel('W/S (lb/ft^2)') 
        MPPB.legend(loc = "best") 
        
        "Individual Figures"
        "----------------------------------------------------------------------"
        "Stall Velocity"
        # # IS
        # fig, vstall = plt.subplots()
        # vstall.plot(Vsr,np.linspace(0, 3,linspace_n),'r-',linewidth=1,label='Vstall')
        # vstall.fill_betweenx(np.linspace(0, 3,linspace_n),Vsrr, Vsr, color='red', alpha=0.2)
        # vstall.set_title('Velocidad de entrada en pérdida (Razón Empuje-Peso)')   
        # vstall.set_ylabel('T/W')
        # vstall.set_xlabel('W/S (N/m^2)') 
        # vstall.legend(loc = "best") 
        # # BS
        # fig, vstallb = plt.subplots()
        # vstallb.plot( conversion(Vsr).Nm2_lbft2,np.linspace(0, 55,linspace_n),'r-',linewidth=1,label='Vstall')
        # vstallb.fill_betweenx(np.linspace(0, 55,linspace_n),conversion(Vsrr).Nm2_lbft2, conversion(Vsr).Nm2_lbft2, color='red', alpha=0.2)
        # vstallb.set_title('Velocidad de entrada en pérdida (lb/hp)')   
        # vstallb.set_ylabel(' W/P (lb/hp)')
        # vstallb.set_xlabel('W/S (lb/ft^2)') 
        # vstallb.legend(loc = "best") 
        
        # " Max Velocity"
        # # JET IS
        # fig, vmax = plt.subplots()
        # vmax.plot(WS,WPSL_Vmax_Jet,'b-',linewidth=1,label='Vmax')
        # vmax.fill_between(WS,WPSL_Vmax_Jet,3, color='blue', alpha=0.2)
        # vmax.set_title('Velocidad máxima JET (Razón Empuje-Peso N/m^2)')   
        # vmax.set_ylabel('T/W')
        # vmax.set_xlabel('W/S (N/m^2)') 
        # vmax.legend(loc = "best") 
        # # JET BS
        # fig, vmaxb = plt.subplots()
        # vmaxb.plot( conversion(WS).Nm2_lbft2,  WPSL_Vmax_Jet,'b-',linewidth=1,label='Vmax')
        # vmaxb.fill_between( conversion(WS).Nm2_lbft2,WPSL_Vmax_Jet,3, color='blue', alpha=0.2)
        # vmaxb.set_title('Velocidad máxima JET (Razón Empuje-Peso lb/ft^2)')   
        # vmaxb.set_ylabel('T/W')
        # vmaxb.set_xlabel('W/S (lb/ft^2)') 
        # vmaxb.legend(loc = "best") 
        # # PROP IS
        # fig, vmaxp = plt.subplots()
        # vmaxp.plot(WS,WPSL_Vmax__Prop,'b-',linewidth=1,label='Vmax')
        # vmaxp.fill_between(WS,WPSL_Vmax__Prop, color='blue', alpha=0.2)
        # vmaxp.set_title('Velocidad máxima PROP (Razón Peso-Potencia N/m^2)')   
        # vmaxp.set_ylabel('W/P (N/W)')
        # vmaxp.set_xlabel('W/S (N/m^2)') 
        # vmaxp.legend(loc = "best")
        # # PROP BS
        # WPSL_Vmax__Prop_lb = conversion(WPSL_Vmax__Prop).N_lbf
        # WPSL_Vmax__Prop_hp = conversion(WPSL_Vmax__Prop_lb).lbW_lbhp
        # fig, vmaxpb = plt.subplots()
        # vmaxpb.plot(conversion(WS).Nm2_lbft2,WPSL_Vmax__Prop_hp,'b-',linewidth=1,label='Vmax')
        # vmaxpb.fill_between(conversion(WS).Nm2_lbft2,WPSL_Vmax__Prop_hp, color='blue', alpha=0.2)
        # vmaxpb.set_title('Velocidad máxima PROP (Razón Peso-Potencia lb/ft^2)')   
        # vmaxpb.set_ylabel('W/P (lb/hp)')
        # vmaxpb.set_xlabel('W/S (lb/ft^2)') 
        # vmaxpb.legend(loc = "best")
        
        
        # "Take-Off Run"
        # #JET IS
        # fig, to = plt.subplots()
        # to.plot(WS,TakeOffRun_Jet,'g-',linewidth=1,label='TakeOffRun')
        # to.fill_between(WS,TakeOffRun_Jet,3, color='green', alpha=0.2)
        # to.set_title('Take-off Run JET (Razón Empuje-Peso N/m^2)')   
        # to.set_ylabel('T/W')
        # to.set_xlabel('W/S (N/m^2)') 
        # to.legend(loc = "best") 
        # #JET BS
        # fig, tob = plt.subplots()
        # tob.plot(conversion(WS).Nm2_lbft2,TakeOffRun_Jet,'g-',linewidth=1,label='TakeOffRun')
        # tob.fill_between(conversion(WS).Nm2_lbft2,TakeOffRun_Jet,3, color='green', alpha=0.2)
        # tob.set_title('Take-off Run JET (Razón Empuje-Peso lb/ft^2)')   
        # tob.set_ylabel('T/W')
        # tob.set_xlabel('W/S (lb/ft^2)') 
        # tob.legend(loc = "best") 
        # #PROP IS
        # fig, to = plt.subplots()
        # to.plot(WS,TakeOffRun_Prop,'g-',linewidth=1,label='TakeOffRun')
        # to.fill_between(WS,TakeOffRun_Prop, color='green', alpha=0.2)
        # to.set_title('Take-off Run PROP (Razón Empuje-Peso N/m^2)')   
        # to.set_ylabel('W/P (N/W)')
        # to.set_xlabel('W/S (N/m^2)') 
        # to.legend(loc = "best") 
        # #PROP BS
        # WPSL_TakeOffRun__Prop_lb = conversion(TakeOffRun_Prop).N_lbf
        # WPSL_TakeOffRun__Prop_hp = conversion(WPSL_TakeOffRun__Prop_lb).lbW_lbhp
        # fig, to = plt.subplots()
        # to.plot(conversion(WS).Nm2_lbft2,WPSL_TakeOffRun__Prop_hp,'g-',linewidth=1,label='TakeOffRun')
        # to.fill_between(conversion(WS).Nm2_lbft2,WPSL_TakeOffRun__Prop_hp, color='green', alpha=0.2)
        # to.set_title('Take-off Run PROP (Razón Empuje-Peso lb/ft^2)')   
        # to.set_ylabel('W/P (lb/hp)')
        # to.set_xlabel('W/S (lb/ft^2)') 
        # to.legend(loc = "best") 
        
        # "ROC"
        # #JET IS
        # fig, roc = plt.subplots()
        # roc.plot(WS,ROC_Jet,'y-',linewidth=1,label='ROC')
        # roc.fill_between(WS,ROC_Jet,3, color='yellow', alpha=0.2)
        # roc.set_title('Rate of Climb JET (N/m^2)')   
        # roc.set_ylabel('T/W')
        # roc.set_xlabel('W/S (N/m^2)') 
        # roc.legend(loc = "best") 
        # #JET BS
        # fig, rocb = plt.subplots()
        # rocb.plot(conversion(WS).Nm2_lbft2,ROC_Jet,'y-',linewidth=1,label='ROC')
        # rocb.fill_between(conversion(WS).Nm2_lbft2,ROC_Jet,3, color='yellow', alpha=0.2)
        # rocb.set_title('Rate of Climb JET  (lb/ft^2)')   
        # rocb.set_ylabel('T/W')
        # rocb.set_xlabel('W/S (lb/ft^2)') 
        # rocb.legend(loc = "best") 
        # #PROP IS
        # fig, roc = plt.subplots()
        # roc.plot(WS,ROC_Prop,'y-',linewidth=1,label='ROC')
        # roc.fill_between(WS,ROC_Prop, color='yellow', alpha=0.2)
        # roc.set_title('Rate of Climb PROP (N/m^2)')   
        # roc.set_ylabel('W/P (N/W)')
        # roc.set_xlabel('W/S (N/m^2)') 
        # roc.legend(loc = "best") 
        # #PROP BS
        # WPSL_ROC__Prop_lb = conversion(ROC_Prop).N_lbf
        # WPSL_ROC_hp = conversion(WPSL_ROC__Prop_lb).lbW_lbhp
        # fig, rocb = plt.subplots()
        # rocb.plot(conversion(WS).Nm2_lbft2,WPSL_ROC_hp,'y-',linewidth=1,label='ROC')
        # rocb.fill_between(conversion(WS).Nm2_lbft2,WPSL_ROC_hp, color='yellow', alpha=0.2)
        # rocb.set_title('Rate of Climb PROP (lb/ft^2)')   
        # rocb.set_ylabel('W/P (lb/hp)')
        # rocb.set_xlabel('W/S (lb/ft^2)') 
        # rocb.legend(loc = "best") 
        
        # "Ceiling"
        # #JET IS
        # fig, ceil = plt.subplots()
        # ceil.plot(WS,ROC_Jet,'k-',linewidth=1,label='Ceiling')
        # ceil.fill_between(WS,ROC_Jet,3, color='k', alpha=0.2)
        # ceil.set_title('Ceiling JET (N/m^2)')   
        # ceil.set_ylabel('T/W')
        # ceil.set_xlabel('W/S (N/m^2)') 
        # ceil.legend(loc = "best") 
        # #JET BS
        # fig, ceilb = plt.subplots()
        # ceilb.plot(conversion(WS).Nm2_lbft2,ROC_Jet,'k-',linewidth=1,label='Ceiling')
        # ceilb.fill_between(conversion(WS).Nm2_lbft2,ROC_Jet,3, color='k', alpha=0.2)
        # ceilb.set_title('Ceiling JET (lb/ft^2)')   
        # ceilb.set_ylabel('T/W')
        # ceilb.set_xlabel('W/S (N/m^2)') 
        # ceilb.legend(loc = "best") 
        # #PROP IS
        # fig, ceilp = plt.subplots()
        # ceilp.plot(WS,Ceiling_Prop,'k-',linewidth=1,label='Ceiling')
        # ceilp.fill_between(WS,Ceiling_Prop, color='k', alpha=0.2)
        # ceilp.set_title('Ceiling PROP (N/m^2)')   
        # ceilp.set_ylabel('W/P (N/W)')
        # ceilp.set_xlabel('W/S (N/m^2)') 
        # ceilp.legend(loc = "best") 
        # #PROP BS
        # WPSL_Ceiling__Prop_lb = conversion(Ceiling_Prop).N_lbf
        # WPSL_Ceiling_hp = conversion(WPSL_Ceiling__Prop_lb).lbW_lbhp
        # fig, rocb = plt.subplots()
        # rocb.plot(conversion(WS).Nm2_lbft2,WPSL_Ceiling_hp,'k-',linewidth=1,label='Ceiling')
        # rocb.fill_between(conversion(WS).Nm2_lbft2,WPSL_Ceiling_hp, color='k', alpha=0.2)
        # rocb.set_title('Ceiling PROP (lb/ft^2)')   
        # rocb.set_ylabel('W/P (lb/hp)')
        # rocb.set_xlabel('W/S (lb/ft^2)') 
        # rocb.legend(loc = "best") 
       
# =============================================================================
Matching_Plot(masses, battery_data, engines, flight_performance, aerodynamic_param,ehekatl_param, reference_area_range)
# =============================================================================

