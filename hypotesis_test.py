import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
import scipy.stats as st
from ipywidgets import interact, fixed, interact_manual, Dropdown, BoundedFloatText
plt.style.use('seaborn')

# -------------- Fuctions ---------------------------------------------------------------

def draw_statistic(function, axes, t_statistic, color, *degrees_of_freedom):
    x = np.linspace(-5,5,1000)
    y_pdf = function.pdf(x, *degrees_of_freedom)   
    
    axes.plot(x, y_pdf, label='pdf')
    axes.set_title("Distribution")
    axes.set_xlabel("x")
    axes.set_xlim(-6,6)  
    
    idx_statistic = (np.abs(x - t_statistic)).argmin()
    axes.fill_between(x[:idx_statistic],y_pdf[:idx_statistic], color, alpha = 0.25)
    
    y_t_statistic_ppf = function.pdf(t_statistic, *degrees_of_freedom)
    axes.vlines(t_statistic, ymin=0, ymax=y_t_statistic_ppf, color=color)
    
    text2 = "(" + str(round(function.cdf(t_statistic, *degrees_of_freedom),3)) + ", " + str(round(t_statistic,2)) + ")"
    axes.text(2.5, 0.3, text2, style='italic', color = "black",fontsize=15 )

    
def draw_critical(function, axes, alpha, color, *degrees_of_freedom):
    t_critical = function.ppf(alpha, *degrees_of_freedom)
    y_t_critical_pdf = function.pdf(t_critical, *degrees_of_freedom)               
    axes.vlines(t_critical, ymin=0, ymax=y_t_critical_pdf, color=color)
    
    text1 = "(" + str(round(alpha,3)) + ", " + str(round(t_critical,2)) + ")"
    axes.text(3.5*t_critical/abs(t_critical)-1.5, 0.3-abs(t_critical/10), text1, style='italic', color = "red",fontsize=15 )
    
    
def fill_side(side, function, axes, t_statistic, alpha, color, *degrees_of_freedom):    
    x = np.linspace(-5,5,1000)
    y_pdf = function.pdf(x, *degrees_of_freedom)  
    
    idx = (np.abs(x - function.ppf(alpha, *degrees_of_freedom))).argmin()
    idx_statistic = (np.abs(x - t_statistic)).argmin()
    
    y_pdf = function.pdf(x,*degrees_of_freedom)   
    if side == 'left':
        axes.fill_between(x[:idx],y_pdf[:idx], color = "red", alpha = 0.15)
    else:
        axes.fill_between(x[idx:],y_pdf[idx:], color = "red", alpha = 0.15) 

# ------------- T-Student --------------------------------------------------------------

def plot_t_critical_values(t_statistic, alpha=0.05, side="two", degrees_of_freedom=2):
        
    function=st.t
    
    fig, axes = plt.subplots(1,1,figsize=(6, 6))  
    
    draw_statistic(function, axes, t_statistic, "black", degrees_of_freedom)
        
    if side == "two" :
        draw_critical(function, axes, alpha/2, "red", degrees_of_freedom)      
        fill_side('left', function, axes, t_statistic, alpha/2, "red", degrees_of_freedom)
        
        draw_critical(function, axes, 1-(alpha/2), "red", degrees_of_freedom) 
        fill_side('right', function, axes, t_statistic, 1 - alpha/2, "red", degrees_of_freedom) 
         
    elif side == "right":
        draw_critical(function, axes, 1-alpha, "red", degrees_of_freedom)
        fill_side('right', function, axes, t_statistic, 1-alpha, "red", degrees_of_freedom)
        
    else:
        draw_critical(function, axes,alpha, "red", degrees_of_freedom)
        fill_side('left', function, axes, t_statistic, alpha, "red", degrees_of_freedom) 
        
    
    plt.show();
    
    
def t_student():
    '''
    This function plots the t_student hypotesis test with different options.
    '''
    interact(plot_t_critical_values,          
             t_statistic = widgets.BoundedFloatText(
                 min = -10,
                 value=2.8,
                 description='T-statistic:',
                 disabled=False,
                 step=0.05
             ),
             side = widgets.Dropdown(
                 options=['right', 'left', 'two'],
                 value='two',
                 description='Type:'   
             ),
             alpha = widgets.BoundedFloatText(
                 value=0.05,
                 description='Alpha:',
                 disabled=False,
                 step=0.005
             ),
             degrees_of_freedom = widgets.IntSlider(
                 value=10, 
                 min=2, 
                 max=30, 
                 step=1
             ))    
    
# ------------- Normal dist --------------------------------------------------------------  
    
def plot_norm_critical_values(t_statistic, alpha=0.05, side="two"):
        
    function=st.norm
    
    fig, axes = plt.subplots(1,1,figsize=(6, 6))  
    
    draw_statistic(function, axes, t_statistic, "black")
        
    if side == "two" :
        draw_critical(function, axes, alpha/2, "red")      
        fill_side('left', function, axes, t_statistic, alpha/2, "red")
        
        draw_critical(function, axes, 1-(alpha/2), "red") 
        fill_side('right', function, axes, t_statistic, 1 - alpha/2, "red") 
         
    elif side == "right":
        draw_critical(function, axes, 1-alpha, "red")
        fill_side('right', function, axes, t_statistic, 1-alpha, "red")
        
    else:
        draw_critical(function, axes,alpha, "red")
        fill_side('left', function, axes, t_statistic, alpha, "red") 
        
    
    plt.show();
    
    
def normal():
    interact(plot_norm_critical_values,     
             t_statistic = widgets.BoundedFloatText(
                 min = -10,
                 value=2.8,
                 description='Z-statistic:',
                 disabled=False,
                 step=0.05
             ),
             side = widgets.Dropdown(
                 options=['right', 'left', 'two'],
                 value='left',
                 description='Type:'   
             ),
             alpha = widgets.BoundedFloatText(
                 value=0.05,
                 description='Alpha:',
                 disabled=False,
                 step=0.005
             ))