import math
import numpy as np
from scipy import signal
from scipy.interpolate import interp1d
import pandas as pd
import matplotlib.pyplot as plt


def load_pulse(sample_rate, pulse_filename):
    pulse_data = pd.read_csv(pulse_filename)
    
    start_time =20
    end_time =50
    
    usePluse = pulse_data[( start_time<pulse_data['Time'])]
    usePluse = usePluse[(usePluse['Time'] <end_time+1)]
    
    ave =usePluse['HR'].mean()
    
    print(ave)
   
    

def main():
   
    INPUT_DIR ='/Volumes/Extreme SSD/pulse_data/'
    # OUTPUT_DIR ='/Users/masayakinefuchi/imageSensing/RGB_pulse/_plotCorrectPulse/result/'   
    
    subject = "yamasaki2-close-HR"
    sample_rate = 256
    pulse_filename = INPUT_DIR + subject + ".txt"
    load_pulse(sample_rate,pulse_filename)


if __name__ == "__main__":
    main()
