# -*- coding: utf-8 -*-
# @Date    : 2019-02-18 16:44:20
# @Author  : Your Name (you@example.org)
# @Link    : http://example.org
# @Version : $1.0$

import os
import sys
sys.path.append(os.getcwd())
import numpy as np
import pandas as pd


features_bike = [
    'Duration', #int
    'StartHour', # string
    'Bike', #int
    'SubscriptionType', # string
    'ZipCode', # string
    'Max_Temperature_F', #int
    # 'Mean_Temperature_F', #int
    'Min_TemperatureF', #int
    # 'Max_Dew_Point_F', #int
	# 'MeanDew_Point_F', #int
	'Min_Dewpoint_F', #int
	'Max_Humidity', #int
	# 'Mean_Humidity', #int
	# 'Min_Humidity', #int
	'Max_Sea_Level_Pressure_In', #float
	# 'Mean_Sea_Level_Pressure_In', #float
	# 'Min_Sea_Level_Pressure_In', #float
	# 'Max_Visibility_Miles', #int
	# 'Mean_Visibility_Miles', #int
	'Min_Visibility_Miles', #int
	'Max_Wind_Speed_MPH', #int
	# 'Mean_Wind_Speed_MPH', #int
	'Max_Gust_Speed_MPH', #float
	'Precipitation_In', #string
	'Cloud_Cover', #int
	'Events', # string
	'Wind_Dir_Degrees', #int
	'Weekday' #string
]


features_taxi = [
        # 'vendor_id', 
        # 'store_and_fwd_flag', 
        # 'rate_code_id', 
        'passenger_count', 
        'trip_distance',
        # 'fare_amount', 
        # 'extra', 
        # 'mta_tax', 
        # 'tip_amount', 
        # 'tolls_amount',
        # 'improvement_surcharge', 
        # 'total_amount', 
        'payment_type', 
        # 'trip_type',
        # 'pickup_date', 
        'pickup_hour', 
        # 'dropoff_date', 
        # 'dropoff_hour',
        # 'duration', 
        # 'STATION', 
        # 'NAME', 
        # 'DATE', 
        'AWND', 
        'PRCP', 
        'SNOW', 
        'SNWD',
        'TMAX', 
        'TMIN',
        'weekday',
        'month'
        ]


synthetic = [
        'x1',
        'x2',
        'x3',
        'x4',
        'x5',
        'x6',
        'x7',
        'x8',
        'x9',
        'x10',
]

colormnist = [
        'x1',
        'x2',
        'x3',
        'x4',
        'x5',
        'x6',
        'x7',
        'x8',
        'x9',
        'x10',
        'x11',
        'x12',
        'x13',
        'x14',
        'x15',
        'x16',
        # 'x17',
        # 'x18',
        # 'x19',
        # 'x20',
        # 'x21',
        # 'x22',
        # 'x23',
        # 'x24',
        # 'x25',
        # 'x26',
        # 'x27',
        # 'x28',
        # 'x29',
        # 'x30',
        # 'x31',
        # 'x32',
        # 'x33',
        # 'x34',
        # 'x35',
        # 'x36',
        # 'x37',
        # 'x38',
        # 'x39',
        # 'x40',
        # 'x41',
        # 'x42',
        # 'x43',
        # 'x44',
        # 'x45',
        # 'x46',
        # 'x47',
        # 'x48', 
        # 'x49', 
        # 'x50',
]

waterbirds = [
        'x1',
        'x2',
        'x3',
        'x4',
        'x5',
        'x6',
        'x7',
        'x8',
        'x9',
        'x10',
        'x11',
        'x12',
        'x13',
        'x14',
        'x15',
        'x16',
]

hastie = [
        'x1',
        'x2',
        'x3',
        'x4',
        'x5',
        'x6',
        'x7',
        'x8',
        'x9',
        'x10',
]

readmission = [
        'admission_type_id',
        'discharge_disposition_id',
        'time_in_hospital',
        'num_lab_procedures',
        'num_procedures',
        'num_medications',
        'number_outpatient',
        'number_emergency',
        'number_inpatient',
        'number_diagnoses',
]

adult = ['age', 
         'workclass', 
         'fnlwgt', 
         'education', 
         'education_num', 
         'marital_status', 
         'occupation', 
         'relationship',
         'sex', 
         'capital_gain', 
         'capital_loss', 
         'hours_per_week', 
         'native_country',]

churnmodel = ['CreditScore',
                'Gender',
                'Age',
                'Tenure',
                'Balance',
                'NumOfProducts',
                'HasCrCard',
                'IsActiveMember',
                'EstimatedSalary',]

higgs = [
        'lepton_pT',
        'lepton_eta',
        'lepton_phi',
        'missing_energy_magnitude',
        'missing_energy_phi',
        'jet1pt',
        'jet1eta',
        'jet1phi',
        'jet1b_tag',
        'jet2pt',
        'jet2eta',
        'jet2phi',
        'jet2b_tag',
        'jet3pt',
        'jet3eta',
        'jet3phi',
        'jet3b_tag',
        'jet4pt',
        'jet4eta',
        'jet4phi',
        'jet4b_tag',
        'm_jj',
        'm_jjj',
        'm_lv',
        'm_jlv',
        'm_bb',
        'm_wbb',
        'm_wwbb',
]

bankadd = [
        'age',
        'job',
        # 'marital',
        'education',
        'default',
        'housing',
        'loan',
        'contact',
        'month',
        'day_of_week',
        'duration',
        'campaign',
        'pdays',
        'previous',
        'poutcome',
        'emp_var_rate',
        'cons_price_idx',
        'cons_conf_idx',
        'euribor3m',
        'nr_employed',
]

creditcard = [
            'LIMIT_BAL', 
            'SEX', 
            'MARRIAGE', 
            'AGE', 
            'PAY_0', 
            'PAY_2', 
            'PAY_3', 
            'PAY_4', 
            'PAY_5', 
            'PAY_6', 
            'BILL_AMT1', 
            'BILL_AMT2', 
            'BILL_AMT3', 
            'BILL_AMT4', 
            'BILL_AMT5', 
            'BILL_AMT6', 
            'PAY_AMT1', 
            'PAY_AMT2', 
            'PAY_AMT3', 
            'PAY_AMT4', 
            'PAY_AMT5', 
            'PAY_AMT6',
]

icustay = [
        'ANIONGAP', 
        'ALBUMIN', 
        'BICARBONATE', 
        'CREATININE',
        'CHLORIDE', 
        'GLUCOSE', 
        'HEMATOCRIT', 
        'HEMOGLOBIN', 
        'LACTATE',
        'MAGNESIUM', 
        'PHOSPHATE', 
        'PLATELET', 
        'POTASSIUM', 
        'PTT', 
        'INR', 
        'PT',
        'SODIUM', 
        'BUN', 
        'WBC', 
        'admission_location_cat', 
        # 'insurance_cat',
        'language_cat', 
        'religion_cat', 
        'marital_status_cat', 
        'ethnicity_cat',
]

def log(logfile,str_in):
    """ Log a string in a file """
    with open(logfile,'a') as f:
        f.write(str_in+'\n')
    print(str_in)

def load_config(cfg_file):
    cfg = {}

    with open(cfg_file,'r') as f:
        for l in f:
            l = l.strip()
            if len(l)>0 and not l[0] == '#':
                vs = l.split('=')
                if len(vs)>0:
                    k,v = (vs[0], eval(vs[1]))
                    if not isinstance(v,list):
                        v = [v]
                    cfg[k] = v
    return cfg

def sample_config(configs):
    cfg_sample = {}
    for k in configs.keys():
        opts = configs[k]
        c = np.random.choice(len(opts),1)[0]
        cfg_sample[k] = opts[c]
    #test
    # for v in cfg_sample.values():
    # 	print(type(v))
    return cfg_sample

def load_data(dataform, feature, dagform, logfile):
    log(logfile, ("data: %s" % dataform))
    log(logfile, ("feature: %s" % feature))
    log(logfile, ("DAG: %s" % dagform))

    df = pd.read_csv(dataform, sep=',', encoding='utf-8')

    dag = pd.read_csv(dagform, sep=';', encoding='utf-8')

    feature_out = None

    if feature=='synthetic':
        feature_out = synthetic
    elif feature=='colormnist':
        feature_out = colormnist
    elif feature=='waterbirds':
        feature_out = waterbirds
    elif feature=='hastie':
        feature_out = hastie
    elif feature=='readmission':
        feature_out = readmission
    elif feature=='adult':
        feature_out = adult
    elif feature=='churnmodel':
        feature_out = churnmodel
    elif feature=='higgs':
        feature_out = higgs
    elif feature=='bankadd':
        feature_out = bankadd
    elif feature=='creditcard':
        feature_out = creditcard
    elif feature=='icustay':
        feature_out = icustay
    elif feature=='icumort':
        feature_out = icustay

    return df, feature_out, dag
    # return feature_out, dag
