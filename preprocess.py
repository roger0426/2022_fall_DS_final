import pandas as pd 
import numpy as np
import argparse
from pathlib import Path
import logging 
from tqdm import tqdm 

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)


if __name__ == '__main__':
    #data = pd.read_csv('train_ver2.csv')
    parser  = argparse.ArgumentParser()
    parser.add_argument('--input_data', type=Path, default=Path('train_ver2.csv'))
    parser.add_argument('--output_data', type=Path, default=Path('train_preprocessed.pkl'))


    args = parser.parse_args()

    
    data = pd.read_csv(args.input_data)
    logging.info('data reading done')

    data['fecha_dato'] = pd.to_datetime(data['fecha_dato'])

    
    data = data.sort_values(by=['ncodpers', 'fecha_dato'])
    logging.info('data sorting done')

    #data = data[data['ncodpers'] < 43136] #for testing 
    
    #clean data
    #preprocessing age
    data['age'] = data['age'].replace(' NA', -1).apply(int)
    #Gender preprocess
    data['sexo'] = data['sexo'].replace('H', 'male').replace('V', 'female')
    #customer seniority preprocessing (how long the customer be a customer)
    data['antiguedad'] = data['antiguedad'].str.strip().replace('NA', -1).fillna(-1).apply(int)
    anti_clean = data.groupby('ncodpers')['antiguedad'].max()
    data['antiguedad'] = data.join(anti_clean, on = 'ncodpers', how='left', rsuffix='_clean')['antiguedad_clean']
    #customer household income preprocessing
    data['renta'] = data['renta'].fillna(-1).apply(int)
    #active customer preprocessing
    data['ind_actividad_cliente'] = data['ind_actividad_cliente'].fillna(0).apply(int)

    logging.info('data cleaning done')

    customer_max_date = data.groupby('ncodpers').agg({'fecha_dato':max})
    id_list = [0] + (pd.merge(customer_max_date, data.reset_index()[['index','ncodpers', 'fecha_dato']].reset_index(), how='inner', on=['ncodpers', 'fecha_dato'])['level_0'] + 1).to_list()
    split_size = 100
    c_num = len(id_list) - 1
    step_size = c_num // split_size

    id_list_short = []
    for i in range(split_size):
        s = id_list[i * step_size]
        id_list_short.append(s)
    id_list_short += [len(data)]

    logging.info('split index done')

        
        

    #preprocess history product used and future product used
    product_key = []
    customer_key = []
    history_product_key = []
    three_month_product_key = []

    for k in data.columns:
        if '_ult1' in k and 'ind_' in k:
            product_key.append(k)
            history_product_key.append(k+'_history')
            three_month_product_key.append(k+'_3month')
            data[k] = data[k].astype(bool)   
            data[k+'_history'] = False 
            data[k+'_3month'] = False
        else:
            customer_key.append(k)

    logging.info('product key select done')

    for split_index in tqdm(range(len(id_list_short)-1)):
        s = id_list_short[split_index]
        e = id_list_short[split_index+1]
        data_split = data.iloc[s:e]

        three_month_product = data_split[product_key].copy(deep=True)
        for k in product_key:
            three_month_product[k] = False
        customer_max_date = data_split.groupby('ncodpers').agg({'fecha_dato':max})

        
        for i in range(3):
            past_month_product = data_split[product_key].copy(deep=True)
            past_month_product = past_month_product.shift(-i-1)
            reset_date = customer_max_date - pd.DateOffset(months=i)
            #past_month_product[['ncodpers', 'fecha_dato']] = x[['ncodpers', 'fecha_dato']]
            reset_index = pd.merge(data_split[['ncodpers', 'fecha_dato']].reset_index(), reset_date, how='inner', left_on=['ncodpers', 'fecha_dato'], right_on=['ncodpers', 'fecha_dato']).set_index('index')
            reset_index[product_key] = False
            past_month_product.update(reset_index[product_key])
            past_month_product = past_month_product.fillna(False)
            three_month_product = three_month_product | past_month_product

        history_product = data_split.groupby(['ncodpers'])[product_key].cumsum().astype(bool)

        data.update(history_product.add_suffix('_history'))
        data.update(three_month_product.add_suffix('_3month'))
        #data = data.join(history_product, how='left', lsuffix='', rsuffix='_history')
        #data = data.join(three_month_product, how='left', lsuffix='', rsuffix='_three_month')
        

    logging.info('product preprocessing done')
    data.to_pickle(args.output_data)