import pandas as pd
import numpy as np
import argparse
from pathlib import Path
import logging 
from tqdm import tqdm 

from fillna import fillna

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)

drop_column_list=['ind_empleado', 'fecha_alta', 'indrel_1mes', 'indresi', 'conyuemp', 'indfall', 'tipodom', 'cod_prov']
drop_row_na_list=['sexo', 'age']
to_bool_list = ['ind_ahor_fin_ult1', 'ind_aval_fin_ult1', 'ind_cco_fin_ult1', 'ind_cder_fin_ult1', 
'ind_cno_fin_ult1', 'ind_ctju_fin_ult1', 'ind_ctma_fin_ult1', 
'ind_ctop_fin_ult1', 'ind_ctpp_fin_ult1', 'ind_deco_fin_ult1', 'ind_deme_fin_ult1', 'ind_dela_fin_ult1', 
'ind_ecue_fin_ult1', 'ind_fond_fin_ult1', 'ind_hip_fin_ult1', 'ind_plan_fin_ult1', 'ind_pres_fin_ult1', 
'ind_reca_fin_ult1', 'ind_tjcr_fin_ult1', 'ind_valo_fin_ult1', 'ind_viv_fin_ult1', 'ind_nomina_ult1', 
'ind_nom_pens_ult1', 'ind_recibo_ult1', 'ind_ahor_fin_ult1_history', 'ind_ahor_fin_ult1_3month', 
'ind_aval_fin_ult1_history', 'ind_aval_fin_ult1_3month', 'ind_cco_fin_ult1_history', 'ind_cco_fin_ult1_3month', 
'ind_cder_fin_ult1_history', 'ind_cder_fin_ult1_3month', 'ind_cno_fin_ult1_history', 'ind_cno_fin_ult1_3month', 
'ind_ctju_fin_ult1_history', 'ind_ctju_fin_ult1_3month', 'ind_ctma_fin_ult1_history', 'ind_ctma_fin_ult1_3month', 
'ind_ctop_fin_ult1_history', 'ind_ctop_fin_ult1_3month', 'ind_ctpp_fin_ult1_history', 'ind_ctpp_fin_ult1_3month', 
'ind_deco_fin_ult1_history', 'ind_deco_fin_ult1_3month', 'ind_deme_fin_ult1_history', 'ind_deme_fin_ult1_3month', 
'ind_dela_fin_ult1_history', 'ind_dela_fin_ult1_3month', 'ind_ecue_fin_ult1_history', 'ind_ecue_fin_ult1_3month', 
'ind_fond_fin_ult1_history', 'ind_fond_fin_ult1_3month', 'ind_hip_fin_ult1_history', 'ind_hip_fin_ult1_3month', 
'ind_plan_fin_ult1_history', 'ind_plan_fin_ult1_3month', 'ind_pres_fin_ult1_history', 'ind_pres_fin_ult1_3month', 
'ind_reca_fin_ult1_history', 'ind_reca_fin_ult1_3month', 'ind_tjcr_fin_ult1_history', 'ind_tjcr_fin_ult1_3month', 
'ind_valo_fin_ult1_history', 'ind_valo_fin_ult1_3month', 'ind_viv_fin_ult1_history', 'ind_viv_fin_ult1_3month', 
'ind_nomina_ult1_history', 'ind_nomina_ult1_3month', 'ind_nom_pens_ult1_history', 'ind_nom_pens_ult1_3month', 
'ind_recibo_ult1_history', 'ind_recibo_ult1_3month']
int_down_list = ['ncodpers', 'age', 'antiguedad', 'renta', 'ind_actividad_cliente', 'renta']
float_down_list = []

if __name__ == '__main__':
    logging.info('Start process')
    #data = pd.read_csv('train_ver2.csv')
    parser  = argparse.ArgumentParser()
    parser.add_argument('--input_data', type=Path, default=Path('./origin_data/train_ver2.csv.zip'))
    parser.add_argument('--output_data', type=Path, default=Path('preprocessed_test.pkl'))


    args = parser.parse_args()

    
    data = pd.read_csv(args.input_data)
    logging.info('data reading done')
    # data = data[:int(len(data)/1000)]
    # print(data.info(verbose=True))
    data.drop(drop_column_list, axis=1, inplace=True)
    # print(data.info(verbose=True))

    data['fecha_dato'] = pd.to_datetime(data['fecha_dato'])
    
    data = data.sort_values(by=['ncodpers', 'fecha_dato'])
    logging.info('data sorting done')

    #data = data[data['ncodpers'] < 43136] #for testing 
    
    # clean data
    data.dropna(axis=0, subset=drop_row_na_list, inplace=True)
    # preprocessing age
    # data['age'] = data['age'].replace(' NA', -1).apply(int)
    data['age'] = data['age'].astype(int)
    data.drop(data[data['age'] >= 120].index, inplace = True)
    # Gender preprocess
    data['sexo'] = data['sexo'].replace('H', 'male').replace('V', 'female')
    # customer seniority preprocessing (how long the customer be a customer)
    data['antiguedad'] = data['antiguedad'].str.strip().replace('NA', -1).fillna(-1).apply(int)
    anti_clean = data.groupby('ncodpers')['antiguedad'].max()
    data['antiguedad'] = data.join(anti_clean, on = 'ncodpers', how='left', rsuffix='_clean')['antiguedad_clean']
    # customer household income preprocessing
    data['renta'] = data['renta'].fillna(-1).apply(int)
    # active customer preprocessing
    data['ind_actividad_cliente'] = data['ind_actividad_cliente'].fillna(0).astype(bool)
    # canal_entrada gather classes whose sample < 1/100
    data['canal_entrada'] = data['canal_entrada'].fillna('NA')
    canal_entrada_others = [canel for canel in data['canal_entrada'].unique() if (data['canal_entrada'].value_counts()[canel] < len(data)/1000)]
    # print(data['canal_entrada'].value_counts())
    data['canal_entrada'] = data['canal_entrada'].replace(canal_entrada_others, '_others')
    # print(data['canal_entrada'].value_counts())
    # print(data['canal_entrada'].unique())

    # pais_residencia gather classes whose sample < 1/100
    data['pais_residencia'] = data['pais_residencia'].fillna('NA')
    pais_residencia_others = [canel for canel in data['pais_residencia'].unique() if (data['pais_residencia'].value_counts()[canel] < len(data)/1000)]
    data['pais_residencia'] = data['pais_residencia'].replace(pais_residencia_others, '_others')
    print(data['pais_residencia'].value_counts())
    print(data['pais_residencia'].unique())
    

    logging.info('data cleaning done')

    customer_max_date = data.groupby('ncodpers').agg({'fecha_dato':max})
    id_list = [0] + (pd.merge(customer_max_date, \
                            data.reset_index()[['index','ncodpers', 'fecha_dato']].reset_index(), \
                                                how='inner', on=['ncodpers', 'fecha_dato'])['level_0'] + 1).to_list()
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

    # print(data.info(verbose=True))
    logging.info('product key select done')

    # without   take 3.5 min for 1/100
    # with      take 1.2 min for 1/100
    for target in to_bool_list: data[target] = data[target].astype(bool)

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
            reset_index = pd.merge(data_split[['ncodpers', 'fecha_dato']].reset_index(), reset_date, how='inner', \
                                                left_on=['ncodpers', 'fecha_dato'], \
                                                right_on=['ncodpers', 'fecha_dato']).set_index('index')
            reset_index[product_key] = False
            past_month_product.update(reset_index[product_key])
            past_month_product = past_month_product.fillna(False)
            three_month_product = three_month_product | past_month_product

        history_product = data_split.groupby(['ncodpers'])[product_key].cumsum().astype(bool)

        # print('d')

        data.update(history_product.add_suffix('_history'))
        data.update(three_month_product.add_suffix('_3month'))
        #data = data.join(history_product, how='left', lsuffix='', rsuffix='_history')
        #data = data.join(three_month_product, how='left', lsuffix='', rsuffix='_three_month')

        # print('e')
    # for name, values in data.items():
    #     print('{name}: {value}'.format(name=name, value=values[0]))
    # print(data.info(verbose=True))

    data = fillna(data)

    for target in to_bool_list: data[target] = data[target].astype(bool)
    for target in int_down_list: data[target] = data[target].apply(pd.to_numeric, errors='coerce', downcast='integer')
    for target in float_down_list: data[target] = data[target].apply(pd.to_numeric, errors='coerce', downcast='float')

    print(data.info(verbose=True))

    # print(data.columns.tolist())
    
    logging.info('product preprocessing done')
    data.to_pickle(args.output_data)