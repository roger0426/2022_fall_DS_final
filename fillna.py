import pandas as pd 
import argparse
from pathlib import Path

def fillna(data):
    #fillna
    data['ind_empleado'] = data['ind_empleado'].fillna('NA')
    data['pais_residencia'] = data['pais_residencia'].fillna('NA')
    data['sexo'] = data['sexo'].fillna('NA')
    data['ult_fec_cli_1t'] = pd.to_datetime(data['ult_fec_cli_1t'])
    data['ind_nuevo'] = data['ind_nuevo'].fillna(0.0).astype(bool)
    data['indrel'] = data['indrel'].fillna(1.0).replace(1., 0).astype(bool)
    data['indrel_1mes'] = data['indrel_1mes'].replace('P', -1).astype(float).fillna(-2).astype(int).astype(str)
    data['tiprel_1mes'] = data['tiprel_1mes'].fillna('NA')
    data['indresi'] = data['indresi'].fillna('S').replace('N', 0).replace('S', 1).astype(bool)
    data['indext'] = data['indext'].fillna('N').replace('N', 0).replace('S', 1).astype(bool)
    data['conyuemp'] = data['conyuemp'].fillna('N').replace('N', 0).replace('S', 1).astype(bool)
    data['canal_entrada'] = data['canal_entrada'].fillna('NA')
    data['indfall'] = data['indfall'].fillna('N').replace('N', 0).replace('S', 1).astype(bool)
    data['tipodom'] = data['tipodom'].fillna(0).astype(bool)
    data['cod_prov'] = data['cod_prov'].fillna(-1)
    data['nomprov'] = data['nomprov'].fillna('NA')
    data['segmento'] = data['segmento'].fillna('NA')

    dummy_column = ['ind_empleado', 'pais_residencia', 'sexo', 'indrel_1mes', 'tiprel_1mes', 'canal_entrada', 'nomprov', 'segmento']
    dummy_data = pd.get_dummies(data[dummy_column]).astype(bool)
    data = pd.concat([data, dummy_data], axis=1)

    return data
    
if __name__ == '__main__':
    parser  = argparse.ArgumentParser()
    parser.add_argument('--input_data', type=Path, default=Path('train_preprocessed.pkl'))
    parser.add_argument('--output_data', type=Path, default=Path('train_preprocessed.pkl'))

    args = parser.parse_args()
    data = pd.read_pickle(args.input_data)
    data = fillna(data)

    data.to_pickle(args.output_data)
    #dummy_data.to_pickle('train_preprocessed_dummy.pkl')
