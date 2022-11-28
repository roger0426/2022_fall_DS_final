import os
import sys
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from skopt import BayesSearchCV
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

onthot_col = ['ind_nuevo','indrel','indext','ind_actividad_cliente',
            'is_Spain','sexo_female','sexo_male','canal_entrada_007','canal_entrada_013',
            'canal_entrada_KAA','canal_entrada_KAB','canal_entrada_KAE','canal_entrada_KAF',
            'canal_entrada_KAG','canal_entrada_KAH','canal_entrada_KAI','canal_entrada_KAJ',
            'canal_entrada_KAP','canal_entrada_KAQ','canal_entrada_KAR','canal_entrada_KAS',
            'canal_entrada_KAT','canal_entrada_KAW','canal_entrada_KAY','canal_entrada_KAZ',
            'canal_entrada_KBZ','canal_entrada_KCC','canal_entrada_KCH','canal_entrada_KCI',
            'canal_entrada_KEY','canal_entrada_KFA','canal_entrada_KFC','canal_entrada_KFD',
            'canal_entrada_KHC','canal_entrada_KHD','canal_entrada_KHE','canal_entrada_KHF',
            'canal_entrada_KHK','canal_entrada_KHL','canal_entrada_KHM','canal_entrada_KHN',
            'canal_entrada_KHQ','canal_entrada_RED','canal_entrada__others','nomprov_ALAVA',
            'nomprov_ALBACETE','nomprov_ALICANTE','nomprov_ALMERIA','nomprov_ASTURIAS',
            'nomprov_AVILA','nomprov_BADAJOZ','nomprov_BALEARS, ILLES','nomprov_BARCELONA',
            'nomprov_BIZKAIA','nomprov_BURGOS','nomprov_CACERES','nomprov_CADIZ','nomprov_CANTABRIA',
            'nomprov_CASTELLON','nomprov_CEUTA','nomprov_CIUDAD REAL','nomprov_CORDOBA',
            'nomprov_CORUÑA, A','nomprov_CUENCA','nomprov_GIPUZKOA','nomprov_GIRONA','nomprov_GRANADA',
            'nomprov_GUADALAJARA','nomprov_HUELVA','nomprov_HUESCA','nomprov_JAEN','nomprov_LEON',
            'nomprov_LERIDA','nomprov_LUGO','nomprov_MADRID','nomprov_MALAGA','nomprov_MELILLA',
            'nomprov_MURCIA','nomprov_NAVARRA','nomprov_OURENSE','nomprov_PALENCIA','nomprov_PALMAS, LAS',
            'nomprov_PONTEVEDRA','nomprov_RIOJA, LA','nomprov_SALAMANCA','nomprov_SANTA CRUZ DE TENERIFE',
            'nomprov_SEGOVIA','nomprov_SEVILLA','nomprov_SORIA','nomprov_TARRAGONA','nomprov_TERUEL',
            'nomprov_TOLEDO','nomprov_VALENCIA','nomprov_VALLADOLID','nomprov_ZAMORA','nomprov_ZARAGOZA',
            'segmento_01 - TOP','segmento_02 - PARTICULARES','segmento_03 - UNIVERSITARIO']
num_col = ['age', 'antiguedad', 'renta', 'fecha_dato_month']

history = ['ind_ahor_fin_ult1_history','ind_aval_fin_ult1_history','ind_cco_fin_ult1_history',
           'ind_cder_fin_ult1_history','ind_cno_fin_ult1_history','ind_ctju_fin_ult1_history',
           'ind_ctma_fin_ult1_history','ind_ctop_fin_ult1_history','ind_ctpp_fin_ult1_history',
           'ind_deco_fin_ult1_history','ind_dela_fin_ult1_history','ind_deme_fin_ult1_history',
           'ind_ecue_fin_ult1_history','ind_hip_fin_ult1_history','ind_fond_fin_ult1_history',
           'ind_plan_fin_ult1_history','ind_pres_fin_ult1_history','ind_reca_fin_ult1_history',
           'ind_tjcr_fin_ult1_history','ind_valo_fin_ult1_history','ind_viv_fin_ult1_history',
           'ind_nomina_ult1_history','ind_nom_pens_ult1_history','ind_recibo_ult1_history']

'''
product = ['ind_ahor_fin_ult1','ind_aval_fin_ult1','ind_cco_fin_ult1','ind_cder_fin_ult1',
           'ind_cno_fin_ult1','ind_ctju_fin_ult1','ind_ctma_fin_ult1','ind_ctop_fin_ult1',
           'ind_ctpp_fin_ult1','ind_deco_fin_ult1','ind_deme_fin_ult1','ind_dela_fin_ult1',
           'ind_ecue_fin_ult1','ind_fond_fin_ult1','ind_hip_fin_ult1','ind_plan_fin_ult1',
           'ind_pres_fin_ult1','ind_reca_fin_ult1','ind_tjcr_fin_ult1','ind_valo_fin_ult1',
           'ind_viv_fin_ult1','ind_nomina_ult1','ind_nom_pens_ult1','ind_recibo_ult1']
'''
product = ["ind_cco_fin_ult1", "ind_ctop_fin_ult1", "ind_nom_pens_ult1", "ind_ctma_fin_ult1"] # 0.65, 0.13, 0.07, 0.01

month3 = ['ind_ahor_fin_ult1_3month','ind_aval_fin_ult1_3month','ind_cco_fin_ult1_3month',
          'ind_cder_fin_ult1_3month','ind_cno_fin_ult1_3month','ind_ctju_fin_ult1_3month',
          'ind_ctma_fin_ult1_3month','ind_ctop_fin_ult1_3month','ind_ctpp_fin_ult1_3month',
          'ind_deco_fin_ult1_3month','ind_deme_fin_ult1_3month','ind_dela_fin_ult1_3month',
          'ind_ecue_fin_ult1_3month','ind_fond_fin_ult1_3month','ind_hip_fin_ult1_3month',
          'ind_plan_fin_ult1_3month','ind_pres_fin_ult1_3month','ind_reca_fin_ult1_3month',
          'ind_tjcr_fin_ult1_3month','ind_valo_fin_ult1_3month','ind_viv_fin_ult1_3month',
          'ind_nomina_ult1_3month','ind_nom_pens_ult1_3month','ind_recibo_ult1_3month']

with open('../data/my_train_v3.pkl', 'rb') as f:
    data = pickle.load(f)

method = sys.argv[1]
print(method)

os.makedirs(f'./result/{method}', exist_ok=True)
fp = open(f"./result/{method}/result.txt", "w")


if method == "svm" or method == "logistic":
    scaled_features = StandardScaler().fit_transform(data[num_col].values)
    X = np.concatenate((data[onthot_col].values, scaled_features), axis=1)
else:
    X = data[onthot_col+num_col].values

for target in product:

    print(target)
    fp.write(f"{target}\n")
    
    X = np.concatenate((X, np.expand_dims(data[target+'_history'].values, axis=1)), axis=1)
    y = data[target+'_3month'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    if method == "svm":
        #################################### SVM ####################################
        from sklearn import svm
        '''
        # 建立 linearSvc 模型，Parameters: C 限制模型的複雜度，防止過度擬合、max_iter 最大迭代次數，預設1000。
        model=svm.LinearSVC(C=1, max_iter=1000)
        # 建立 kernel='linear' 模型
        model=svm.SVC(kernel='linear', C=1)
        # 建立 kernel='poly' 模型，degree:3代表轉換到三次空間進行分類、gamma: 數值越大越能做複雜的分類邊界。
        model=svm.SVC(kernel='poly', degree=3, gamma='auto', C=1)
        # 建立 kernel='rbf' 模型（adial Basis Function 高斯轉換）
        model=svm.SVC(kernel='rbf', gamma=0.7, C=1)
        '''
        # model=svm.LinearSVC(C=1, max_iter=1000, class_weight='balanced',)
        model = BayesSearchCV(
            svm.SVC(class_weight='balanced'),
            {
                'C': (1e-6, 1e+2, 'log-uniform'),
                'gamma': (1e-6, 1e+1, 'log-uniform'),
                'degree': (4, 6, 8),  # integer valued parameter
                'kernel': ['linear', 'poly', 'rbf'],  # Categorical parameter
            },
            n_iter=32,
            n_jobs=8,
            n_points=8,
            cv=3, 
            scoring="roc_auc",
            random_state=91
        )
        #################################### SVM ####################################
    elif method == "knn":
        #################################### KNN ####################################
        from sklearn.neighbors import KNeighborsClassifier
        # model = KNeighborsClassifier(n_neighbors=5)
        model = BayesSearchCV(
            KNeighborsClassifier(), 
            {
                'n_neighbors' : list(range(2,11)) , 
                'algorithm' : ['auto','ball_tree','kd_tree','brute']
            }, 
            n_iter=32, 
            n_jobs=8,
            n_points=8, 
            cv=3, 
            scoring="roc_auc",
            random_state=91
        )
        #################################### KNN ####################################

    elif method == "logistic":
        ############################# Logistic Regression #############################
        from sklearn.linear_model import LogisticRegression
        from skopt.space.space import Real, Categorical
        # model = LogisticRegression(multi_class='auto', 
        #                            solver='lbfgs',
        #                            penalty='l2',
        #                            random_state=0,
        #                            class_weight='balanced', # balanced: n_samples / (n_classes * np.bincount(y))
        #                            max_iter=5000)
        search_space =[
        {
            "solver": Categorical(['liblinear']),
            "penalty": Categorical(['l1', 'l2']),
            "fit_intercept": Categorical([True, False]),
            #"warm_start": Categorical([True, False])
        },
        {
            "solver": Categorical(['lbfgs', 'newton-cg', 'sag']),
            "penalty": Categorical(['l2', 'none']),
            "fit_intercept": Categorical([True, False]),
            #"warm_start": Categorical([True, False])
        },
        {
            "solver": Categorical(['saga']),
            "penalty": Categorical(['l1', 'l2', 'none']),
            "fit_intercept": Categorical([True, False]),
            #"warm_start": Categorical([True, False])
        },
        {
            "solver": Categorical(['saga']),
            "penalty": Categorical(['elasticnet']),
            "fit_intercept": Categorical([True, False]),
            "l1_ratio": Real(0, 1, prior='uniform'),
            #"warm_start": Categorical([True, False])
        },]
        model = BayesSearchCV(
            LogisticRegression(max_iter=5000, class_weight='balanced'), 
            search_space,
            n_iter=32, 
            n_jobs=8,
            n_points=8, 
            cv=3, 
            scoring="roc_auc",
            random_state=91
        )
        ############################# Logistic Regression #############################

    elif method == "decision_tree":
        ############################### Decision Tree ###############################
        from sklearn.tree import DecisionTreeClassifier
        model = DecisionTreeClassifier(criterion = 'entropy',
                                       max_depth=6,
                                       random_state=91,
                                       class_weight='balanced')
        ############################### Decision Tree ###############################

    elif method == "random_forest":
        ############################### Random Forest ###############################
        '''
        隨機森林是進階版的決策樹。使用 Bagging + 隨機特徵的技術所產生出來的 Ensemble learning 演算法。

        Parameters:
        - n_estimators: 森林中樹木的數量，預設=100。
        - max_features: 劃分時考慮的最大特徵數，預設auto。
        - criterion: 亂度的評估標準，gini/entropy。預設為gini。
        - max_depth: 樹的最大深度。
        - splitter: 特徵劃分點選擇標準，best/random。預設為best。
        - random_state: 亂數種子，確保每次訓練結果都一樣，splitter=random 才有用。
        - min_samples_split: 至少有多少資料才能再分
        - min_samples_leaf: 分完至少有多少資料才能分

        Attributes:
        - feature_importances_: 查詢模型特徵的重要程度。

        Methods:
        - fit: 放入X、y進行模型擬合。
        - predict: 預測並回傳預測類別。
        - score: 預測成功的比例。
        - predict_proba: 預測每個類別的機率值。
        - get_depth: 取得樹的深度。
        '''
        # from sklearn.ensemble import RandomForestClassifier
        from imblearn.ensemble import BalancedRandomForestClassifier
        # model = RandomForestClassifier(n_estimators=100, 
        #                                criterion = 'gini',
        #                                class_weight='balanced')
        model = BalancedRandomForestClassifier(n_estimators=100, 
                                               criterion = 'gini')
        ############################### Random Forest ###############################

    elif method == "xgboost":
        ################################## XGBoost ##################################
        from xgboost import XGBClassifier
        '''
        Boosting 則是希望能夠由後面生成的樹，來修正前面樹學的不好的地方。

        Parameters:
        - n_estimators: 總共迭代的次數，即決策樹的個數。預設值為100。
        - max_depth: 樹的最大深度，默認值為6。
        - booster: gbtree 樹模型(預設) / gbliner 線性模型
        - learning_rate: 學習速率，預設0.3。
        - gamma: 懲罰項係數，指定節點分裂所需的最小損失函數下降值。

        Attributes:
        - feature_importances_: 查詢模型特徵的重要程度。

        Methods:
        - fit: 放入X、y進行模型擬合。
        - predict: 預測並回傳預測類別。
        - score: 預測成功的比例。
        - predict_proba: 預測每個類別的機率值。
        '''
        bal = len(y)//sum(y)
        search_space = {
                    'max_depth': range (2, 10, 1),
                    'n_estimators': range(60, 250, 40),
                    'learning_rate': [0.1, 0.01, 0.005],
                    'scale_pos_weight': [int(bal*0.7), bal, int(bal*1.5), bal*2]
                    }
        model = BayesSearchCV(
            XGBClassifier(tree_method='gpu_hist', gpu_id=0), 
            search_space,
            n_iter=32, 
            n_jobs=8,
            n_points=8, 
            cv=3, 
            scoring="roc_auc",
            random_state=91
        )
        # model = XGBClassifier(n_estimators=100, learning_rate= 0.3)
        ################################## XGBoost ##################################

    elif method == "lightgbm":
        '''
        cd LightGBM && rm -rf build && mkdir build && cd build && cmake -DUSE_GPU=1 
        ../../LightGBM && make -j4 && cd ../python-package && python3 setup.py install --precompile --gpu;
        '''
        ################################# LightGBM ##################################
        from lightgbm import LGBMClassifier
        '''
        Parameters:
        - num_iterations: 總共迭代的次數，即決策樹的個數。預設值為100。
        - learning_rate: 學習速率，預設0.1。
        - boosting: 選擇 boosting 種類。共四種 gbdt、rf、dart、goss，預設為 gbdt。
        - max_depth: 樹的最大深度，預設值為-1即表示無限制。
        - min_data_in_leaf: 一個子葉中最少數據，可用於處理過擬合。預設20筆。
        - max_bin: 將特徵值放入桶中的最大bins數。預設為255。

        Attributes:
        - feature_importances_: 查詢模型特徵的重要程度。

        Methods:
        - fit: 放入X、y進行模型擬合。
        - predict: 預測並回傳預測類別。
        - score: 預測成功的比例。
        - predict_proba: 預測每個類別的機率值。
        '''
        model = LGBMClassifier(device= 'gpu', gpu_platform_id=0, gpu_device_id=0, class_weight='balanced')
        # model = LGBMClassifier(is_unbalance=True)
        
        ################################# LightGBM ##################################
    elif method == "catboost":
        ################################# CatBoost ##################################
        from catboost import CatBoostClassifier
        '''
        Parameters:
        - iterations: 總共迭代的次數，即決策樹的個數。預設值為 1000。
        - use_best_model: 設定 True 時必須給定驗證集，將會留下驗證集分中數最高的模型。
        - early_stopping_rounds: 連續訓練N代，若結果未改善則提早停止訓練。
        - od_type: IncToDec/Iter，預設 Iter 防止 Overfitting 評估方式，若設定前者需要設定閥值。
        - eval_metric: 模型評估方式。
        - loss_function: 計算loss方法。
        - verbose: True(1)/Flase(0)，預設1顯示訓練過程。
        - random_state: 亂數種子，確保每次訓練結果都一樣。
        - learning_rate: 預設 automatically。
        - depth: 樹的深度，預設 6。
        - cat_features: 輸入類別特徵的索引，它會自動幫你處理。
        
        Attributes:
        - feature_importances_: 查詢模型特徵的重要程度。
        
        Methods:
        - fit: 放入X、y進行模型擬合。
        - predict: 預測並回傳預測類別。
        - score: 預測成功的比例。
        '''
        model = CatBoostClassifier(verbose=10,
                                   random_state=0,
                                   scale_pos_weight=len(y)//sum(y))
        ################################# CatBoost ##################################
    else:
        raise ValueError("method not exist.")
    
    best_model = f"./result/{method}/{target}_model.pkl"
    # load the model
    # model = pickle.load(open(best_model, 'rb'))
    
    model.fit(X_train, y_train)
    # 使用訓練資料預測分類
    # predicted=model.predict(X_train)
    # 計算準確率
    score = model.score(X_train, y_train)
    print('train roc_auc_score: ', score)
    fp.write(f"train roc_auc_score: {score}\n")
    score = model.score(X_test, y_test)
    print('test roc_auc_score: ', score)
    fp.write(f"test roc_auc_score: {score}\n")

    if method == "decision_tree":
        ############################### Decision Tree ###############################
        # plot tree
        # import graphviz
        import pydotplus
        from sklearn.tree import export_graphviz
        dot_data = export_graphviz(model, out_file=None, 
                                feature_names=X_train.columns,
                                class_names=['False', 'True'],
                                filled=True, rounded=True,  
                                special_characters=True)  
        # graph = graphviz.Source(dot_data) 
        graph = pydotplus.graph_from_dot_data(dot_data)
        graph.write_pdf(f'./result/{method}/{target}.pdf')
        ############################### Decision Tree ###############################

    if method == "random_forest":
        ############################### Random Forest ###############################
        from sklearn import tree
        fig, axes = plt.subplots(nrows = 1,ncols = 5,figsize = (10,2), dpi=900)
        # 繪製前五棵樹
        for index in range(0, 5):
            tree.plot_tree(model.estimators_[index],
                        feature_names=X_train.columns,
                        class_names=['False', 'True'],
                        filled = True,
                        ax = axes[index])
        fig.savefig(f'./result/{method}/{target}.eps',format='eps',bbox_inches = "tight")
        plt.clf()
        ############################### Random Forest ###############################

    if method == "xgboost":
        ################################## XGBoost ##################################
        from xgboost import plot_importance
        # from xgboost import plot_tree
        ax = plot_importance(model, title='Feature Importance', ylabel='feature', grid=False)
        ax.figure.tight_layout()
        ax.figure.savefig(f'./result/{method}/{target}_featW.png')
        plt.clf()
        ################################## XGBoost ##################################

    if method == "catboost":
        ################################# CatBoost ##################################
        import shap
        explainercat = shap.TreeExplainer(model)
        # shap_values_cat_test = explainercat.shap_values(X_test)
        shap_values_cat_train = explainercat.shap_values(X_train)
        shap.summary_plot(shap_values_cat_train, X_train, plot_type="bar", show=False)
        plt.savefig(f"./result/{method}/{target}_featW.png")
        plt.clf()
        ################################# CatBoost ##################################
    
    # get confusion matrix
    probs = model.predict_proba(X_test)
    pred = np.argmax(probs, 1)
    confusion_matrix = pd.crosstab(y_test, pred,
                                   rownames=['Actual'],
                                   colnames=['Predicted'])
    plot = sns.heatmap(confusion_matrix, linewidth=.5, annot=True, fmt=',.0f')
    plot.set_title(f'{method} % {target}')
    fig = plot.get_figure()
    fig.savefig(f'./result/{method}/{target}_confmat.png')
    plt.clf()
    
    # keep probabilities for the positive outcome only
    probs = probs[:, 1]
    # calculate scores
    # auc = roc_auc_score(y_test, probs)
    # calculate roc curves
    fpr, tpr, _ = roc_curve(y_test, probs)
    plt.plot(fpr, tpr, marker='.', label='Logistic')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(f'{method} {target} ROC')
    plt.savefig(f'./result/{method}/{target}_roc.png')
    plt.clf()
    # save the model
    pickle.dump(model.best_estimator_, open(best_model, 'wb'))

fp.close()