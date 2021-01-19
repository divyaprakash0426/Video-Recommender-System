
from data_manipulation import *
from Matrix_Factorization import *
from Probabilistic_MF import *
from SVD import *
from helper_functions import *
from ExtraTrees_and_XGB import *

z, total_u,total_p,videoid2videoid,train,test,uv_table,raw_data = data()

reg1=ExtraTreesRegressor()
reg2=xg.XGBRegressor()
ET_XG(reg1)
ET_XG(reg2)

final_table =cf()

caculate_mse(final_table)
conf_mat_plot(final_table,title='MF')
recommendation(final_table, 10,10,rawId= True)

final_table1 =svdrecommendation(factors=150)
caculate_mse(final_table1)
conf_mat_plot(final_table1,title='SVD')
recommendation(final_table1, 10,10,rawId= True)

final_table_MF =Matrix_Factorization( factors=30, maxIter=100, LRate=0.02, GD_end=1e-3, plot=1)
caculate_mse(final_table_MF)
conf_mat_plot(final_table_MF,title='Matrix_Factorization')
recommendation(final_table_MF, 10,10,rawId= True)

final_table_PMF =Probabilistic_Matrix_Factorization( factors=30, maxIter=100, LRate=0.02, GD_end=1e-3, plot=1)
caculate_mse(final_table_PMF)
conf_mat_plot(final_table_PMF,title='Probabilistic_Matrix_Factorization')
recommendation(final_table_PMF, 10,10,rawId= True)


final_table = (final_table_MF + final_table_PMF)/2
caculate_mse(final_table)
conf_mat_plot(final_table,title='Avg_of_MF_and_PMF')
recommendation(final_table, 10,10,rawId= True)
