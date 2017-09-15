# coding:utf-8
'''
Created on 2017/9/15.

@author: chk01
'''
import pandas as pd

TRAIN_FILE = 'C:/Users/chk01/Desktop/TF/census/census-income.data'
TEST_FILE = 'C:/Users/chk01/Desktop/TF/census/census-income.test'
# COLUMNS = [
#     'age'
# ]
COLUMNS = [
    'age', 'class_of_worker', 'detailed_industry_recode',
    'detailed_occupation_recode', 'education', 'wage_per_hour',
    'enroll_in_edu_inst_last_wk', 'marital_stat', 'major_industry_code',
    'major_occupation_code', 'race', 'hispanic_origin', 'sex',
    'member_of_labor_union', 'reason_for_unemployment',
    'full_or_part_time_employment_stat', 'capital_gains', 'capital_losses',
    'dividends_from_stocks', 'tax_filer_stat', 'region_of_previous_residence',
    'state_of_previous_residence', 'detailed_household_and_family_stat',
    'detailed_household_summary_in_household', 'instance_weight',
    'migration_code_change_in_msa', 'migration_code_change_in_reg',
    'migration_code_move_within_reg', 'live_in_this_house_1year_ago',
    'migration_prev_res_in_sunbelt', 'num_persons_worked_for_employer',
    'family_members_under18', 'country_of_birth_father',
    'country_of_birth_mother', 'country_of_birth_self', 'citizenship',
    'own_business_or_self_employed',
    'fill_inc_questionnaire_for_veteran_admin', 'veterans_benefits',
    'weeks_worked_in_year', 'year', 'label'
]
LABEL_COLUMN = 'label'

df_train = pd.read_csv(TRAIN_FILE, names=COLUMNS,skipinitialspace=True)
# print(df_train)
# df_test = pd.read_csv(TEST_FILE, names=COLUMNS, skipinitialspace=True)
df_train = df_train.dropna(how='any', axis=0)
# df_test = df_test.dropna(how='any', axis=0)
df_train[['age']] = df_train[['age']].astype(str)
# df_test[['detailed_occupation_recode']] = df_test[['detailed_industry_recode']].astype(str)
#
df_train[LABEL_COLUMN] = (
    df_train[LABEL_COLUMN].apply(lambda x: '+' in x)).astype(int)
print(df_train[LABEL_COLUMN])
# df_test[LABEL_COLUMN] = (
#     df_test[LABEL_COLUMN].apply(lambda x: '+' in x)).astype(int)
print (df_train.dtypes)
# a = ['one', 'two', 'three']
# b = [1, 2, 3]
# english_column = pd.Series(a, name='english')
# number_column = pd.Series(b, name='number')
# # predictions = pd.concat([english_column, number_column], axis=1)
# # another way to handle
# save = pd.DataFrame({'english': a, 'number': b})
# save.to_csv('b.txt')
# print(pd.read_csv('b.txt'))
