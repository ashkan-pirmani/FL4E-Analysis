import pandas as pd
import numpy as np
import argparse

def clean_target(df_in):
    if "target" not in df_in.columns:
        print("Unable to clean target as the column is not in the dataframe")
        return {}

    failures = {}

    fail_ids = df_in[~(df_in["target"].isin([0,1]))]["id"].astype(int).tolist()

    for fail_id in fail_ids:
        failures[fail_id] = "target"

    return failures

def clean_mean_radius(df_in):
    if "mean_radius" not in df_in.columns:
        print("Unable to clean mean_radius as the column is not in the dataframe")
        return {}

    failures = {}
    fail_ids = df_in[(df_in["mean_radius"]<5) | (df_in["mean_radius"]>30)]["id"].astype(int).tolist()


    for fail_id in fail_ids:
        failures[fail_id] = "mean_radius"

    return failures

def clean_mean_texture(df_in):
    if "mean_texture" not in df_in.columns:
        print("Unable to clean mean_texture as the column is not in the dataframe")
        return {}

    failures = {}
    fail_ids = df_in[(df_in["mean_texture"]<5) | (df_in["mean_texture"]>40)]["id"].astype(int).tolist()


    for fail_id in fail_ids:
        failures[fail_id] = "mean_texture"

    return failures

def clean_mean_perimeter(df_in):
    if "mean_perimeter" not in df_in.columns:
        print("Unable to clean mean_perimeter as the column is not in the dataframe")
        return {}

    failures = {}
    fail_ids = df_in[(df_in["mean_perimeter"]<25) | (df_in["mean_perimeter"]>200)]["id"].astype(int).tolist()


    for fail_id in fail_ids:
        failures[fail_id] = "mean_perimeter"

    return failures

def clean_mean_area(df_in):
    if "mean_area" not in df_in.columns:
        print("Unable to clean mean_area as the column is not in the dataframe")
        return {}

    failures = {}
    fail_ids = df_in[(df_in["mean_area"]<50) | (df_in["mean_area"]>3000)]["id"].astype(int).tolist()


    for fail_id in fail_ids:
        failures[fail_id] = "mean_area"

    return failures

def clean_mean_smoothness(df_in):
    if "mean_smoothness" not in df_in.columns:
        print("Unable to clean mean_smoothness as the column is not in the dataframe")
        return {}

    failures = {}
    fail_ids = df_in[(df_in["mean_smoothness"]<0) | (df_in["mean_smoothness"]>1)]["id"].astype(int).tolist()


    for fail_id in fail_ids:
        failures[fail_id] = "mean_smoothness"

    return failures

def clean_mean_compactness(df_in):
    if "mean_compactness" not in df_in.columns:
        print("Unable to clean mean_compactness as the column is not in the dataframe")
        return {}

    failures = {}
    fail_ids = df_in[(df_in["mean_compactness"]<0) | (df_in["mean_compactness"]>1)]["id"].astype(int).tolist()


    for fail_id in fail_ids:
        failures[fail_id] = "mean_compactness"

    return failures

def clean_mean_concavity(df_in):
    if "mean_concavity" not in df_in.columns:
        print("Unable to clean mean_concavity as the column is not in the dataframe")
        return {}

    failures = {}
    fail_ids = df_in[(df_in["mean_concavity"]<0) | (df_in["mean_concavity"]>1)]["id"].astype(int).tolist()


    for fail_id in fail_ids:
        failures[fail_id] = "mean_concavity"

    return failures

def clean_mean_concave_points(df_in):
    if "mean_concave_points" not in df_in.columns:
        print("Unable to clean mean_concave_points as the column is not in the dataframe")
        return {}

    failures = {}
    fail_ids = df_in[(df_in["mean_concave_points"]<0) | (df_in["mean_concave_points"]>1)]["id"].astype(int).tolist()


    for fail_id in fail_ids:
        failures[fail_id] = "mean_concave_points"

    return failures

def clean_mean_symmetry(df_in):
    if "mean_symmetry" not in df_in.columns:
        print("Unable to clean mean_symmetry as the column is not in the dataframe")
        return {}

    failures = {}
    fail_ids = df_in[(df_in["mean_symmetry"]<0) | (df_in["mean_symmetry"]>1)]["id"].astype(int).tolist()


    for fail_id in fail_ids:
        failures[fail_id] = "mean_symmetry"

    return failures

def clean_mean_mean_fractal_dimension(df_in):
    if "mean_fractal_dimension" not in df_in.columns:
        print("Unable to clean mean_concave_points as the column is not in the dataframe")
        return {}

    failures = {}
    fail_ids = df_in[(df_in["mean_fractal_dimension"]<0) | (df_in["mean_fractal_dimension"]>1)]["id"].astype(int).tolist()


    for fail_id in fail_ids:
        failures[fail_id] = "mean_fractal_dimension"

    return failures


def update_all_failures(failures, all_failures):
    for fail_id, variables in failures.items():
        if fail_id not in all_failures:
            all_failures[fail_id] = []
        if isinstance(variables, list):
            all_failures[fail_id].extend(variables)
        else:
            all_failures[fail_id].append(variables)

def clean_data(df_in, auth_obj = 0, project_id = 0, fail_value=None, send_qa_staus=False):
    df = df_in.copy()

    df.replace(regex=r'^\s*$', value=np.nan, inplace=True) # Replaces empty strings by np.nan
    df = df.where(pd.notnull(df), None) # replaces null values with None

    all_failures = {}
    update_all_failures(clean_target(df), all_failures)
    update_all_failures(clean_mean_radius(df), all_failures)
    update_all_failures(clean_mean_texture(df), all_failures)
    update_all_failures(clean_mean_perimeter(df), all_failures)
    update_all_failures(clean_mean_area(df), all_failures)
    update_all_failures(clean_mean_smoothness(df), all_failures)
    update_all_failures(clean_mean_compactness(df), all_failures)
    update_all_failures(clean_mean_concavity(df), all_failures)
    update_all_failures(clean_mean_concave_points(df), all_failures)
    update_all_failures(clean_mean_symmetry(df), all_failures)
    update_all_failures(clean_mean_mean_fractal_dimension(df), all_failures)

    if send_qa_staus:
        send_all_qa_status(all_failures, auth_obj, df["id"].astype(int).tolist(), project_id)

    #df_flagged = flag_failures_in_df(df_in, all_failures, fail_value)

    #return repair_data(df_flagged)
    return df, all_failures

def upload_csv(input_csv, core_csv_source, core_csv_target, user_id):
    df_core = pd.read_csv(core_csv_source)
    df_new = pd.read_csv(input_csv)

    df_new = df_new[['mean_radius', 'mean_texture', 'mean_perimeter',
           'mean_area', 'mean_smoothness', 'mean_compactness', 'mean_concavity',
           'mean_concave_points', 'mean_symmetry', 'mean_fractal_dimension',
           'target']]

    df_new["id"] = np.arange(df_new.shape[0])

    df_cleaned, failures = clean_data(df_new)
    if len(failures) > 0:
        print(failures)
        #return failures
    else:
        df_new.drop(columns = ["id"],inplace = True)
        df_new["user_id"] = user_id
        df_new["entry_id"] = np.arange(df_new.shape[0])

        df_core_sub = df_core.drop(df_core.loc[df_core["user_id"]==user_id].index)
        df_core = pd.concat((df_core_sub,df_new))
        df_core.to_csv(core_csv_target,index = False)
        print(f"Uploaded data to {core_csv_target}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Client side.')
    parser.add_argument('--user_id',  type=str, help = "ID of data custodian", default = 1)
    parser.add_argument('--core_csv_source',  type=str, help = "Source path to the core dataset csv", default = "./data/core.csv")
    parser.add_argument('--core_csv_target',  type=str, help = "Target path to the core dataset csv", default = "./data/core_new.csv")
    parser.add_argument('--input_csv',  type=str, help = "Path  to the input csv", default = "./data/federated_central.csv")
    args = parser.parse_args()
    print(args)

    upload_csv(**vars(args))
    
    
    
    
    
with open("DoneU.txt", "w") as file:
    file.write("Upload Process is Done")
