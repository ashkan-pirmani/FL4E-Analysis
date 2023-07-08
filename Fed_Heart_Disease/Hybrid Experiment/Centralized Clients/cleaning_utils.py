import pandas as pd
import numpy as np
import argparse


def clean_condition(df_in):
    if "condition" not in df_in.columns:
        print("Unable to clean condition as the column is not in the dataframe")
        return {}

    failures = {}

    fail_ids = df_in[(df_in["condition"] < 0) | (df_in["condition"] > 1)]["id"].astype(int).tolist()

    for fail_id in fail_ids:
        failures[fail_id] = "condition"

    return failures


def clean_age(df_in):
    if "age" not in df_in.columns:
        print("Unable to clean age as the column is not in the dataframe")
        return {}

    failures = {}
    fail_ids = df_in[(df_in["age"] < 29) | (df_in["age"] > 77)]["id"].astype(int).tolist()

    for fail_id in fail_ids:
        failures[fail_id] = "age"

    return failures


def clean_sex(df_in):
    if "sex" not in df_in.columns:
        print("Unable to clean sex as the column is not in the dataframe")
        return {}

    failures = {}
    fail_ids = df_in[(df_in["sex"] < 0) | (df_in["sex"] > 1)]["id"].astype(int).tolist()

    for fail_id in fail_ids:
        failures[fail_id] = "sex"

    return failures


def clean_cp(df_in):
    if "cp" not in df_in.columns:
        print("Unable to clean cp as the column is not in the dataframe")
        return {}

    failures = {}
    fail_ids = df_in[(df_in["cp"] < 0) | (df_in["cp"] > 4)]["id"].astype(int).tolist()

    for fail_id in fail_ids:
        failures[fail_id] = "cp"

    return failures


def clean_trestbps(df_in):
    if "trestbps" not in df_in.columns:
        print("Unable to clean trestbps as the column is not in the dataframe")
        return {}

    failures = {}
    fail_ids = df_in[(df_in["trestbps"] < 94) | (df_in["trestbps"] > 201)]["id"].astype(int).tolist()

    for fail_id in fail_ids:
        failures[fail_id] = "trestbps"

    return failures


def clean_chol(df_in):
    if "chol" not in df_in.columns:
        print("Unable to clean chol as the column is not in the dataframe")
        return {}

    failures = {}
    fail_ids = df_in[(df_in["chol"] < 126) | (df_in["chol"] > 565)]["id"].astype(int).tolist()

    for fail_id in fail_ids:
        failures[fail_id] = "chol"

    return failures


def clean_fbs(df_in):
    if "fbs" not in df_in.columns:
        print("Unable to clean fbs as the column is not in the dataframe")
        return {}

    failures = {}
    fail_ids = df_in[(df_in["fbs"] < 0) | (df_in["fbs"] > 1)]["id"].astype(int).tolist()

    for fail_id in fail_ids:
        failures[fail_id] = "fbs"

    return failures


def clean_restecg(df_in):
    if "restecg" not in df_in.columns:
        print("Unable to clean restecg as the column is not in the dataframe")
        return {}

    failures = {}
    fail_ids = df_in[(df_in["restecg"] < 0) | (df_in["restecg"] > 2)]["id"].astype(int).tolist()

    for fail_id in fail_ids:
        failures[fail_id] = "restecg"

    return failures


def clean_thalach(df_in):
    if "thalach" not in df_in.columns:
        print("Unable to clean thalach as the column is not in the dataframe")
        return {}

    failures = {}
    fail_ids = df_in[(df_in["thalach"] < 71) | (df_in["thalach"] > 202)]["id"].astype(int).tolist()

    for fail_id in fail_ids:
        failures[fail_id] = "thalach"

    return failures


def clean_exang(df_in):
    if "exang" not in df_in.columns:
        print("Unable to clean exang as the column is not in the dataframe")
        return {}

    failures = {}
    fail_ids = df_in[(df_in["exang"] < 0) | (df_in["exang"] > 1)]["id"].astype(int).tolist()

    for fail_id in fail_ids:
        failures[fail_id] = "exang"

    return failures


def clean_oldpeak(df_in):
    if "oldpeak" not in df_in.columns:
        print("Unable to clean oldpeak as the column is not in the dataframe")
        return {}

    failures = {}
    fail_ids = df_in[(df_in["oldpeak"] < 0) | (df_in["oldpeak"] > 6.2)]["id"].astype(int).tolist()

    for fail_id in fail_ids:
        failures[fail_id] = "oldpeak"

    return failures


def clean_slope(df_in):
    if "slope" not in df_in.columns:
        print("Unable to clean slope as the column is not in the dataframe")
        return {}

    failures = {}
    fail_ids = df_in[(df_in["slope"] < 0) | (df_in["slope"] > 2)]["id"].astype(int).tolist()

    for fail_id in fail_ids:
        failures[fail_id] = "slope"

    return failures


def clean_ca(df_in):
    if "ca" not in df_in.columns:
        print("Unable to clean ca as the column is not in the dataframe")
        return {}

    failures = {}
    fail_ids = df_in[(df_in["ca"] < 0) | (df_in["ca"] > 3)]["id"].astype(int).tolist()

    for fail_id in fail_ids:
        failures[fail_id] = "ca"

    return failures


def clean_thal(df_in):
    if "thal" not in df_in.columns:
        print("Unable to clean thal as the column is not in the dataframe")
        return {}

    failures = {}
    fail_ids = df_in[(df_in["thal"] < 0) | (df_in["thal"] > 2)]["id"].astype(int).tolist()

    for fail_id in fail_ids:
        failures[fail_id] = "thal"

    return failures


def update_all_failures(failures, all_failures):
    for fail_id, variables in failures.items():
        if fail_id not in all_failures:
            all_failures[fail_id] = []
        if isinstance(variables, list):
            all_failures[fail_id].extend(variables)
        else:
            all_failures[fail_id].append(variables)


def clean_data(df_in, auth_obj=0, project_id=0, fail_value=None, send_qa_staus=False):
    df = df_in.copy()

    df.replace(regex=r'^\s*$', value=np.nan, inplace=True)  # Replaces empty strings by np.nan
    df = df.where(pd.notnull(df), None)  # replaces null values with None

    all_failures = {}
    update_all_failures(clean_condition(df), all_failures)
    update_all_failures(clean_age(df), all_failures)
    update_all_failures(clean_sex(df), all_failures)
    update_all_failures(clean_cp(df), all_failures)
    update_all_failures(clean_trestbps(df), all_failures)
    update_all_failures(clean_chol(df), all_failures)
    update_all_failures(clean_fbs(df), all_failures)
    update_all_failures(clean_restecg(df), all_failures)
    update_all_failures(clean_thalach(df), all_failures)
    update_all_failures(clean_exang(df), all_failures)
    update_all_failures(clean_oldpeak(df), all_failures)
    update_all_failures(clean_slope(df), all_failures)
    update_all_failures(clean_ca(df), all_failures)
    update_all_failures(clean_thal(df), all_failures)

    if send_qa_staus:
        send_all_qa_status(all_failures, auth_obj, df["id"].astype(int).tolist(), project_id)

    # df_flagged = flag_failures_in_df(df_in, all_failures, fail_value)

    # return repair_data(df_flagged)
    return df, all_failures


def upload_csv(input_csv, core_csv_source, core_csv_target, user_id):
    df_core = pd.read_csv(core_csv_source)
    df_new = pd.read_csv(input_csv)

    df_new = df_new[
        ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal',
         'condition']]

    df_new["id"] = np.arange(df_new.shape[0])

    df_cleaned, failures = clean_data(df_new)
    if len(failures) > 0:
        print(failures)
        # return failures
    else:
        df_new.drop(columns=["id"], inplace=True)
        df_new["user_id"] = user_id
        df_new["entry_id"] = np.arange(df_new.shape[0])

        df_core_sub = df_core.drop(df_core.loc[df_core["user_id"] == user_id].index)
        df_core = pd.concat((df_core_sub, df_new))
        df_core.to_csv(core_csv_target, index=False)
        print(f"Uploaded data to {core_csv_target}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Client side.')
    parser.add_argument('--user_id', type=str, help="ID of data custodian", default=1)
    parser.add_argument('--core_csv_source', type=str, help="Source path to the core dataset csv",
                        default="./data/core.csv")
    parser.add_argument('--core_csv_target', type=str, help="Target path to the core dataset csv",
                        default="./data/core_new.csv")
    parser.add_argument('--input_csv', type=str, help="Path  to the input csv", default="./data/federated_central.csv")
    args = parser.parse_args()
    print(args)

    upload_csv(**vars(args))

with open("DoneU.txt", "w") as file:
    file.write("Upload Process is Done")




#