# ----------------------------------------------------------
############################################################
#                  Gate Assignment Model
#                    YIU Cho Yin James
#               cho-yin.yiu@connect.polyu.hk
#                    Student, ISE4014
#            The Hong Kong Polytechnic University
#                    12 December 2020
############################################################
# ----------------------------------------------------------

# ----------------------------------------------------------
############################################################
#                     program begins
############################################################
# ----------------------------------------------------------

# ----------------------------------------------------------
############################################################
#                 for importing libraries
############################################################
# ----------------------------------------------------------

# import libraries
import argparse
import pandas as pd
import numpy as np
from pulp import LpMinimize, LpProblem, lpSum, LpVariable, PULP_CBC_CMD


# ----------------------------------------------------------

# ----------------------------------------------------------
############################################################
#                     for arguments
############################################################
# ----------------------------------------------------------

# processing arguments
def process_command():
    # initialize parser
    parser = argparse.ArgumentParser()

    # adding optional argument
    # csv input
    parser.add_argument("-p", "--pax_flow_database", help="Passenger Flow Database")
    parser.add_argument("-b", "--baggage_flow_database", help="Baggage Flow Database")
    parser.add_argument("-d", "--pax_distance_matrix", help="Passenger Distance Matrix")
    parser.add_argument("-e", "--baggage_distance_matrix", help="Baggage Distance Matrix")

    # read arguments from command line
    return parser.parse_args()


# ----------------------------------------------------------

# ----------------------------------------------------------
############################################################
#                     for import csv
############################################################
# ----------------------------------------------------------

# read the passenger flow database
def read_pax_flow():
    # read using pandas
    df = pd.read_csv(args.pax_flow_database)

    # return the dataframe df
    return df


# ----------------------------------------------------------

# read the baggage flow database
def read_baggage_flow():
    # read using pandas
    df = pd.read_csv(args.baggage_flow_database)

    # return the dataframe df
    return df


# ----------------------------------------------------------

# read the passenger distance matrix
def read_pax_distance_matrix():
    # read using pandas
    df = pd.read_csv(args.pax_distance_matrix)

    # return the dataframe df
    return df


# ----------------------------------------------------------

# read the baggage distance matrix
def read_bag_distance_matrix():
    # read using pandas
    df = pd.read_csv(args.baggage_distance_matrix)

    # return the dataframe df
    return df


# ----------------------------------------------------------

# ----------------------------------------------------------
############################################################
#                 for objective function
############################################################
# ----------------------------------------------------------

def objective_general(df_01, df_02, df_03, df_04):
    # read pd to numpy
    pax_number_original, bag_number_original = df_01.to_numpy(), df_02.to_numpy()
    pax_distance_original = df_03.transpose().to_numpy()  # transpose is needed for multiplication
    bag_distance_original = df_04.transpose().to_numpy()  # transpose is needed for multiplication

    # for flight numbers and gate numbers
    columns_flight, columns_gate = list(df_01.transpose().to_numpy()[0]), list(pax_distance_original[0])
    pax_number, bag_number = np.delete(pax_number_original, 0, 1), np.ceil((np.delete(bag_number_original, 0, 1) / 5))
    pax_distance, bag_distance = np.delete(pax_distance_original, 0, 0), np.delete(bag_distance_original, 0, 0)

    # do matrix multiplication
    pax_number_distance_matrix = np.matmul(pax_number, pax_distance)
    bag_number_distance_matrix = np.matmul(bag_number, bag_distance)

    # for for loop
    rows, columns = len(pax_number_distance_matrix), len(pax_number_distance_matrix[0])

    # dummy array for defining array for summation
    x_i_j, x_i_j_with_pax_coefficient, x_i_j_with_bag_coefficient = [], [], []
    for i in range(rows):
        x_i, x_i_with_pax_coefficient, x_i_with_bag_coefficient = [], [], []
        for j in range(columns):
            lp_variable_name = "x_Flight_" + str(columns_flight[i]) + "_Gate_" + str(columns_gate[j])
            x = LpVariable(name=lp_variable_name, cat="Binary")
            x_i.append(x)
            x_i_with_pax_coefficient.append(pax_number_distance_matrix[i][j] * x)
            x_i_with_bag_coefficient.append(bag_number_distance_matrix[i][j] * x)
        x_i_j.append(x_i)
        x_i_j_with_pax_coefficient.append(x_i_with_pax_coefficient)
        x_i_j_with_bag_coefficient.append(x_i_with_bag_coefficient)

    # for getting flattened
    x_i_j_with_pax_coefficient_flattened = list(np.array(x_i_j_with_pax_coefficient).flatten())
    x_i_j_with_bag_coefficient_flattened = list(np.array(x_i_j_with_bag_coefficient).flatten())

    # case 1: passenger only [can be ignored]
    # model = objective_pax(rows, x_i_j, x_i_j_with_pax_coefficient_flattened)

    # case 2: passenger only with additional constraint [can be ignored]
    # model = objective_pax_additional_constraint(rows, x_i_j, x_i_j_with_pax_coefficient_flattened)

    # case 3: passenger with baggage
    model = objective_pax_with_baggage(rows, x_i_j, x_i_j_with_pax_coefficient_flattened,
                                       x_i_j_with_bag_coefficient_flattened)

    # solve the problem
    solver = PULP_CBC_CMD(msg=False)
    model.solve(solver)
    output_array = []

    # print result
    print("Result:")
    print(f"Total Walking Distance (Optimal): {model.objective.value()}")
    for var in model.variables():
        if var.value() == 1:
            flight, gate_assigned = var.name.partition('Flight_')[2].partition('_')[0], var.name.partition('Gate_')[2]
            print(f"Flight {flight}: Gate {gate_assigned}")
            output_array.append([flight, gate_assigned])

    return output_array


# ----------------------------------------------------------

# step 1 [can be ignored]
def objective_pax(rows, x_i_j, x_i_j_with_pax_coefficient_flattened):
    # define the model problem to be solved
    model = LpProblem(name="objective_pax", sense=LpMinimize)

    # for getting transpose
    x_i_j_transpose = np.array(x_i_j).transpose().tolist()

    # add objective function to the model
    model += lpSum(x_i_j_with_pax_coefficient_flattened)

    # add constraints to the model
    for i in range(rows):
        lp_constraint_name = "only_one_gate_" + str(i + 1)
        model += (lpSum(x_i_j[i]) == 1, lp_constraint_name)

    for i in range(rows):
        lp_constraint_name = "only_one_aircraft_" + str(i + 1)
        model += (lpSum(x_i_j_transpose[i]) == 1, lp_constraint_name)

    return model


# ----------------------------------------------------------

# step 2: gate 10 and 14 cannot accommodate the aircraft in flight 1 (CX505) [can be ignored]
def objective_pax_additional_constraint(rows, x_i_j, x_i_j_with_pax_coefficient_flattened):
    # define the model problem to be solved
    model = LpProblem(name="objective_pax_additional_constraint", sense=LpMinimize)

    # for getting transpose
    x_i_j_transpose = np.array(x_i_j).transpose().tolist()

    # add objective function to the model
    model += lpSum(x_i_j_with_pax_coefficient_flattened)

    # add constraints to the model
    for i in range(rows):
        lp_constraint_name = "only_one_gate_" + str(i + 1)
        model += (lpSum(x_i_j[i]) == 1, lp_constraint_name)

    for i in range(rows):
        lp_constraint_name = "only_one_aircraft_" + str(i + 1)
        model += (lpSum(x_i_j_transpose[i]) == 1, lp_constraint_name)

    # two additional criteria
    model += (x_i_j[0][2] == 0, "aircraft_for_f1_cannot_be_assigned_to_gate_10")
    model += (x_i_j[0][4] == 0, "aircraft_for_f1_cannot_be_assigned_to_gate_14")

    return model


# ----------------------------------------------------------

# step 3: baggage would also be considered
def objective_pax_with_baggage(rows, x_i_j, x_i_j_with_pax_coefficient_flattened, x_i_j_with_bag_coefficient_flattened):
    # define the model problem to be solved
    model = LpProblem(name="objective_pax_with_baggage", sense=LpMinimize)

    # for getting transpose
    x_i_j_transpose = np.array(x_i_j).transpose().tolist()

    # add objective function to the model
    model += (lpSum(x_i_j_with_pax_coefficient_flattened) + 3 * x_i_j_with_bag_coefficient_flattened)

    # add constraints to the model
    for i in range(rows):
        lp_constraint_name = "only_one_gate_" + str(i + 1)
        model += (lpSum(x_i_j[i]) == 1, lp_constraint_name)

    for i in range(rows):
        lp_constraint_name = "only_one_aircraft_" + str(i + 1)
        model += (lpSum(x_i_j_transpose[i]) == 1, lp_constraint_name)

    # additional constraints if Gate 11 and Gate 15 cannot be assigned to CX505 [can be omitted]
    # model += (x_i_j[0][3] == 0, "aircraft_for_f1_cannot_be_assigned_to_gate_11")
    # model += (x_i_j[0][5] == 0, "aircraft_for_f1_cannot_be_assigned_to_gate_15")

    return model


# ----------------------------------------------------------

# ----------------------------------------------------------
############################################################
#                     main functions
############################################################
# ----------------------------------------------------------

def main():
    df_01, df_02 = read_pax_flow(), read_baggage_flow()
    df_03, df_04 = read_pax_distance_matrix(), read_bag_distance_matrix()

    output_array = objective_general(df_01, df_02, df_03, df_04)
    df_output_array = pd.DataFrame(output_array, columns=['Flight', 'Gate'])
    df_output_array.to_csv('output_result.csv', index=False)


# ----------------------------------------------------------
############################################################
#                     main program
############################################################
# ----------------------------------------------------------

args = process_command()
if __name__ == '__main__':
    main()

# ----------------------------------------------------------
############################################################
#                   main program end
############################################################
# ----------------------------------------------------------

# ----------------------------------------------------------
############################################################
#                      program end
############################################################
# ----------------------------------------------------------
