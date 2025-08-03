import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import math
from typing import List, Tuple
from io import StringIO
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

# --- Page config ---
st.set_page_config(page_title="EV Routing App", layout="wide")

# --- Sidebar ---
st.sidebar.title("EV Routing App")
mode = st.sidebar.radio("Select Input Mode", ["Use Test Case", "Upload Your Own File"])
uploaded_file = None

if mode == "Use Test Case":
    test_dir = "Test_Case_CSV_Files"
    test_files = sorted([f for f in os.listdir(test_dir) if f.endswith(".csv")])
    selected_file = st.sidebar.selectbox("Choose Test Case", ["-- Select a test case --"] + test_files)

    if selected_file != "-- Select a test case --":
        with open(os.path.join(test_dir, selected_file), "r") as f:
            uploaded_file = StringIO(f.read())
        st.sidebar.success(f"Loaded: {selected_file}")
    else:
        st.sidebar.info("Please select a test case file to proceed.")

elif mode == "Upload Your Own File":
    uploaded_file = st.sidebar.file_uploader("Upload EV routing CSV", type="csv")

st.title("Electric Vehicle Routing with Discharge Mode")

# --- Discharge mode toggle ---
discharge_mode = st.sidebar.checkbox("Enable Discharge Mode (optional)")

# --- Solver function ---
def solve_vrp(vehicle_count, capacity, depot, customers, distance_matrix, discharge_mode=False):
    all_points = [(depot["X"], depot["Y"])] + list(zip(customers["X"], customers["Y"]))
    demands = [0] + list(customers["Demand"])

    penalties = [0]
    for _, row in customers.iterrows():
        if discharge_mode and row["NodeType"].lower() == "dischargestation":
            penalties.append(-10)  # Negative reward
        else:
            penalties.append(0)

    manager = pywrapcp.RoutingIndexManager(len(all_points), vehicle_count, 0)
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_idx, to_idx):
        from_node = manager.IndexToNode(from_idx)
        to_node = manager.IndexToNode(to_idx)
        return int(distance_matrix[from_node][to_node])

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    def demand_callback(from_idx):
        from_node = manager.IndexToNode(from_idx)
        return int(demands[from_node])

    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index, 0, [capacity] * vehicle_count, True, "Capacity"
    )

    if discharge_mode:
        for node in range(1, len(all_points)):
            penalty = penalties[node]
            if penalty < 0:
                routing.AddDisjunction([manager.NodeToIndex(node)], -penalty)

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC

    solution = routing.SolveWithParameters(search_parameters)

    output = []
    if solution:
        for vehicle_id in range(vehicle_count):
            index = routing.Start(vehicle_id)
            route = []
            while not routing.IsEnd(index):
                route.append(manager.IndexToNode(index))
                index = solution.Value(routing.NextVar(index))
            route.append(0)
            if len(route) > 2:
                output.append((vehicle_id, route))
    return output

# --- File Handling ---
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)

        # Check required columns
        expected_cols = {"ID", "X", "Y", "Demand", "NodeType"}
        if not expected_cols.issubset(set(df.columns)):
            st.error(f"CSV must include columns: {expected_cols}")
        else:
            depot = df[df["NodeType"].str.lower() == "depot"].iloc[0]
            customers = df[df["NodeType"].str.lower() != "depot"]

            all_points = [(depot["X"], depot["Y"])] + list(zip(customers["X"], customers["Y"]))
            n = len(all_points)
            distance_matrix = [
                [math.hypot(x1 - x2, y1 - y2) for x2, y2 in all_points]
                for x1, y1 in all_points
            ]

            vehicle_count = 3
            capacity = 100

            solution = solve_vrp(vehicle_count, capacity, depot, customers, distance_matrix, discharge_mode)

            st.success(f"Routing complete using {vehicle_count} vehicle(s)")
            for v_id, route in solution:
                st.markdown(f"**Vehicle {v_id + 1} Route:**")
                route_labels = []
                for idx in route:
                    if idx == 0:
                        route_labels.append("Depot")
                    else:
                        row = customers.iloc[idx - 1]
                        if row["NodeType"].lower() == "dischargestation":
                            route_labels.append(f"{row['ID']} (Discharge)")
                        else:
                            route_labels.append(f"{row['ID']} (Customer)")
                st.write(" → ".join(route_labels))

    except Exception as e:
        st.error(f"Error reading file: {e}")

else:
    st.info("Please upload a CSV file or choose a test case to begin.")

# --- Footer ---
st.markdown("---")
st.markdown(
    """
    <div style="text-align:center;">
        <strong>Developed by Team-Mind_Mesh</strong><br>
        Developers: Amrutha D , Vishnu V<br>
        College: REVA University, Bangalore<br>
        Contact: vishnuv2309@gmail.com <br>
    </div>
    """,
    unsafe_allow_html=True
)
st.markdown("""
    <style>
        footer {visibility: hidden;}
        .stApp {bottom: 0;}
    </style>
""", unsafe_allow_html=True)
