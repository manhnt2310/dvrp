import glob
import os
import pandas as pd
import numpy as np
import math

# ======================
# Global Constants
# ======================
SERVICE_TIME        = 1.0           # phút phục vụ tại mỗi khách
TRUCK_SPEED         = 50.0 / 60.0   # đơn vị khoảng cách / phút (50 đv/h)
MAX_CAPACITY        = 500           # sức chứa tối đa mỗi xe
NEW_VEHICLE_PENALTY = 50.0          # (không dùng trong greedy nhưng giữ để đối chiếu)
PENALTY_FACTOR      = 100.0         # (không dùng trong greedy)
SERVED_BONUS        = 150.0         # (không dùng trong greedy)
DEMAND_BONUS        = 2.0           # (không dùng trong greedy)

# ======================
# Utils
# ======================
def euclidean_distance(x1,y1,x2,y2):
    return math.hypot(x1-x2, y1-y2)

# ======================
# Greedy Solver
# ======================
def greedy_dscvrptw(df, num_vehicles=6):
    N = len(df)
    served = set([0])              # depot index = 0
    unserved = set(range(1, N))
    total_vehicle_idle = 0.0
    total_customer_wait = 0.0

    # initialize 1 vehicle
    vehicles = [{
        'route': [0],
        'current_node': 0,
        'time': 0.0,
        'capacity': MAX_CAPACITY,
        'distance': 0.0
    }]

    def available_customers(t):
        return [i for i in unserved if df.at[i, 'time'] <= t]

    idx_v = 0
    while unserved:
        if idx_v >= len(vehicles):
            if len(vehicles) < num_vehicles:
                # spawn new xe at thời điểm xuất hiện sớm nhất
                t0 = min(df.at[i, 'time'] for i in unserved)
                vehicles.append({
                    'route': [0],
                    'current_node': 0,
                    'time': t0,
                    'capacity': MAX_CAPACITY,
                    'distance': 0.0
                })
            else:
                break

        veh = vehicles[idx_v]
        custs = available_customers(veh['time'])
        if not custs:
            # fast‑forward đến lần xuất hiện kế
            t_next = min(df.at[i, 'time'] for i in unserved)
            total_vehicle_idle += (t_next - veh['time'])
            veh['time'] = t_next
            custs = available_customers(veh['time'])
            if not custs:
                idx_v += 1
                continue

        # chọn khách khả dụng có deadline nhỏ nhất, tie-break by distance
        best, best_metric = None, (float('inf'), float('inf'))
        for i in custs:
            row = df.iloc[i]
            if row.demand > veh['capacity']:
                continue
            prev = df.iloc[veh['current_node']]
            d = euclidean_distance(prev.x, prev.y, row.x, row.y)
            depart = veh['time'] + d/TRUCK_SPEED
            arrival = max(depart, row.open)
            if arrival > row.close:
                continue
            metric = (row.close, d)
            if metric < best_metric:
                best_metric = metric
                best = i

        if best is None:
            idx_v += 1
            continue

        # phục vụ best
        row = df.iloc[best]
        prev = df.iloc[veh['current_node']]
        d = euclidean_distance(prev.x, prev.y, row.x, row.y)
        depart = veh['time'] + d/TRUCK_SPEED
        idle = max(0.0, row.open - depart)
        total_vehicle_idle += idle
        arrival = max(depart, row.open)
        customer_wait = max(0.0, arrival - row.time)
        total_customer_wait += customer_wait

        veh['time'] = arrival + SERVICE_TIME
        veh['capacity'] -= row.demand
        veh['distance'] += d
        veh['current_node'] = best
        veh['route'].append(best)

        unserved.remove(best)
        served.add(best)

    # quay về depot
    for veh in vehicles:
        if veh['current_node'] != 0:
            last = df.iloc[veh['current_node']]
            depot = df.iloc[0]
            d0 = euclidean_distance(last.x, last.y, depot.x, depot.y)
            veh['distance'] += d0
            veh['route'].append(0)

    return vehicles, total_vehicle_idle, total_customer_wait

# ======================
# Main: chạy thử k=6..10
# ======================
if __name__ == "__main__":
    input_dir   = "F:/KLTN/100_Cus"            # thư mục chứa nhiều .csv
    output_path = "F:/KLTN/code/output_greedy/greedy100_1.txt"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Lấy danh sách tất cả file .csv
    csv_files = glob.glob(os.path.join(input_dir, "*.csv"))
    if not csv_files:
        print(f"No CSV files found in {input_dir}")
        exit()

    # Mở file kết quả để ghi (ghi đè mỗi lần chạy)
    with open(output_path, "w", encoding="utf-8") as fout:
        for csv_path in csv_files:
            filename = os.path.basename(csv_path)
            fout.write(f"=== Results for {filename} ===\n")

            # Đọc và chuẩn hóa dữ liệu
            df = pd.read_csv(csv_path)
            for col in ['x','y','demand','open','close','time']:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            # Chạy greedy với k từ 6 đến 10
            for k in range(1, 11):
                vehicles, v_idle, c_wait = greedy_dscvrptw(df, num_vehicles=k)
                total_dist = sum(v['distance'] for v in vehicles)

                # Ghi summary
                fout.write(f"\n-- k = {k} (dùng {len(vehicles)} xe) --\n")
                fout.write(f"Total distance      = {total_dist:.2f}\n")
                fout.write(f"Vehicle total idle  = {v_idle:.2f}\n")
                fout.write(f"Customer total wait = {c_wait:.2f}\n")

                # Ghi chi tiết route từng xe
                for i, v in enumerate(vehicles, start=1):
                    fout.write(f"  Xe{i}: route={v['route']}, dist={v['distance']:.2f}\n")

            fout.write("\n" + ("="*40) + "\n\n")

    print(f"All results saved to {output_path}")