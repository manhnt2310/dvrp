import os
import glob
import pandas as pd
import numpy as np
import math

# ======================
# Global Constants
# ======================
SERVICE_TIME        = 1.0
TRUCK_SPEED         = 50.0 / 60.0
MAX_CAPACITY        = 500

# ======================
# Utils
# ======================
def euclidean_distance(x1,y1,x2,y2):
    return math.hypot(x1-x2, y1-y2)

# ======================
# Greedy Solver (fixed k vehicles)
# ======================
def greedy_dscvrptw(df, num_vehicles=6):
    N = len(df)
    unserved = set(range(1, N))
    total_vehicle_idle = 0.0
    total_customer_wait = 0.0

    # <-- khởi tạo đúng num_vehicles xe ngay từ đầu -->
    vehicles = [{
        'route': [0],
        'current_node': 0,
        'time': 0.0,
        'capacity': MAX_CAPACITY,
        'distance': 0.0
    } for _ in range(num_vehicles)]

    def available_customers(t):
        return [i for i in unserved if df.at[i, 'time'] <= t]

    idx_v = 0
    # Chạy đến khi đã xét xong tất cả xe mà vẫn còn khách hoặc hết khách
    while unserved and idx_v < num_vehicles:
        veh = vehicles[idx_v]
        custs = available_customers(veh['time'])

        if not custs:
            # fast‑forward đến lần xuất hiện kế
            t_next = min(df.at[i, 'time'] for i in unserved)
            idle = t_next - veh['time']
            total_vehicle_idle += idle
            veh['time'] = t_next
            custs = available_customers(veh['time'])
            if not custs:
                idx_v += 1
                continue

        # Chọn khách khả dụng deadline nhỏ nhất, tie-break = distance
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
            # Xe này không thể phục vụ thêm -> qua xe tiếp theo
            idx_v += 1
            continue

        # Phục vụ khách 'best'
        row = df.iloc[best]
        prev = df.iloc[veh['current_node']]
        d = euclidean_distance(prev.x, prev.y, row.x, row.y)
        depart = veh['time'] + d/TRUCK_SPEED
        idle = max(0.0, row.open - depart)
        total_vehicle_idle += idle
        arrival = max(depart, row.open)
        customer_wait = max(0.0, arrival - row.time)
        total_customer_wait += customer_wait

        # Cập nhật trạng thái xe
        veh['time'] = arrival + SERVICE_TIME
        veh['capacity'] -= row.demand
        veh['distance'] += d
        veh['current_node'] = best
        veh['route'].append(best)

        unserved.remove(best)
        # Giữ idx_v để tiếp tục cho xe này (có thể tiếp tục phục vụ thêm)

    # Quay về depot với mỗi xe
    for veh in vehicles:
        if veh['current_node'] != 0:
            last = df.iloc[veh['current_node']]
            depot = df.iloc[0]
            d0 = euclidean_distance(last.x, last.y, depot.x, depot.y)
            veh['distance'] += d0
            veh['route'].append(0)

    return vehicles, total_vehicle_idle, total_customer_wait


# ======================
# Main: chạy trên toàn bộ folder, k=2..4, lưu vào text file
# ======================
if __name__=="__main__":
    input_dir   = "F:/KLTN/30_Customer"
    output_path = "F:/KLTN/code/output_greedy/greedy30_1.txt"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    csv_files = glob.glob(os.path.join(input_dir, "*.csv"))
    if not csv_files:
        print(f"No CSV files found in {input_dir}")
        exit()

    with open(output_path, "w", encoding="utf-8") as fout:
        for csv_path in csv_files:
            filename = os.path.basename(csv_path)
            fout.write(f"=== Results for {filename} ===\n")

            df = pd.read_csv(csv_path)
            for col in ['x','y','demand','open','close','time']:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            for k in range(2, 5):  # k = 2..4
                vehicles, v_idle, c_wait = greedy_dscvrptw(df, num_vehicles=k)
                total_dist = sum(v['distance'] for v in vehicles)

                fout.write(f"\n-- k = {k} (luôn dùng {k} xe) --\n")
                fout.write(f"Total distance      = {total_dist:.2f}\n")
                fout.write(f"Vehicle total idle  = {v_idle:.2f}\n")
                fout.write(f"Customer total wait = {c_wait:.2f}\n")
                for i, v in enumerate(vehicles, start=1):
                    fout.write(f"  Xe{i}: route={v['route']}, dist={v['distance']:.2f}\n")

            fout.write("\n" + ("="*50) + "\n\n")

    print(f"All results saved to {output_path}")
