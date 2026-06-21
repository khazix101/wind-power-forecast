import glob as py_glob

import xarray as xr
import pandas as pd
import numpy as np

def era5_instant_to_wind_csv(nc_paths=None, output_csv="data/wind_nc/output/wind_data.csv",
                              noise_std=None):
    """
    从多个 ERA5 instant.nc 文件提取风电变量，合并导出到一个 CSV
    自动计算风速、温度转摄氏度、空气密度
    限定5个指定经纬度 + 时间戳向前8小时 + 编号1-5

    Parameters
    ----------
    noise_std : dict or None
        为各变量添加高斯噪声，格式 {变量名: 标准差}。
        例如 noise_std={"u100": 0.1, "v100": 0.1, "t2m": 0.5, "sp": 10.0, "blh": 10.0}
        None 表示不添加噪声。
    """
    # ===================== 固定5个坐标点 =====================
    target_points = {
        1: (41, 96),
        2: (40.5, 96.5),
        3: (40, 97),
        4: (39.5, 97.5),
        5: (39,98)
    }

    # ===================== 自动发现所有 .nc 文件 =====================
    if nc_paths is None:
        nc_paths = sorted(py_glob.glob("data/wind_nc/*.nc"))
    if not nc_paths:
        print("错误：未找到任何 .nc 文件")
        return None
    print(f"找到 {len(nc_paths)} 个 NetCDF 文件：{nc_paths}")

    # ===================== 逐文件处理 =====================
    all_frames = []
    for i, nc_path in enumerate(nc_paths, 1):
        print(f"\n===== 正在处理第 {i}/{len(nc_paths)} 个文件：{nc_path} =====")
        ds = xr.open_dataset(nc_path)

        # 时间戳修正：向前8小时
        ds["valid_time"] = ds["valid_time"] + pd.Timedelta(hours=8)

        # 提取风电核心变量
        vars_needed = ["u100", "v100", "t2m", "sp","blh"]
        available_vars = [v for v in vars_needed if v in ds.data_vars]
        print(f"提取到风电变量：{available_vars}")
        ds = ds[available_vars]

        # 筛选指定5个坐标点 + 编号
        for point_id, (lat, lon) in target_points.items():
            point_ds = ds.sel(latitude=lat, longitude=lon, method="nearest")
            df_point = point_ds.to_dataframe().reset_index()
            df_point["point_id"] = point_id
            all_frames.append(df_point)

        ds.close()

    # 合并所有文件的所有点
    df = pd.concat(all_frames, ignore_index=True)

    # 去除重复行（同一时刻、同一地点出现多次）
    df = df.drop_duplicates(subset=["point_id", "valid_time"]).reset_index(drop=True)

    # ── 在派生计算之前对原始 ERA5 变量添加高斯噪声 ──
    if noise_std is not None:
        rng = np.random.default_rng(42)
        for var, std in noise_std.items():
            if var in df.columns:
                noise = rng.normal(0, std, size=len(df))
                df[var] = df[var] + noise
                print(f"已对 {var} 添加高斯噪声（std={std}）")
            else:
                print(f"警告：变量 {var} 不存在，跳过噪声添加")

    # 4. 计算100m风速
    if "u100" in df.columns and "v100" in df.columns:
        df["wind_speed_100m"] = np.sqrt(df["u100"]**2 + df["v100"]**2)
        df["wind_dir_100m"] = np.mod(270-np.rad2deg(np.arctan2(df["u100"],df["v100"])),360) #北风0°/360°
        print("已计算100m风速 wind_speed_100m")
        print("已计算100m风速 wind_dir_100m")

    # 5. 温度转摄氏度
    if "t2m" in df.columns:
        df["t2m_degC"] = df["t2m"] - 273.15
        print("已将温度转换为摄氏度 t2m_degC")

    # 6. 气压处理（Pa -> hPa）
    if "sp" in df.columns:
        df["sp_hPa"] = df["sp"] / 100.0
        print("已将气压转换为 hPa：sp_hPa")

        # ===================== 空气密度计算 =====================
        # rho = p / (R * T)
        # p: Pa
        # T: K
        # R: 干空气气体常数 287.05 J/(kg·K)

        if "t2m" in df.columns:
            R = 287.05
            df["air_density"] = df["sp"] / (R * df["t2m"])
            print("已计算空气密度 air_density")

    # 7. BLH（边界层高度）处理
    if "blh" in df.columns:

        # ERA5中blh单位为 m
        # 防止异常负值
        df["blh"] = df["blh"].clip(lower=0)

    # ===================== 理论风电功率计算 (Vestas V90-2.0MW) =====================
    def power_curve_v90(v_hub):
        curve = np.array([
            [0, 0], [1, 0], [2, 0], [3, 0], [4, 35], [5, 80],
            [6, 150], [7, 260], [8, 410], [9, 610], [10, 870],
            [11, 1180], [12, 1540], [13, 1850], [14, 1970],
            [15, 2000], [16, 2000], [17, 2000], [18, 2000],
            [19, 2000], [20, 2000], [21, 2000], [22, 2000],
            [23, 2000], [24, 2000], [25, 2000], [26, 0],
        ], dtype=float)
        return np.interp(v_hub, curve[:, 0], curve[:, 1])

    def wind_at_hub(v100, z_ref=100, z_hub=90, z0=0.03):
        return v100 * (np.log(z_hub / z0) / np.log(z_ref / z0))

    def compute_power(v100, rho, rho_ref=1.225):
        return power_curve_v90(wind_at_hub(v100)) * (rho / rho_ref)

    if "wind_speed_100m" in df.columns and "air_density" in df.columns:
        df["power_kW"] = compute_power(df["wind_speed_100m"].values, df["air_density"].values)
        print("已计算风电功率 power_kW (Vestas V90-2.0MW)")


    # ===================== 时间排序 =====================
    df = df.sort_values(
        by=["point_id", "valid_time"]
    ).reset_index(drop=True)

    # ===================== 字段重排列 =====================
    preferred_order = [
        "valid_time",
        "point_id",
        "latitude",
        "longitude",

        "u100",
        "v100",
        "wind_speed_100m",
        "wind_dir_100m",

        "t2m",
        "t2m_degC",

        "sp",
        "sp_hPa",
        "air_density",

        "blh",
        "blh_log",

        "power_kW",
    ]

    # 只保留实际存在列
    final_cols = [c for c in preferred_order if c in df.columns]
    remaining_cols = [c for c in df.columns if c not in final_cols]

    df = df[final_cols + remaining_cols]

    # 8. 导出CSV
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    print(f"\n风电数据已成功导出：{output_csv}")
    print(f"数据维度：{df.shape[0]} 行 × {df.shape[1]} 列")

    # ===================== 数据概览 =====================
    print("\n===== 数据预览 =====")
    print(df.head())

    print("\n===== 缺失值统计 =====")
    print(df.isnull().sum())

    return df

if __name__ == "__main__":
    era5_instant_to_wind_csv(noise_std={
        "u100": 0.1,   # m/s，风速分量噪声
        "v100": 0.1,   # m/s
        "t2m": 0.5,    # K，温度噪声
        "sp": 10.0,    # Pa，气压噪声
        "blh": 10.0,   # m，边界层高度噪声
    })