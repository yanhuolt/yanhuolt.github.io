import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy import text
from datetime import datetime, timedelta
import mysql.connector
import json
from typing import Dict, Optional, Tuple, List
import time

# 字段映射表
field_map = {
    "1": {"top_temp": "24111", "mid_temp": "24112", "dust": "41106", "so2": "41107", "nox": "41108", "co": "41109",
          "hcl": "41110", "bag_pressure_diff": "37111", "o2_content": "41105", "carbon_dosage": "36111",
          "ammonia_slip": "34101"},
    "2": {"top_temp": "24211", "mid_temp": "24212", "dust": "41206", "so2": "41207", "nox": "41208", "co": "41209",
          "hcl": "41210", "bag_pressure_diff": "37211", "o2_content": "41205", "carbon_dosage": "36211",
          "ammonia_slip": "34201"},
    "3": {"top_temp": "24311", "mid_temp": "24312", "dust": "41306", "so2": "41307", "nox": "41308", "co": "41309",
          "hcl": "41310", "bag_pressure_diff": "37311", "o2_content": "41305", "carbon_dosage": "36311",
          "ammonia_slip": "34301"}
}

# # 监测参数字段映射
# MONITOR_FIELD_MAPPING = {
#     'bag_pressure_diff': '37111',  # 布袋除尘器压力
#     'o2_content': '41105',        # 氧含量
#     'carbon_dosage': '36111',     # 活性炭投加量
#     'ammonia_slip': '34101'       # 氨逃逸
# }


# 建立数据库连接
def create_db_connection():
    """创建数据库连接"""
    return mysql.connector.connect(
        host='rm-bp182r45h60392d0fjo.mysql.rds.aliyuncs.com',
        user='lianwei',
        password='Lianwei0907#',
        database='ai_huanbaogcs_ljfs'
    )


# 创建数据库连接引擎
def create_engine_connection():
    """创建SQLAlchemy引擎"""
    return create_engine(
        'mysql+pymysql://lianwei:Lianwei0907#@rm-bp182r45h60392d0fjo.mysql.rds.aliyuncs.com/ai_huanbaogcs_ljfs')




# 全局变量记录最后处理时间
last_processed_time = None
initial_load_completed = False  # 标记是否已完成初始数据加载

def get_last_processed_time():
    """从均值表中获取最后处理的时间"""
    try:
        engine = create_engine_connection()
        with engine.connect() as conn:
            # 首先检查表是否存在
            table_exists = conn.execute(text("""
                SELECT COUNT(*) FROM information_schema.tables 
                WHERE table_schema = 'ai_huanbaogcs_ljfs' 
                AND table_name = 'warning_data_test'
            """)).scalar() > 0

            if not table_exists:
                print("表 warning_data_test 不存在，请先创建表")
                return None

            # 检查表中是否有数据
            has_data = conn.execute(text("SELECT COUNT(*) FROM warning_data_test")).scalar() > 0

            if not has_data:
                print("表 warning_data_test 为空，将从头开始处理数据")
                return None

            # 表存在且有数据，获取最后时间
            result = conn.execute(
                text("SELECT MAX(start_time) as last_time FROM warning_data_test")).fetchone()

            if result and result[0]:
                # 直接返回datetime对象，不需要转换
                return result[0]
            return None

    except Exception as e:
        print(f"获取最后处理时间出错: {e}")
        return None
    finally:
        if 'engine' in locals():
            engine.dispose()


# ========== 数据处理函数 ==========
def fetch_data(table_name: str, flag: int = None,
               start_time: datetime = None, end_time: datetime = None) -> pd.DataFrame:
    """
    通用数据获取函数
    :param table_name: 表名 ('warning_mean_data' 或 'realtime_gk_data')
    :param flag: 数据标志 (1:小时均值, 2:日均值, 3:5分钟均值)
    :param start_time: 开始时间 (用于增量处理)
    :param end_time: 结束时间 (当前时间)
    :return: 包含查询结果的DataFrame
    """
    engine = create_engine_connection()
    try:
        with engine.connect() as conn:
            # 特殊处理realtime_gk_data表
            if table_name == 'realtime_gk_data':
                query = """
                    SELECT id, mn, furnace_id, status, data_time, data 
                    FROM realtime_gk_data
                    WHERE status = '工况正常'
                """
                params = {}

                # 添加时间条件
                if start_time is not None:
                    query += " AND data_time >= :start_time"
                    params['start_time'] = start_time

                if end_time is not None:
                    query += " AND data_time <= :end_time"
                    params['end_time'] = end_time

                query += " ORDER BY furnace_id, data_time"

                df = pd.read_sql(text(query), conn, params=params)
                print(f"从{table_name}获取到{len(df)}条实时数据")

                if df.empty:
                    print("没有找到正常工况的实时数据")
                    return df

                # 解析data字段中的JSON数据
                df['data'] = df['data'].apply(lambda x: json.loads(x) if isinstance(x, str) else x)

                # 提取需要的监测参数（按炉号分别处理）
                for _, row in df.iterrows():
                    furnace_id = str(row['furnace_id'])
                    if furnace_id in field_map:
                        for param_name, field_id in field_map[furnace_id].items():
                            df.at[_, param_name] = row['data'].get(field_id, 0)

                return df

            # 处理warning_mean_data表
            elif table_name == 'warning_mean_data':
                query = "SELECT * FROM warning_mean_data"
                conditions = []
                params = {}

                if flag is not None:
                    conditions.append("flag = :flag")
                    params['flag'] = flag

                if start_time is not None:
                    conditions.append("data_time >= :start_time")
                    params['start_time'] = start_time

                if end_time is not None:
                    conditions.append("data_time <= :end_time")
                    params['end_time'] = end_time

                if conditions:
                    query += " WHERE " + " AND ".join(conditions)

                query += " ORDER BY data_time"
                df = pd.read_sql(text(query), conn, params=params)

                # 转换时间字段
                if 'data_time' in df.columns:
                    df['data_time'] = pd.to_datetime(df['data_time'])

                return df

            else:
                print(f"不支持的表格名称: {table_name}")
                return pd.DataFrame()

    except Exception as e:
        print(f"从{table_name}获取数据出错: {e}")
        return pd.DataFrame()
    finally:
        engine.dispose()



# def calculate_corrected(ammonia_slip: float, o2_content: float) -> float:
#     """计算氨逃逸折算值"""
#     try:
#         # 确保所有输入都是有效数字
#         ammonia_slip = float(ammonia_slip) if ammonia_slip is not None else 0
#         o2_content = float(o2_content) if o2_content is not None else 0
#
#         # 防止除以零
#         denominator = (21 - o2_content)
#         if denominator <= 0:
#             return 0
#
#         return ammonia_slip * (21 - 11) / denominator
#     except Exception:
#         return 0  # 如果出现任何错误，返回0而不是None



# ========== 预警规则检查 ==========
def check_warning_rules(hourly_df: pd.DataFrame, daily_df: pd.DataFrame, five_min_df: pd.DataFrame):
    """检查预警规则"""
    warning_data = []
    engine = create_engine_connection()

    try:
        # 处理小时均值数据
        if not hourly_df.empty:
            for _, row in hourly_df.iterrows():
                data_time = row['data_time']
                end_time = data_time + timedelta(minutes=59, seconds=59)

                # 温度预警
                if row['tem_median_avg'] > 1300:
                    warning_data.append(create_warning_record(row, end_time, '预警', '炉膛温度过高'))
                elif row['tem_median_avg'] > 1200:
                    warning_data.append(create_warning_record(row, end_time, '预警', '炉膛温度偏高'))

                # 污染物浓度预警
                if row['dust_avg'] > 30:
                    warning_data.append(create_warning_record(row, end_time, '预警', '烟气中颗粒物(PM)浓度较高'))
                if row['nox_avg'] > 300:
                    warning_data.append(create_warning_record(row, end_time, '预警', '烟气中氮氧化物(NOx)浓度较高'))
                if row['so2_avg'] > 100:
                    warning_data.append(create_warning_record(row, end_time, '预警', '烟气中二氧化硫(SO₂)浓度较高'))
                if row['hcl_avg'] > 60:
                    warning_data.append(create_warning_record(row, end_time, '预警', '烟气中氯化氢(HCl)浓度较高'))
                if row['co_avg'] > 100:
                    warning_data.append(create_warning_record(row, end_time, '预警', '烟气中一氧化碳(CO)浓度较高'))

        # 处理日均值数据
        if not daily_df.empty:
            for _, row in daily_df.iterrows():
                data_time = row['data_time']
                end_time = data_time + timedelta(hours=23, minutes=59, seconds=59)

                # 污染物排放超标报警
                if row['dust_avg'] > 20:
                    warning_data.append(create_warning_record(row, end_time, '报警', '烟气中颗粒物(PM)排放超标'))
                if row['nox_avg'] > 250:
                    warning_data.append(create_warning_record(row, end_time, '报警', '烟气中氮氧化物(NOx)排放超标'))
                if row['so2_avg'] > 80:
                    warning_data.append(create_warning_record(row, end_time, '报警', '烟气中二氧化硫(SO₂)排放超标'))
                if row['hcl_avg'] > 50:
                    warning_data.append(create_warning_record(row, end_time, '报警', '烟气中氯化氢(HCl)排放超标'))
                if row['co_avg'] > 80:
                    warning_data.append(create_warning_record(row, end_time, '报警', '烟气中一氧化碳(CO)排放超标'))

        # 处理5分钟均值数据
        if not five_min_df.empty:
            for _, row in five_min_df.iterrows():
                data_time = row['data_time']
                end_time = data_time + timedelta(minutes=4, seconds=59)

                if row['tem_median_avg'] < 850:
                    warning_data.append(create_warning_record(row, end_time, '报警', '炉膛温度低于850℃'))

        # 保存预警数据
        save_warnings_to_db(engine, warning_data)

    except Exception as e:
        print(f"检查预警规则出错: {e}")
    finally:
        engine.dispose()


def create_warning_record(row: pd.Series, end_time: datetime,
                          warning_type: str, warning_event: str) -> dict:
    """创建预警记录字典"""
    return {
        'mn': row['mn'],
        'furnace_id': row['furnace_id'],
        'start_time': row['data_time'],
        'end_time': end_time,
        'warning_type': warning_type,
        'warning_event': warning_event
    }


def check_new_warning_rules(realtime_df: pd.DataFrame):
    """检查新增的6条预警规则"""
    warning_status = {}
    engine = create_engine_connection()

    try:
        if not realtime_df.empty:
            # 按炉号分组处理
            for furnace_id, group in realtime_df.groupby('furnace_id'):
                group = group.sort_values('data_time')
                mn = group.iloc[0]['mn']
                mn_furnace = f"{mn}_{furnace_id}"

                # 初始化预警状态
                if mn_furnace not in warning_status:
                    warning_status[mn_furnace] = {
                        'bag_pressure_high': None,
                        'bag_pressure_low': None,
                        'o2_high': None,
                        'o2_low': None,
                        'carbon_low': None,
                        'ammonia_high': None
                    }

                # 逐条检查数据
                for _, row in group.iterrows():
                    data_time = row['data_time']
                    data = row['data']

                    # 1. 布袋除尘器压力预警
                    check_pressure_warning(engine, warning_status, mn_furnace,
                                           row, data_time, data)

                    # 2. 氧含量预警
                    check_o2_warning(engine, warning_status, mn_furnace,
                                     row, data_time, data)

                    # 3. 活性炭投加量预警
                    check_carbon_warning(engine, warning_status, mn_furnace,
                                         row, data_time, data)

                    # 4. 氨逃逸预警
                    check_ammonia_warning(engine, warning_status, mn_furnace,
                                          row, data_time, data)

    except Exception as e:
        print(f"检查新增预警规则出错: {e}")
    finally:
        engine.dispose()


# ========== 具体预警检查函数 ==========
def check_pressure_warning(engine, warning_status, mn_furnace, row, data_time, data):
    """检查布袋除尘器压力预警"""
    furnace_id = str(row['furnace_id'])
    pressure = float(data.get(field_map[furnace_id]['bag_pressure_diff'], 0))

    # 高压预警
    if pressure > 2000:
        if warning_status[mn_furnace]['bag_pressure_high'] is None:
            warning_status[mn_furnace]['bag_pressure_high'] = (data_time, None)
            print(f"触发高压预警: {mn_furnace} 压力值: {pressure}")
    elif warning_status[mn_furnace]['bag_pressure_high'] is not None:
        start_time, _ = warning_status[mn_furnace]['bag_pressure_high']
        save_warnings_to_db(engine, [{
            'mn': row['mn'],
            'furnace_id': row['furnace_id'],
            'start_time': start_time,
            'end_time': data_time,
            'warning_type': '预警',
            'warning_event': '布袋除尘器压力损失偏高'
        }])
        warning_status[mn_furnace]['bag_pressure_high'] = None

    # 低压预警
    if pressure < 500:
        if warning_status[mn_furnace]['bag_pressure_low'] is None:
            warning_status[mn_furnace]['bag_pressure_low'] = (data_time, None)
            print(f"触发低压预警: {mn_furnace} 压力值: {pressure}")
    elif warning_status[mn_furnace]['bag_pressure_low'] is not None:
        start_time, _ = warning_status[mn_furnace]['bag_pressure_low']
        save_warnings_to_db(engine, [{
            'mn': row['mn'],
            'furnace_id': row['furnace_id'],
            'start_time': start_time,
            'end_time': data_time,
            'warning_type': '预警',
            'warning_event': '布袋除尘器压力损失偏低'
        }])
        warning_status[mn_furnace]['bag_pressure_low'] = None


def check_o2_warning(engine, warning_status, mn_furnace, row, data_time, data):
    """检查氧含量预警"""
    # o2_content = float(data.get(field_map['o2_content'], 0))
    furnace_id = str(row['furnace_id'])
    o2_content = float(data.get(field_map[furnace_id]['o2_content'], 0))

    # 氧含量偏高预警(>10%)
    if o2_content > 10:
        if warning_status[mn_furnace]['o2_high'] is None:
            warning_status[mn_furnace]['o2_high'] = (data_time, None)
            print(f"触发氧含量高预警: {mn_furnace} 氧含量: {o2_content}%")
    elif warning_status[mn_furnace]['o2_high'] is not None:
        start_time, _ = warning_status[mn_furnace]['o2_high']
        save_warnings_to_db(engine, [{
            'mn': row['mn'],
            'furnace_id': row['furnace_id'],
            'start_time': start_time,
            'end_time': data_time,
            'warning_type': '预警',
            'warning_event': '焚烧炉出口氧含量偏高'
        }])
        warning_status[mn_furnace]['o2_high'] = None
        print(f"结束氧含量高预警: {mn_furnace} 当前氧含量: {o2_content}%")

    # 氧含量偏低预警(<6%)
    if o2_content < 6:
        if warning_status[mn_furnace]['o2_low'] is None:
            warning_status[mn_furnace]['o2_low'] = (data_time, None)
            print(f"触发氧含量低预警: {mn_furnace} 氧含量: {o2_content}%")
    elif warning_status[mn_furnace]['o2_low'] is not None:
        start_time, _ = warning_status[mn_furnace]['o2_low']
        save_warnings_to_db(engine, [{
            'mn': row['mn'],
            'furnace_id': row['furnace_id'],
            'start_time': start_time,
            'end_time': data_time,
            'warning_type': '预警',
            'warning_event': '焚烧炉出口氧含量偏低'
        }])
        warning_status[mn_furnace]['o2_low'] = None
        print(f"结束氧含量低预警: {mn_furnace} 当前氧含量: {o2_content}%")


def check_carbon_warning(engine, warning_status, mn_furnace, row, data_time, data):
    """检查活性炭投加量预警"""
    # carbon_dosage = float(data.get(field_map['carbon_dosage'], 0))

    furnace_id = str(row['furnace_id'])
    carbon_dosage = float(data.get(field_map[furnace_id]['carbon_dosage'], 0))

    # 活性炭投加量不足预警(<2.0)
    if carbon_dosage < 2.0:
        if warning_status[mn_furnace]['carbon_low'] is None:
             warning_status[mn_furnace]['carbon_low'] = (data_time, None)
             print(f"触发活性炭不足预警: {mn_furnace} 投加量: {carbon_dosage}kg/h")
        elif warning_status[mn_furnace]['carbon_low'] is not None:
            start_time, _ = warning_status[mn_furnace]['carbon_low']
            save_warnings_to_db(engine, [{
        'mn': row['mn'],
        'furnace_id': row['furnace_id'],
        'start_time': start_time,
        'end_time': data_time,
        'warning_type': '预警',
        'warning_event': '活性炭投加量不足'
    }])
    warning_status[mn_furnace]['carbon_low'] = None
    print(f"结束活性炭不足预警: {mn_furnace} 当前投加量: {carbon_dosage}kg/h")


def check_ammonia_warning(engine, warning_status, mn_furnace, row, data_time, data):
    """检查氨逃逸预警"""
    # ammonia_slip = float(data.get(field_map['ammonia_slip'], 0))
    # o2_content = float(data.get(field_map['o2_content'], 0))

    furnace_id = str(row['furnace_id'])
    ammonia_slip = float(data.get(field_map[furnace_id]['ammonia_slip'], 0))

    # # 计算氨逃逸折算值
    # corrected_ammonia = calculate_corrected(ammonia_slip, o2_content)

    # 氨逃逸偏高预警(>8mg/m³)
    if ammonia_slip > 8:
        if warning_status[mn_furnace]['ammonia_high'] is None:
            warning_status[mn_furnace]['ammonia_high'] = (data_time, None)
            print(f"触发氨逃逸预警: {mn_furnace} 氨逃逸值: {ammonia_slip:.2f}mg/m³")
    elif warning_status[mn_furnace]['ammonia_high'] is not None:
        start_time, _ = warning_status[mn_furnace]['ammonia_high']
        save_warnings_to_db(engine, [{
            'mn': row['mn'],
            'furnace_id': row['furnace_id'],
            'start_time': start_time,
            'end_time': data_time,
            'warning_type': '预警',
            'warning_event': '氨逃逸偏高'
        }])
        warning_status[mn_furnace]['ammonia_high'] = None
        print(f"结束氨逃逸预警: {mn_furnace} 当前氨逃逸值: {ammonia_slip:.2f}mg/m³")




# ========== 数据库操作函数 ==========
def save_warnings_to_db(engine, warnings: List[dict]):
    """批量保存预警数据到数据库"""
    if not warnings:
        print("没有需要保存的预警数据")
        return

    try:
        with engine.connect() as conn:
            trans = conn.begin()
            try:
                # 检查哪些预警已存在
                existing_warnings = []
                for warning in warnings:
                    query = text("""
                        SELECT COUNT(*) FROM warning_data_test
                        WHERE mn = :mn AND furnace_id = :furnace_id 
                        AND start_time = :start_time AND warning_event = :warning_event
                    """)
                    result = conn.execute(query, {
                        'mn': warning['mn'],
                        'furnace_id': warning['furnace_id'],
                        'start_time': warning['start_time'],
                        'warning_event': warning['warning_event']
                    }).scalar()

                    if result == 0:
                        existing_warnings.append(warning)

                # 只插入不存在的预警记录
                if existing_warnings:
                    conn.execute(text("""
                        INSERT INTO warning_data_test
                        (mn, furnace_id, start_time, end_time, warning_type, 
                         warning_event, is_deleted, create_time, update_time)
                        VALUES 
                        (:mn, :furnace_id, :start_time, :end_time, :warning_type, 
                         :warning_event, 0, NOW(), NOW())
                    """), existing_warnings)

                    print(f"成功插入{len(existing_warnings)}条预警数据")
                else:
                    print("所有预警数据已存在，无需重复插入")

                trans.commit()
            except Exception as e:
                trans.rollback()
                print(f"保存预警数据出错: {e}")
                raise
    except Exception as e:
        print(f"数据库连接出错: {e}")


def check_ongoing_warning_exists(engine, mn, furnace_id, warning_event):
    """检查是否已存在相同的未结束预警记录"""
    try:
        with engine.connect() as conn:
            query = text("""
                SELECT COUNT(*) FROM warning_data_test
                WHERE mn = :mn AND furnace_id = :furnace_id 
                AND warning_event = :warning_event AND end_time IS NULL
            """)
            result = conn.execute(query, {
                'mn': mn,
                'furnace_id': furnace_id,
                'warning_event': warning_event
            }).scalar()
            return result > 0
    except Exception as e:
        print(f"检查持续预警出错: {e}")
        return False  # 出错时假定不存在，让上层逻辑处理


# ========== 主流程函数 ==========
def process_initial_data():
    """处理初始数据（全量数据）"""
    global initial_load_completed, last_processed_time
    print("开始处理初始数据...")

    current_time = datetime.now()
    hourly_df = fetch_data('warning_mean_data', flag=1, end_time=current_time)
    daily_df = fetch_data('warning_mean_data', flag=2, end_time=current_time)
    five_min_df = fetch_data('warning_mean_data', flag=3, end_time=current_time)
    realtime_df = fetch_data('realtime_gk_data', end_time=current_time)

    if not hourly_df.empty:
        check_warning_rules(hourly_df, daily_df, five_min_df)
    if not realtime_df.empty:
        check_new_warning_rules(realtime_df)

    # 更新最后处理时间为当前时间
    last_processed_time = current_time
    initial_load_completed = True
    print(f"初始数据处理完成，最后处理时间: {last_processed_time}")


def process_incremental_data():
    """处理增量数据"""
    global last_processed_time
    print("开始处理增量数据...")

    current_time = datetime.now()
    # last_time = get_last_processed_time()
    last_time = last_processed_time or get_last_processed_time()

    if last_time is None:
        process_initial_data()
        return

    hourly_df = fetch_data('warning_mean_data', flag=1, start_time=last_time, end_time=current_time)
    daily_df = fetch_data('warning_mean_data', flag=2, start_time=last_time, end_time=current_time)
    five_min_df = fetch_data('warning_mean_data', flag=3, start_time=last_time, end_time=current_time)
    realtime_df = fetch_data('realtime_gk_data', start_time=last_time, end_time=current_time)

    if not hourly_df.empty:
        check_warning_rules(hourly_df, daily_df, five_min_df)
    if not realtime_df.empty:
        check_new_warning_rules(realtime_df)

    last_processed_time = current_time
    print(f"增量数据处理完成，最后处理时间: {last_processed_time}")


# ========== 定时任务 ==========
def run_warning_check():
    """运行预警检查"""
    global last_processed_time
    print(f"{datetime.now()} 开始执行预警检查...")

    # 获取最后处理时间（优先使用内存中的值）
    last_time = last_processed_time or get_last_processed_time()

    if last_time is None:
        print("首次运行，执行全量数据处理")
        process_initial_data()
    else:
        print(f"增量处理，最后处理时间: {last_time}")
        process_incremental_data()

    print(f"{datetime.now()} 预警检查完成")


def scheduled_warning_check():
    """定时执行预警检查"""
    while True:
        try:
            run_warning_check()
        except Exception as e:
            print(f"执行预警检查出错: {e}")
        time.sleep(300)  # 5分钟执行一次
if __name__ == "__main__":
    # 启动定时任务
    print("启动定时预警检查任务，每5分钟执行一次...")
    scheduled_warning_check()