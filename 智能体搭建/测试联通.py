import pymysql
from faker import Faker
from datetime import datetime, timedelta
import random

# ---------- 1. 连接 ----------
conn = pymysql.connect(
    host='192.168.0.109',
    port=3306,
    user='root',
    passwd='lianwei123',
    db='lianwei_agent',
    charset='utf8mb4'
)
cur = conn.cursor()

fake = Faker('zh_CN')