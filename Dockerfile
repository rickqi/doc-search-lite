FROM python:3.11-slim

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Install project
COPY . .
RUN pip install --no-cache-dir -e ".[dev]"

# Create sample knowledge base
RUN mkdir -p /data/sample /data/raw/sample/index
RUN python -c "
import os
docs = {
    'annual-leave.md': '''# 年假管理制度

## 适用范围
本制度适用于公司全体员工。

## 年假天数
- 工龄1-5年：5天
- 工龄5-10年：10天
- 工龄10年以上：15天

## 申请流程
1. 员工通过OA系统提交年假申请
2. 部门经理审批
3. HR备案
''',
    'travel-reimbursement.md': '''# 差旅报销标准

## 住宿标准
- 一线城市：500元/晚
- 二线城市：350元/晚
- 其他城市：250元/晚

## 交通标准
- 飞机：经济舱
- 高铁：二等座
- 出租车：实报实销

## 餐饮补贴
- 早餐：20元
- 午餐：50元
- 晚餐：50元
''',
    'data-protection.md': '''# 数据保护管理规定

## 数据分类
- 公开数据：对外公开的信息
- 内部数据：仅限内部使用
- 敏感数据：需严格授权

## 安全要求
1. 所有敏感数据必须加密存储
2. 数据传输使用HTTPS/TLS
3. 定期进行安全审计
4. 数据访问需留日志
''',
    'remote-work.md': '''# 远程办公管理规定

## 适用范围
适用于经审批同意远程办公的员工。

## 工作要求
1. 保持通讯畅通（企业微信/钉钉）
2. 每日提交工作日报
3. 按时参加线上会议
4. 确保工作环境安全

## 设备管理
公司提供笔记本电脑，员工需妥善保管。
''',
}
os.makedirs('/data/raw/sample', exist_ok=True)
for name, content in docs.items():
    with open(f'/data/raw/sample/{name}', 'w', encoding='utf-8') as f:
        f.write(content)
"

# Build index
RUN python -m src.cli build-index /data/raw/sample

EXPOSE 7860

# Start API server with sample index
CMD ["python", "-m", "src.api", "--host", "0.0.0.0", "--port", "7860"]
