import sqlite3
import os
import pickle
import uuid

class ObjectStore:
    def __init__(self, db_path='objects.db'):
        """
        :param db_path: 数据库存储文件的路径
        """
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """初始化数据库，如果表不存在则创建。"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS objects (
                uuid TEXT PRIMARY KEY,
                data BLOB
            )
        ''')
        conn.commit()
        conn.close()

    def add_object(self, uuid_str, obj):
        """
        存储（或更新）一个对象到数据库中。
        :param uuid_str: 作为主键的 UUID 字符串
        :param obj: Python 对象，方法内部会使用 pickle 序列化
        """
        data_blob = pickle.dumps(obj)
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''
            INSERT OR REPLACE INTO objects (uuid, data) VALUES (?, ?)
        ''', (uuid_str, data_blob))
        conn.commit()
        conn.close()

    def get_object(self, uuid_str):
        """
        通过 UUID 从数据库中读取对应对象。
        :param uuid_str: UUID 字符串
        :return: 反序列化后的 Python 对象，如果不存在则返回 None
        """
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('SELECT data FROM objects WHERE uuid = ?', (uuid_str,))
        row = c.fetchone()
        conn.close()
        if row:
            return pickle.loads(row[0])
        return None

    def delete_object(self, uuid_str):
        """
        通过 UUID 删除数据库中的对象。
        :param uuid_str: UUID 字符串
        :return: True 表示删除成功，False 表示未找到
        """
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('DELETE FROM objects WHERE uuid = ?', (uuid_str,))
        rowcount = c.rowcount
        conn.commit()
        conn.close()
        return rowcount > 0


# -------------------------
# 下面是一个简单的测试示例
# -------------------------
if __name__ == "__main__":
    # 初始化对象存储
    store = ObjectStore(db_path="test_objects.db")

    # 生成一个新的 UUID 并准备一个测试对象
    my_uuid = str(uuid.uuid4())
    my_object = {"name": "Alice", "age": 30, "hobbies": ["reading", "sports"]}

    # 1) 存储该对象
    store.add_object(my_uuid, my_object)
    print(f"对象已保存, UUID = {my_uuid}")

    # 2) 读取并验证
    retrieved_obj = store.get_object(my_uuid)
    print("读取到的对象:", retrieved_obj)

    # 3) 删除对象
    deleted = store.delete_object(my_uuid)
    print("删除结果:", "成功" if deleted else "失败")

    # 4) 再次尝试读取
    should_be_none = store.get_object(my_uuid)
    print("再次读取结果:", should_be_none)

    # 结束
    print("测试完成！")
