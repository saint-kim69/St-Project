from typing import Any

import pymongo



class MongoDBManager:
    def __init__(self, database_name, collection_name):
        mongo_dns = ''
        self.client = pymongo.MongoClient(mongo_dns)
        self.db = self.client[database_name]
        self.collection = self.db[collection_name]

    def save(self, rows: list[dict[str, Any]]):
        self.collection.insert_many(rows)

    def filter(self, **kwargs) -> list[dict[str, Any]]:
        return self.collection.find(kwargs)
    
    def __enter__(self):
        return self
    

    def __exit__(self):
        self.client.close()