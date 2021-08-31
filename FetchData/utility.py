from .constant import *
from lxml import etree
import requests
from functools import reduce
import random
import time


def get_date_string():
    """
    获得形如20161010这样的年月日字符串
    :return:
    """
    current = time.localtime()
    return time.strftime("%Y%m%d", current)


def create_headers():
    headers = dict()
    headers["User-Agent"] = random.choice(USER_AGENTS)
    headers["Referer"] = f"http://www.lianjia.com"
    return headers


class ZuFang(object):
    def __init__(self, *args):
        # # COLUMNS and # args must be equal
        assert len(DATA_COLUMNS) == len(args)
        self.info = dict(zip(DATA_COLUMNS, args))

    def text(self):
        return reduce(lambda x, y: str(x)+'\t'+str(y), self.info.values())




