from bs4 import BeautifulSoup
from FetchData.utility import *
import threadpool
import threading
import random
import requests
import skimage.io as io
import time
import json
import re
import os


class ZuFangBaseSpider:
    def __init__(self, city: str):
        self.city = city
        self.base_url = f'http://{self.city}.lianjia.com/'
        # 准备日期信息，爬到的数据存放到日期相关文件夹下
        self.date_string = get_date_string()
        print('Today date is: %s' % self.date_string)
        self.data_path = f'data/{self.date_string}'
        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)
            os.makedirs(f'{self.data_path}/images')

        self.total_num = 0  # 总的小区个数，用于统计
        print(f"Target site is {self.base_url}")
        self.mutex = threading.Lock()  # 创建锁
        self.chinese_area_dict = {}
        self.area_dict = {}
        self.chinese_city_district_dict = {}
        self.thread_pool_size = 50

    def collect_area_zufang_data(self, area_name: str, fmt="csv") -> None:
        """
        对于每个板块,获得这个板块下所有出租房的信息
        并且将这些信息写入文件保存
        :param city_name: 城市
        :param area_name: 板块
        :param fmt: 保存文件格式
        :return: None
        """
        district_name = self.area_dict.get(area_name, "")
        csv_file = f'{self.data_path}/{district_name}_{area_name}.tsv'
        if os.path.exists(f'{self.data_path}/{district_name}_{area_name}.tsv'):
            with open(f'{self.data_path}/{district_name}_{area_name}.tsv', 'r') as f:
                cur_text = f.read()
            if len(cur_text) > 20:
                print("Already crawled area: " + area_name + ", data saved at : " + csv_file)
                return

        with open(csv_file, "w") as f:
            # 开始获得需要的板块数据
            zufangs = self.get_area_zufang_info(area_name)
            # 锁定
            if self.mutex.acquire(1):
                self.total_num += len(zufangs)
                # 释放
                self.mutex.release()
            if fmt == "csv":
                f.write('fetch date' + "\t" + '\t'.join(DATA_COLUMNS) + "\n")
                for zufang in zufangs:
                    f.write(self.date_string + "\t" + zufang.text() + "\n")
        print("Finish crawl area: " + area_name + ", save data to : " + csv_file)

    def fetch_images(self, images_bar: [BeautifulSoup], house_id: str) -> [str]:
        images_urls = []
        if images_bar:
            images = []
            for bar in images_bar:
                img_list = bar.find_all('img')
                if img_list:
                    images.extend(img_list)
            if not images:
                # no picture at all
                return images_urls
            images_urls = list(map(lambda x: (x.attrs['data-src'], x.attrs['data-name']), images))

            if not os.path.exists(f'{self.data_path}/images/{house_id}'):
                os.mkdir(f'{self.data_path}/images/{house_id}')
            else:
                print(f'Images for {house_id} already downloaded!')
                return images_urls
            # # todo :remove!!!!
            # return images_urls

            # fetching images
            print(f'Downloading Images {house_id} !')
            for idx, (img_url, room_type) in enumerate(images_urls):
                try:
                    I = io.imread(img_url)
                    io.imsave(f'{self.data_path}/images/{house_id}/{idx}_{room_type}.png', I)
                except Exception as e:
                    print(e)
        return images_urls

    @staticmethod
    def fetch_house_information(house_information_dom: BeautifulSoup) -> dict:
        info_dict = {}
        if house_information_dom:
            house_information = house_information_dom.findChild('ul')
            if house_information:
                info_list = house_information.findAll('li')
                info_text = list(map(lambda x: x.text, info_list))

                for i_t in info_text:
                    if '：' in i_t:
                        name, value = i_t.split('：')
                        info_dict[name] = value
        return info_dict

    @staticmethod
    def fetch_facilities(facility_information_dom: BeautifulSoup) -> dict:
        facility_dict = {}
        if facility_information_dom:
            info_list = facility_information_dom.findAll('li')[1:]
            for d_e in info_list:
                facility_dict[d_e.text.strip()] = 'facility_no' not in d_e.attrs['class']
        return facility_dict

    @staticmethod
    def house_description(house_description_dom: BeautifulSoup) -> str:
        house_description = 'Not Found'
        if house_description_dom:
            desc_dom = house_description_dom.find('p')
            if desc_dom:
                house_description = desc_dom.text
            house_description = re.sub('\t*', '', house_description)
            house_description = re.sub('\n*', ' ', house_description)
        return house_description.strip()

    @staticmethod
    def fetch_agent(agent_name_dom: BeautifulSoup) -> str:
        agent_name = 'Not Found'
        if agent_name_dom:
            agent_name = agent_name_dom.text if agent_name_dom else ''
        return agent_name

    @staticmethod
    def fetch_recommended(house_id: str) -> (list, list):
        recommend_request_url = f'https://sh.lianjia.com/zufang/aj/house/similarRecommend?house_code={house_id}&city_id=310000&has_nearby=1'
        recommend_res = requests.get(recommend_request_url)
        recommend_res_data = recommend_res.json()
        nearby_house_id = []
        recommend_house_id_geo = []
        if recommend_res.status_code == 200:
            if 'nearby_house' in recommend_res_data['data']:
                nearby_house_id = list(map(lambda x: x['house_code'], recommend_res_data['data']['nearby_house']))
            # save latitude and longitude here
            if 'recommend_list' in recommend_res_data['data']:
                recommend_house_id_geo = list(map(lambda x: (x['code'], x['geolatitudelongitude']),
                                                  recommend_res_data['data']['recommend_list']))
        return nearby_house_id, recommend_house_id_geo

    @staticmethod
    def fetch_tags(tags_dom: BeautifulSoup) -> [str]:
        tag_list = []
        if tags_dom:
            img_dom = tags_dom.findAll('img')
            if img_dom:
                for t_image in img_dom:
                    tag_list.append(t_image.attrs['alt'])

            text_dom = tags_dom.findAll('i')
            if text_dom:
                for t_text in text_dom:
                    tag_list.append(t_text.text)
        return tag_list

    @staticmethod
    def fetch_traffic(traffic_dom: BeautifulSoup) -> dict:
        subway_info = {}
        dom = traffic_dom.findAll('ul')
        if dom:
            traffic_info_dom = traffic_dom.findAll('ul')[-1]
            for subway in traffic_info_dom.findAll('li'):
                station = subway.findAll('span')[0].text
                distance = subway.findAll('span')[1].text
                subway_info[station] = distance
        return subway_info

    def fetch_one_house(self, chinese_district: str, chinese_area: str, page: str,
                        house_elem: BeautifulSoup) -> ZuFang or None:
        headers = create_headers()
        try:
            price = house_elem.find('span', class_="content__list--item-price")
            desc1 = house_elem.find('p', class_="content__list--item--title")
            desc2 = house_elem.find('p', class_="content__list--item--des")
            detail_url_suffix = house_elem.find('a', class_="twoline").attrs['href']
            detail_url = self.base_url + detail_url_suffix
            house_id = detail_url_suffix.split('/')[-1].split('.')[0]

            res_detail = requests.get(detail_url, timeout=10, headers=headers)
            html_detail = res_detail.content
            soup_detail = BeautifulSoup(html_detail, "lxml")

            images_bar = soup_detail.findAll('div', class_='content__article__slide__item')
            images_urls = self.fetch_images(images_bar[1:], house_id)

            house_information_dom = soup_detail.find('div', class_='content__article__info')
            info_dict = self.fetch_house_information(house_information_dom)

            facility_information_dom = soup_detail.find('ul', class_='content__article__info2')
            facility_dict = self.fetch_facilities(facility_information_dom)

            house_description_dom = soup_detail.find('div', class_='content__article__info3', id='desc')
            house_description = self.house_description(house_description_dom)

            agent_name_dom = soup_detail.find('span', class_='contact_name')
            agent_name = self.fetch_agent(agent_name_dom)

            nearby_house_id, recommend_house_id_geo = self.fetch_recommended(house_id)

            tags_dom = soup_detail.find('p', class_='content__aside--tags')
            tag_list = self.fetch_tags(tags_dom)

            traffic_dom = soup_detail.find('div', class_='content__article__info4')

            subway_info = self.fetch_traffic(traffic_dom)

            price = price.text.strip().replace(" ", "").replace("元/月", "")
            # print(price)
            desc1 = desc1.text.strip().replace("\n", "")
            desc2 = desc2.text.strip().replace("\n", "").replace(" ", "")
            # print(desc1)

            infos = desc1.split(' ')
            xiaoqu = infos[0]
            layout = infos[1]
            descs = desc2.split('/')
            # print(descs[1])
            size = descs[1].replace("㎡", "平米")

            print("{0} {1} {2} {3} {4} {5} {6}".format(
                chinese_district, chinese_area, xiaoqu, layout, size, price))

            zufang = ZuFang(chinese_district, chinese_area, xiaoqu, layout, size, price, detail_url, house_id,
                            json.dumps(images_urls, ensure_ascii=False), json.dumps(info_dict, ensure_ascii=False),
                            json.dumps(facility_dict, ensure_ascii=False),
                            house_description, agent_name, json.dumps(recommend_house_id_geo),
                            json.dumps(nearby_house_id), json.dumps(tag_list, ensure_ascii=False),
                            json.dumps(subway_info, ensure_ascii=False))
        except Exception as e:
            # raise e
            print("=" * 20 + " page no data")
            print(e)
            print(page)
            print("=" * 20)
            zufang = None
        return zufang

    def get_area_zufang_info(self, area_name: str) -> [ZuFang]:
        area_dict = self.area_dict
        chinese_area_dict = self.chinese_area_dict

        matches = None
        """
        通过爬取页面获取城市指定版块的租房信息
        :param city_name: 城市
        :param area_name: 版块
        :return: 出租房信息列表
        """
        total_page = 1
        district_name = area_dict.get(area_name, "")
        chinese_district = self.chinese_city_district_dict[district_name]
        chinese_area = chinese_area_dict.get(area_name, "")
        zufang_list = list()
        page = f'{self.base_url}/zufang/{area_name}/'
        print(page)

        headers = create_headers()
        response = requests.get(page, timeout=10, headers=headers)
        html = response.content
        soup = BeautifulSoup(html, "lxml")

        # 获得总的页数
        try:
            page_box = soup.find_all('div', class_='content__pg')[0]
            # print(page_box)
            matches = re.search('.*data-totalpage="(\d+)".*', str(page_box))
            total_page = int(matches.group(1))
            # print(total_page)
        except Exception as e:
            print("\tWarning: only find one page for {0}".format(area_name))
            print(e)

        # 从第一页开始,一直遍历到最后一页
        headers = create_headers()
        for num in range(1, total_page + 1):
            page = f'{self.base_url}/zufang/{area_name}/pg{num}'
            print(page)
            response = requests.get(page, timeout=10, headers=headers)
            html = response.content
            soup = BeautifulSoup(html, "lxml")

            # 获得有小区信息的panel
            # if SPIDER_NAME == "lianjia":
            #     ul_element = soup.find('ul', class_="house-lst")
            #     house_elements = ul_element.find_all('li')
            # else:
            #     ul_element = soup.find('div', class_="content__list")
            #     house_elements = ul_element.find_all('div', class_="content__list--item")
            ul_element = soup.find('div', class_="content__list")
            if not ul_element:
                continue
            house_elements = ul_element.find_all('div', class_="content__list--item")

            if len(house_elements) == 0:
                continue
            # else:
            #     print(len(house_elements))

            for house_elem in house_elements:
                re_data = self.fetch_one_house(chinese_district, chinese_area, page, house_elem)
                if re_data:
                    zufang_list.append(re_data)
        return zufang_list

    def get_districts(self):
        """
        获取各城市的区县中英文对照信息
        :param city: 城市
        :return: 英文区县名列表
        """
        url = f'{self.base_url}/xiaoqu/'
        headers = create_headers()
        response = requests.get(url, timeout=10, headers=headers)
        html = response.content
        root = etree.HTML(html)
        elements = root.xpath(CITY_DISTRICT_XPATH)
        en_names = list()
        ch_names = list()
        for element in elements:
            link = element.attrib['href']
            en_names.append(link.split('/')[-2])
            ch_names.append(element.text)

        for index, name in enumerate(en_names):
            self.chinese_city_district_dict[name] = ch_names[index]

        return en_names

    def get_areas(self, district):
        """
        通过城市和区县名获得下级板块名
        :param city: 城市
        :param district: 区县
        :return: 区县列表
        """
        page = f"{self.base_url}/xiaoqu/{district}"
        areas = list()
        chinese_area_dict = self.chinese_area_dict
        try:
            headers = create_headers()
            response = requests.get(page, timeout=20, headers=headers)
            html = response.content
            root = etree.HTML(html)
            links = root.xpath(DISTRICT_AREA_XPATH)

            # 针对a标签的list进行处理
            for link in links:
                relative_link = link.attrib['href']
                # 去掉最后的"/"
                relative_link = relative_link[:-1]
                # 获取最后一节
                area = relative_link.split("/")[-1]
                # 去掉区县名,防止重复
                if area != district:
                    chinese_area = link.text
                    chinese_area_dict[area] = chinese_area
                    # print(chinese_area)
                    areas.append(area)
            return areas
        except Exception as e:
            print(e)

    def start(self):
        city = self.city
        area_dict = self.area_dict

        # collect_area_zufang('sh', 'beicai')  # For debugging, keep it here
        t1 = time.time()  # 开始计时

        # 获得城市有多少区列表, district: 区县
        districts = self.get_districts()
        print('City: {0}'.format(city))
        print('Districts: {0}'.format(districts))

        # 获得每个区的板块, area: 板块
        areas = list()
        for district in districts:
            areas_of_district = self.get_areas(district)
            print('{0}: Area list:  {1}'.format(district, areas_of_district))
            # 用list的extend方法,L1.extend(L2)，该方法将参数L2的全部元素添加到L1的尾部
            areas.extend(areas_of_district)
            # 使用一个字典来存储区县和板块的对应关系, 例如{'beicai': 'pudongxinqu', }
            for area in areas_of_district:
                area_dict[area] = district
        print("Area:", areas)
        print("District and areas:", area_dict)

        # 准备线程池用到的参数
        nones = [None for i in range(len(areas))]
        args = zip(zip(areas), nones)
        # areas = areas[0: 1]

        # 针对每个板块写一个文件,启动一个线程来操作
        pool_size = self.thread_pool_size
        pool = threadpool.ThreadPool(pool_size)
        # for test
        # self.collect_area_zufang_data('caolu')
        my_requests = threadpool.makeRequests(self.collect_area_zufang_data, args)
        [pool.putRequest(req) for req in my_requests]
        pool.wait()
        pool.dismissWorkers(pool_size, do_join=True)  # 完成后退出

        # 计时结束，统计结果
        t2 = time.time()
        print("Total crawl {0} areas.".format(len(areas)))
        print("Total cost {0} second to crawl {1} data items.".format(t2 - t1, self.total_num))


if __name__ == "__main__":
    spider = ZuFangBaseSpider('sh')
    spider.start()
