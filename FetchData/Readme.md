# Data Description
The tsv file contain following columns:
``` python3
columns = ['fetch_date', 'district', 'area', 'xiaoqu', 'layout', 'size', 'price', 'url', 'house_id', 'relative_image_path',
                'house_info_dict', 'facility_info_dict', 'house_description', 'agent_name', 'recommend_house_id_geo',
                'nearby_house_id', 'tag_list', 'subway_info']
``` 
Detail:
-   fetch_date: The date of fetching, in format "YYYYMMDD"
-   district: The district of the house
-   area: A more detail location than district
-   xiaoqu: Name of the housing estate, in format "{rental type}·{housing estate}"
-   layout: Layout of the house
-   size: Floor space
-   price: Rental price
-   url: The url of the house in website
-   house_id: A unique id for one house
-   relative_image_path: Displayed images in website, in format "[({image url, room type})]"
-   house_info_dict: Information about "面积, 朝向, 维护, 入住, 楼层, 电梯, 车位, 用水, 民水, 用电, 民电, 燃气, 有, 采暖, 自采暖"
-   facility_info_dict: The furniture and household appliances information
-   house_description: The description of the house provided by the house agent
-   agent_name: The name of the agent
-   recommend_house_id_geo: The house id and geographic information of recommended houses.
-   nearby_house_id: The house id of nearby houses
-   tag_list: The tags of the house
-   subway_info: A list of nearby subway station and distance.
-   geo_lat: The latitude of the house
-   geo_lng: The longitude of the house
