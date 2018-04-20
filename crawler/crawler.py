from config import *
import requests
import json
import os
import argparse
import shutil

def retrieve_style_json():
    print("retrieving styles json ... ")
    dic = {}
    for style in STYLES:
        dic[style] = []
        for page in range(1, PAGE_LIMIT):
            url = BASE_URL + STYLE_URL_PREFIX + style + PAGINATION_URL_PREFIX + str(page)
            try:
                response = requests.get(url, timeout=METADATA_REQUEST_TIMEOUT)
                res_json = response.json()['Paintings']
                if res_json is None:
                    break
                dic[style].extend(res_json)
            except Exception as e:
                print(str(e))
                continue

    with open(STYLE_JSON, 'w') as f:
        json.dump(dic, f)
    print("Generate json file!")


def retrieve_image():
    with open(STYLE_JSON, 'r') as f:
        paints_styles_info = json.load(f)

    num = sum([len(paints_styles_info[style]) for style in paints_styles_info.keys()])
    
    print('Totally image number: ' + str(num))
    print("downloading images ... ")

    count = 0
    success = 0

    for style in paints_styles_info.keys():
        print("starting on style:" + style)
        paths_infos = paints_styles_info[style]

        os.makedirs(os.path.join(IMAGE_PATH, style), exist_ok=True)

        for paints_info in paths_infos:
            count += 1
            url = paints_info["image"]
            title = paints_info["title"]
            name = url.split('/')[-1]

            try: 
                response = requests.get(url, timeout=PAINTINGS_REQUEST_TIMEOUT, stream=True)

                filepath = os.path.join(IMAGE_PATH, style, name)

                # print(filepath)

                with open(filepath, 'wb') as f:
                    response.raw.decode_content = True
                    shutil.copyfileobj(response.raw, f)
                    print("["+str(count)  + "] - Save file: " + name)
                    success += 1
            except Exception as e:
                print("["+str(count)  + "] - Fail")
                print(str(e))

    print("Finish downloading!")

    print('Totally image number: ' + str(num))

    print('Success number: ' + str(success))



def main():
    # parser = argparse.ArgumentParser()
 #    parser.add_argument('-name', '--name', type=str, 
 #        required=True, help='the name of the style json file.')
 #    parser.add_argument("-m", "--mode", help = "select mode by 'train' or test",
 #        choices = ["train", "test"], default = "test")
    
 #    args = parser.parse_args()

    os.makedirs(IMAGE_PATH, exist_ok=True)
    os.makedirs(JSON_PATH, exist_ok=True)
    retrieve_style_json()
    retrieve_image()


if __name__ == "__main__":
    main()