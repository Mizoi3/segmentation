import cv2
import numpy as np

import requests
from bs4 import BeautifulSoup
import urllib.request
import os

# 画像を保存するディレクトリを作成
os.makedirs('./images', exist_ok=True)

# 検索ワードを指定
keyword = '一人暮らし 部屋'

# Google画像検索から画像を取得
url = 'https://www.google.co.jp/search?q=' + keyword + '&source=lnms&tbm=isch'
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.116 Safari/537.36'}

# ページ情報を取得
r = requests.get(url, headers=headers)
soup = BeautifulSoup(r.content, 'html.parser')

# 画像URLを取得
img_urls = []
for link in soup.find_all('img'):
    img_urls.append(link.get('src'))

# 画像を取得
for i, img_url in enumerate(img_urls[:100]):
    try:
        # 画像を保存
        urllib.request.urlretrieve(img_url, './images/' + keyword + str(i) + '.jpg')
    except:
        # 画像取得失敗時はスキップ
        continue

# 画像を読み込む
img = cv2.imread('./images/image.jpg')

# セマンティックセグメンテーションを行う
seg = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
seg.setBaseImage(img)
seg.switchToSelectiveSearchFast()
rects = seg.process()

# 各領域を分類する
for (x, y, w, h) in rects[:10]:
    # 各領域を切り出す
    roi = img[y:y+h, x:x+w]
    # 切り出した領域をリサイズする
    roi = cv2.resize(roi, (64, 64))
    # 画像をグレースケールに変換する
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    # 画像を2値化する
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    # 輪郭を抽出する
    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    # 輪郭を走査して、壁、ベッド、机、イス、テーブルに分類する
    for c in contours:
        area = cv2.contourArea(c)
        if area > 1000:
            # 壁
            if area > 10000:
                cv2.drawContours(img, [c], -1, (0, 0, 255), 2)
            # ベッド
            elif area > 5000:
                cv2.drawContours(img, [c], -1, (0, 255, 0), 2)
            # 机
            elif area > 2000:
                cv2.drawContours(img, [c], -1, (255, 0, 0), 2)
            # イス
            elif area > 500:
                cv2.drawContours(img, [c], -1, (255, 255, 0), 2)
            # テーブル
            else:
                cv2.drawContours(img, [c], -1, (0, 255, 255), 2)

# 結果を表示する
cv2.imshow('result', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
