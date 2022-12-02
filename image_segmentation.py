import cv2
import numpy as np

import requests
import shutil
import os

# 画像を取得するURLを作成
url = 'https://www.google.co.jp/search?q=%E4%B8%80%E4%BA%BA%E6%9A%AE%E3%82%89%E3%81%97+%E9%83%A8%E5%B1%8B&source=lnms&tbm=isch&sa=X&ved=2ahUKEwjV_7yV_KXqAhXKfXAKHV_6CzQQ_AUoAXoECBQQAw&biw=1536&bih=722'

# 画像を取得して保存する
if not os.path.exists('./images'):
    os.mkdir('./images')

for i in range(100):
    res = requests.get(url, stream=True)
    if res.status_code == 200:
        with open('./images/image{0:03d}.jpg'.format(i), 'wb') as f:
            shutil.copyfileobj(res.raw, f)

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
