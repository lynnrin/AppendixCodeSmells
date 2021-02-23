import glob

from PIL import Image, ImageChops


def cropImage(image): #引数は画像の相対パス
    # 画像ファイルを開く
    img = Image.open(image)

    # 周りの部分は強制的にトリミング
    w, h = img.size
    box = (w*0.05, h*0.05, w*0.95, h*0.95)
    img = img.crop(box)
    #
    # # 背景色画像を作成
    # bg = Image.new("RGB", img.size, img.getpixel((0, 0)))
    # # bg.show()
    #
    # # 背景色画像と元画像の差分画像を作成
    # diff = ImageChops.difference(img, bg)
    # # diff.show()
    #
    # # 背景色との境界を求めて画像を切り抜く
    # croprange = diff.convert("RGB").getbbox()
    # crop_img = img.crop(croprange)
    # # crop_img.show()

    return img


files = glob.glob('/Users/lynnrin/Documents/master_theis/result/**/*/*/concat.png')

for image in files:
    print(image)
    crop_img = cropImage(image)
    crop_img.save(image)



