from tools.seg_predict import segment_images
from tools.angle import process_images

if __name__ == '__main__':
    # 使用函数
    segment_images('./data/images', './weight/best_seg.pt', "./data/mask")
    process_images("data/mask",'data/images', "output")
