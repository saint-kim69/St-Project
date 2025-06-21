import yaml

class Detector:
    def __init__(self, weights_path, cfg_path, yaml_path, img_size=640, conf_thres=0.25, iou_thres=0.45):
        with open(yaml_path, 'r') as f:
            self.yaml_data = yaml.safe_load(f)
            self.class_names = self.yaml_data['names']

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(f'using device: {self.device}')

        self.model = torch.hub.load('ultralytics/yolov4', 'custom', path=weights_path)
        self.model.conf = conf_thres
        self.model.iou = iou_thres
        self.model.img_size = img_size
        self.model.to(self.device)

    def detect(self, image_path):
        img = cv2.imread(image_path)
        if not img:
            print(f'이미지를 로드할 수 없습니다.: {image_path}')
            return None
        
        results = self.model(img)

        return results
    
    def visualize(self, image_path, save_path=None):
        img = cv2.imread(image_path)
        results = self.detect(image_path)
        results.render()

        output_img = results.ims[0]

        if save_path:
            cv2.imwrite(save_path, output_img)
            print(f'결과 이미지가 저장되었습니다. : {save_path}')
