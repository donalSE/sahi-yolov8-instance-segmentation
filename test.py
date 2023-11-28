from sahi import AutoDetectionModel
from sahi.predict import get_prediction, get_sliced_prediction

detection_model = AutoDetectionModel.from_pretrained(
model_type='yolov8',
model_path='best.pt',
config_path='data.yaml',
confidence_threshold=0.5,
device='cpu' # if torch.cuda.is_available() else 'cpu'
)

pred = get_sliced_prediction('innisbrook-resort-copperhead_image.png', detection_model,
                             slice_height=800,
                             slice_width=1856,
overlap_height_ratio=0.05, overlap_width_ratio=0.05)

pred.export_visuals('test_results/', file_name='innisbrook.jpeg')