# load a model; pre-trained on COCO
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

pretrained_model_dict = {
    'fasterrcnn_resnet50_fpn': torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
}


def pretrained_model(model_name, model_dict=pretrained_model_dict,num_classes=2):
    model = model_dict[model_name]
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model