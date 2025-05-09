import torch
import torchvision.models as models
import torchvision

def get_model(model_name, num_classes):
    if model_name == "resnet18":
        model = models.resnet18(pretrained=True)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

    elif model_name == "resnet34":
        model = models.resnet34(pretrained=True)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

    elif model_name == "resnet50":
        model = models.resnet50(pretrained=True)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

    elif model_name == "efficientnet_b0":
        model = models.efficientnet_b0(pretrained=True)
        model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)

    elif model_name == "efficientnet_b4":
        model = models.efficientnet_b4(pretrained=False)
        model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)

    elif model_name == "densenet121":
        model = models.densenet121(pretrained=True)
        model.classifier = torch.nn.Linear(model.classifier.in_features, num_classes)

    elif model_name == "densenet169":
        model = models.densenet169(pretrained=True)
        model.classifier = torch.nn.Linear(model.classifier.in_features, num_classes)

    elif model_name == "densenet161":
        model = models.densenet161(pretrained=True)
        model.classifier = torch.nn.Linear(model.classifier.in_features, num_classes)

    elif model_name == "densenet201":
        model = models.densenet201(pretrained=True)
        model.classifier = torch.nn.Linear(model.classifier.in_features, num_classes)

    elif model_name == "vgg16":
        model = models.vgg16(pretrained=True)
        model.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        model.classifier = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(512, 512),
            torch.nn.ReLU(True),
            torch.nn.Dropout(),
            torch.nn.Linear(512, num_classes)
        )

    elif model_name == "vgg19":
        model = models.vgg19(pretrained=True)
        model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, num_classes)

    elif model_name == "inception_v3":
        model = models.inception_v3(pretrained=True, aux_logits=True)
        model.AuxLogits.fc = torch.nn.Linear(model.AuxLogits.fc.in_features, num_classes)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

    elif model_name == "mobilenet_v2":
        model = models.mobilenet_v2(pretrained=True)
        model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)

    elif model_name == "googlenet":
        model = models.googlenet(pretrained=True, aux_logits=True)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

    else:
        raise ValueError(f"Model {model_name} is not supported yet.")

    return model
