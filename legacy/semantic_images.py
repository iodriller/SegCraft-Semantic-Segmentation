# sample execution (requires torchvision)
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import torch
import os


class semantic_images():
    def __init__(self, net='deeplabv3_resnet101', jpg_qualty=40, overlay_directory='overlay/'):
        self.overlay_directory = overlay_directory
        self.net = net
        self.jpg_qualty = jpg_qualty
        self.overlay_directory = overlay_directory

    def create_directory(self):
        if not os.path.exists(self.overlay_directory):
            os.mkdir(self.overlay_directory)

    def create_model(self):
        print(self.net)
        model = torch.hub.load('pytorch/vision:v0.6.0', self.net, pretrained=True)
        model.eval()
        print('the model details for semantic segmentation:', model.eval())
        return model

    def create_semantic_images(self, dirname, file_name, model):

        input_image = Image.open(os.path.join(dirname, file_name))

        ## Modify here, if a different net is being used
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        input_tensor = preprocess(input_image)
        input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model

        # move the input and model to GPU for speed if available
        if torch.cuda.is_available():
            input_batch = input_batch.to('cuda')
            model.to('cuda')

        with torch.no_grad():
            output = model(input_batch)['out'][0]
        output_predictions = output.argmax(0)

        # create a color pallette, selecting a color for each class
        palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
        colors = torch.as_tensor([i for i in range(21)])[:, None] * palette
        colors = (colors % 255).numpy().astype("uint8")

        # plot the semantic segmentation predictions of 21 classes in each color
        r = Image.fromarray(output_predictions.byte().cpu().numpy()).resize(input_image.size)
        r.putpalette(colors)
        ##

        plt.imshow(r)
        plt.imsave(self.overlay_directory + file_name, r, dpi=self.jpg_qualty)
        plt.show()
