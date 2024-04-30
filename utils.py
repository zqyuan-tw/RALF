import random
import numpy as np
import torch
from tqdm import tqdm
from torchmetrics import AUROC, Accuracy, ROC
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from tsnecuda import TSNE
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import BinaryClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

def fixed_seed(myseed):
    np.random.seed(myseed)
    random.seed(myseed)
    torch.manual_seed(myseed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed(myseed)
        torch.cuda.manual_seed_all(myseed)


def predict(
    model,
    batch
):
    model.eval()
    with torch.no_grad():
        logits = model(batch)

    return logits


def evaluate(
    model,
    dataloader,
    criterion,
    device
):

    auc = AUROC(task='binary')
    acc = Accuracy(task='binary')
    roc = ROC(task='binary')

    val_loss = 0
    for images, labels in tqdm(dataloader, desc="Evaluation", unit='batch', leave=False):
        images = images.to(device)
        labels = labels.to(device)

        preds = predict(model, images)
        loss  = criterion(preds, labels)
        val_loss += loss.item()

        auc.update(preds.cpu(), labels.int().cpu())
        acc.update(preds.cpu(), labels.int().cpu())
        roc.update(preds.cpu(), labels.int().cpu())
        
    fpr, tpr, thresholds = roc.compute()
    fnr = 1 - tpr
    eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]

    return val_loss / len(dataloader), auc.compute().item(), acc.compute().item(), eer.item()


def draw_freq_mask(model, fig_path=None):
    plt.figure()
    plt.axis('off')
    plt.imshow(torch.sigmoid(model.destructor.freq_mask).detach().cpu(), cmap='rainbow')
    plt.colorbar()
    if fig_path is not None:
        plt.savefig(fig_path)
    

def draw_reconstruction(image, model, fig_path=None, device=torch.device('cpu')):
    if image.dim() == 3:
        image = image.unsqueeze(0)
    image = image.to(device)
    model = model.to(device)
    
    rec_img = model.reconstructor(model.destructor(image)).detach().cpu().squeeze(0)
    
    plt.figure()
    plt.axis('off')
    plt.imshow(rec_img.permute(1,2,0))
    if fig_path is not None:
        plt.savefig(fig_path)
    
    return rec_img
    

def plot_tsne(model, dataloader, fig_path=None, device=torch.device('cpu')):
    
    model = model.to(device)
    
    feature_maps = []
    def hook_feat_map(mod, inp, out):
        feature_maps.append(out)
    
    model.classifier.stage2[-2].register_forward_hook(hook_feat_map)
    
    labels = []
    
    model.eval()
    for img, lab in tqdm(dataloader, desc='TSNE'):
        predict(model, img.to(device))
        labels.extend(lab.squeeze(1).tolist())

    X = torch.cat(feature_maps, dim=0).cpu().numpy()
    
    X_embedded = TSNE(n_components=2, perplexity=30).fit_transform(X)
    
    assert X_embedded.shape[0] == len(labels), "The number of embedding and label does not match !"
    
    classes = ['fake', 'real']
    colors = ListedColormap(['red', 'green'])
    
    plt.figure()
    plt.axis('off')
    scatter = plt.scatter(X_embedded[:,0], X_embedded[:,1], s=1, c=labels, cmap=colors, marker='.')
    plt.legend(handles=scatter.legend_elements()[0], labels=classes)
    if fig_path is not None:
        plt.savefig(fig_path)


def grad_cam(model, image, use_cuda=True, fig_path=None):
    if image.dim() == 3:
        image = image.unsqueeze(0)
    target_layers = [model.classifier.stage2[-4]]
    targets = [BinaryClassifierOutputTarget(0)]
    
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=use_cuda)
    grayscale_cams = cam(input_tensor=image, targets=targets)
    cam_image = show_cam_on_image(image.squeeze().permute(1, 2, 0).numpy(), grayscale_cams[0,:], use_rgb=True)
        
    plt.figure()
    plt.axis('off')
    plt.imshow(cam_image)
    if fig_path is not None:
        plt.savefig(fig_path)