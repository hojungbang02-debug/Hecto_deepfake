import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import torch
import torch.nn as nn
import timm
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from src.model import DeepFakeModelDinoV2
from src.dataset import DeepFakeDataset
from train import CONFIG

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = DeepFakeModelDinoV2(model_name='dinov2_vitb14', in_chs=12, hidden_dim=[512, 256], drop_rate=0.3, pretrained=True).to(device)
    model.load_state_dict(torch.load('./model/best_model.pth', map_location=device, weights_only=True))
    print("모델 가중치 로드 완료!")
    print()
    print(model)
    print()
    model.eval()

    loader = DataLoader(
        DeepFakeDataset(root_dir='./train_data', mode='val', image_size=CONFIG['image_size']),
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        num_workers=6,
        pin_memory=True
    )

    embeddings = []
    labels = []

    for images, targets in tqdm(loader, desc="Extracting Embeddings"):
        images = images.to(device)

        with  torch.inference_mode(), torch.no_grad(), torch.amp.autocast(device_type=device.type, enabled=True):
            feats = model.foward_features(images)
            embeddings.append(feats.cpu())
        labels.append(targets)

    embeddings = torch.cat(embeddings, dim=0).float()
    labels = torch.cat(labels, dim=0).int()

    embeddings_np = embeddings.numpy()
    labels_np = labels.numpy()


    tsne = TSNE(
        n_components=2,
        perplexity=30,
        learning_rate="auto",
        init="pca",
        random_state=42,
    )

    emb_2d = tsne.fit_transform(embeddings_np)

    plt.figure(figsize=(7, 7))

    plt.scatter(
        emb_2d[labels_np == 0, 0],
        emb_2d[labels_np == 0, 1],
        s=8,
        alpha=0.6,
        label="Real"
    )

    plt.scatter(
        emb_2d[labels_np == 1, 0],
        emb_2d[labels_np == 1, 1],
        s=8,
        alpha=0.6,
        label="Fake"
    )

    plt.legend()
    plt.title("Backbone Embedding Visualization (Real vs Fake)")
    plt.axis("off")
    plt.show()

    clf = LogisticRegression(max_iter=1000)
    scores = cross_val_score(clf, embeddings_np, labels_np, cv=5, scoring="roc_auc")
    print("ROC AUC:", scores.mean())

    clf.fit(embeddings_np, labels_np)
    scores = clf.decision_function(embeddings_np)

    plt.hist(scores[labels_np==0], bins=100, alpha=0.6, label="Real")
    plt.hist(scores[labels_np==1], bins=100, alpha=0.6, label="Fake")
    plt.legend()
    plt.title("Projection onto Linear Decision Axis")
    plt.show()
    

    emb_norm = embeddings_np / np.linalg.norm(embeddings_np, axis=1, keepdims=True)
    tsne = TSNE(
        n_components=2,
        perplexity=30,
        learning_rate="auto",
        init="pca",
        random_state=42,
    )

    emb_2d = tsne.fit_transform(emb_norm)
    plt.figure(figsize=(7, 7))

    plt.scatter(
        emb_2d[labels_np == 0, 0],
        emb_2d[labels_np == 0, 1],
        s=8,
        alpha=0.6,
        label="Real"
    )

    plt.scatter(
        emb_2d[labels_np == 1, 0],
        emb_2d[labels_np == 1, 1],
        s=8,
        alpha=0.6,
        label="Fake"
    )

    plt.legend()
    plt.title("Backbone Embedding Norm Visualization (Real vs Fake)")
    plt.axis("off")
    plt.show()
