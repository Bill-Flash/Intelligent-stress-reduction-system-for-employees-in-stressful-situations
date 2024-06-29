from Regression_Model import StressPredictor
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_squared_error, r2_score

def load_model(model_path, input_dim):
    model = StressPredictor(input_dim)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def predict(model, dataloader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for tensors, labels in dataloader:
            outputs = model(tensors).squeeze()
            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return all_preds, all_labels

if __name__ == "__main__":
    # 加载实际测试数据和标签
    test_data = np.load('test_data.npy')  # 替换为实际的文件路径
    test_labels = np.load('test_labels.npy')  # 替换为实际的文件路径
    test_tensors = torch.tensor(test_data, dtype=torch.float32)
    test_labels = torch.tensor(test_labels, dtype=torch.float32)

    test_dataset = TensorDataset(test_tensors, test_labels)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # load model
    model_path = 'stress_predictor_model.pth'
    input_dim = 8
    model = load_model(model_path, input_dim)

    # make predict
    preds, labels = predict(model, test_dataloader)

    mse = mean_squared_error(labels, preds)
    r2 = r2_score(labels, preds)

    print(f'Final MSE: {mse:.4f}, R2: {r2:.4f}')




