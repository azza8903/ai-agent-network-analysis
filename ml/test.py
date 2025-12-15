import numpy as np
import torch
from models.RF import getRF
import torch.utils.data as Data
import const_rf as const
import csv
import pre_recall

def load_data(fpath):
    train = np.load(fpath, allow_pickle=True).item()
    train_X, train_y = train['dataset'], train['label']
    return train_X, train_y


def load_model(class_num, path, device):
    model = getRF(class_num)
    model.load_state_dict(torch.load(path + '.pth'))
    model = model.to(device)
    return model

def main(agent_type):
    test_dataset = f'ml/datasets/{agent_type}_test.npy'

    device = "mps" if torch.backends.mps.is_available() else \
            "cuda" if torch.cuda.is_available() else "cpu"

    ml_model = load_model(const.num_classes, f'ml/datasets/{agent_type}_train', device).eval()

    features, test_y = load_data(test_dataset)

    test_x = torch.unsqueeze(torch.from_numpy(features), dim=1).type(torch.FloatTensor)
    test_x = test_x.to(device)
    test_y = torch.squeeze(torch.from_numpy(test_y)).type(torch.LongTensor)
    test_data = Data.TensorDataset(test_x, test_y)
    test_loader = Data.DataLoader(dataset=test_data, batch_size=1, shuffle=False)
    test_results = []
    with torch.no_grad():
        for v, (x, y) in enumerate(test_loader):
            x = x.reshape(1, 1, 4, 600)   # (N, C, H, W)
            defense_output = ml_model(x).cpu().squeeze().detach().numpy()
            pre = np.argmax(defense_output)
            test_results.append([y.item(), pre])

    result_file = f'ml/datasets/{agent_type}_test.csv'
    with open(result_file, 'w+', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        for item in test_results:
            writer.writerow(item)

    acc = pre_recall.pre_recCall(result_file, result_file[:-4] + '_ana.csv', const.num_classes)
    print('avg acc:' + str(acc))


if __name__ == '__main__':
    agent_type = "weather_agent"
    main(agent_type)