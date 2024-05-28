from dataset import DeepFakeDataset
from model import *
from train_utils import prepare_dirs, run_training, go_test
from torch.utils.data import DataLoader
import ujson as json

if __name__ == "__main__":

    config_path = "train_config_0.json"

    with open(config_path, "r", encoding='utf8') as cfg_file:
        cfg = json.load(cfg_file)

    torch.cuda.set_device(cfg['device'])

    save_pt, save_im_pt = prepare_dirs(cfg)

    transforms = prepare_augmentation()
    train_dataset = DeepFakeDataset(data_dir=cfg['train_data_dir'], transforms=transforms)
    test_dataset = DeepFakeDataset(data_dir=cfg['test_data_dir'], transforms=transforms)
    valid_dataset = DeepFakeDataset(data_dir=cfg['valid_data_dir'], transforms=transforms)
    my_dataset = DeepFakeDataset(data_dir=cfg['FaceForensics'], transforms=transforms)

    if torch.cuda.is_available():
        print(torch.cuda.get_device_name(cfg['device']) + ' is available!')
    else:
        print('No GPU!!!')
        exit(-1)

    model = Meso4().cuda()

    # model.summary()

    model_optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08)

    start_ep = 0
    if cfg['file_to_start']:
        chpnt = cfg['file_to_start'].split("/")[-2]
        start_ep = int(chpnt) + 1

        checkpoint = torch.load(cfg['file_to_start'])
        model.load_state_dict(checkpoint['model_state_dict'])  # ['model_state_dict'])
        print("Successfully loaded weights from", cfg['file_to_start'])
        if cfg['to_validate']:
            loss = nn.CrossEntropyLoss()
            valid_loader = DataLoader(dataset=valid_dataset,
                                     batch_size=1024,
                                     shuffle=True,
                                     num_workers=10)
            test_loss, recall, precision = go_test(valid_loader, cfg, model, loss, save_im_pt, int(chpnt))
            print(test_loss, recall, precision)

            # test_loader = DataLoader(dataset=test_dataset,
            #                           batch_size=1024,
            #                           shuffle=True,
            #                           num_workers=10)
            # test_loss, acc = go_test(test_loader, cfg, model, loss, save_im_pt, int(chpnt))
            # print(test_loss, acc)
            #
            # my_loader = DataLoader(dataset=my_dataset,
            #                           batch_size=1024,
            #                           shuffle=True,
            #                           num_workers=10)
            # test_loss, acc = go_test(my_loader, cfg, model, loss, save_im_pt, int(chpnt))
            # print(test_loss, acc)

            exit(0)

    run_training(config=cfg,
                 recognizer=model,
                 optimizer=model_optimizer,
                 train_dataset=train_dataset,
                 test_dataset=test_dataset,
                 valid_dataset=valid_dataset,
                 my_dataset=my_dataset,
                 save_pt=save_pt,
                 save_im_pt=save_im_pt,
                 start_ep=start_ep)
