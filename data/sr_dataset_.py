import os.path
import torch.utils.data as data
from .util import *
from .deblur import *

class LRGTDataset(data.Dataset):

    def __init__(self, opt):
        super(LRGTDataset, self).__init__()
        self.opt = opt
        self.paths_LR = None
        self.paths_GT = None
        self.LR_env = None  # environment for lmdb
        self.GT_env = None

        # read image list from subset list txt
        if opt['subset_file'] is not None and opt['phase'] == 'train':
            with open(opt['subset_file']) as f:
                self.paths_GT = sorted([os.path.join(opt['dataroot_GT'], line.rstrip('\n')) \
                        for line in f])
            if opt['dataroot_LR'] is not None:
                raise NotImplementedError('Now subset only supports generating LR on-the-fly.')
        else:  # read image list from lmdb or image files
            self.GT_env, self.paths_GT = get_image_paths(opt['data_type'], opt['dataroot_GT'])
            self.LR_env, self.paths_LR = get_image_paths(opt['data_type'], opt['dataroot_LR'])

        assert self.paths_GT, 'Error: GT path is empty.'
        if self.paths_LR and self.paths_GT:
            assert len(self.paths_LR) == len(self.paths_GT), \
                'GT and LR datasets have different number of images - {}, {}.'.format(\
                len(self.paths_LR), len(self.paths_GT))

        self.random_scale_list = [1]

    def __getitem__(self, index):

        GT_path, LR_path = None, None
        scale = self.opt['scale']
        GT_size = self.opt['GT_size']

        # get GT image
        GT_path = self.paths_GT[index]
        # print(self.GT_env)
        img_GT =  read_img(self.GT_env, GT_path)
        # print(" load img_GT {}".format(img_GT.shape))
        # print(" load GT_path {}".format(GT_path))
        # print(" load GT_size {}".format(GT_size))

        # modcrop in the validation / test phase
        if self.opt['phase'] != 'train':
            print("start val=====>")
            LR_path = self.paths_LR[index]
            img_LR = read_img(self.LR_env, LR_path)
            img_LR = cv2.resize(img_LR, (1024, 1024), interpolation=cv2.INTER_LINEAR)
            img_GT = cv2.resize(img_GT, (1024, 1024), interpolation=cv2.INTER_LINEAR)



        # change color space if necessary
        if self.opt['color']:
            img_GT =  channel_convert(img_GT.shape[2], self.opt['color'], [img_GT])[0]

        # get LR image
        if self.paths_LR and self.opt['phase'] == 'train':
            LR_path = self.paths_LR[index]
            img_LR =  read_img(self.LR_env, LR_path)
            img_LR = cv2.resize(img_LR, (GT_size, GT_size), interpolation=cv2.INTER_LINEAR)
            # print(" load img_LR {}".format(img_LR.shape))
            img_GT = cv2.resize(img_GT, (GT_size, GT_size), interpolation=cv2.INTER_LINEAR)
            # img_input

        else:
            # randomly scale during training
            if self.opt['phase'] == 'train':
                random_scale = random.choice(self.random_scale_list)
                H_s, W_s, _ = img_GT.shape

                def _mod(n, random_scale, scale, thres):
                    rlt = int(n * random_scale)
                    rlt = (rlt // scale) * scale
                    return thres if rlt < thres else rlt

                H_s = _mod(H_s, random_scale, scale, GT_size)
                W_s = _mod(W_s, random_scale, scale, GT_size)
                img_GT = cv2.resize(np.copy(img_GT), (W_s, H_s), interpolation=cv2.INTER_LINEAR)
                # force to 3 channels
                if img_GT.ndim == 2:
                    img_GT = cv2.cvtColor(img_GT, cv2.COLOR_GRAY2BGR)

            H, W, _ = img_GT.shape
            # using matlab imresize
            img_LR =  imresize_np(img_GT, 1 / scale, True)
            if img_LR.ndim == 2: img_LR = np.expand_dims(img_LR, axis=2)

        if self.opt['phase'] == 'train':
            # # if the image size is too small
            # H, W, _ = img_GT.shape
            # if H < GT_size or W < GT_size:
            #     img_GT = cv2.resize(np.copy(img_GT), (GT_size, GT_size), interpolation=cv2.INTER_LINEAR)
            #     if img_GT.ndim == 2: img_GT = np.expand_dims(img_GT, axis=2)
            #     # using matlab imresize
            #     img_LR =  imresize_np(img_GT, 1 / scale, True)
            #     if img_LR.ndim == 2: img_LR = np.expand_dims(img_LR, axis=2)
            #
            # H, W, C = img_LR.shape
            # LR_size = GT_size // scale
            #
            # # randomly crop
            # rnd_h = random.randint(0, max(0, H - LR_size))
            # rnd_w = random.randint(0, max(0, W - LR_size))
            # img_LR = img_LR[rnd_h:rnd_h + LR_size, rnd_w:rnd_w + LR_size, :]
            # # print(" after crop {}".format(img_LR.shape))
            # rnd_h_GT, rnd_w_GT = int(rnd_h * scale), int(rnd_w * scale)
            # img_GT = img_GT[rnd_h_GT:rnd_h_GT + GT_size, rnd_w_GT:rnd_w_GT + GT_size, :]
            # # print(" after crop {}".format(img_GT.shape))
            #
            # img_GT = cv2.resize(img_GT, (GT_size, GT_size), interpolation=cv2.INTER_LINEAR)
            # img_LR = cv2.resize(img_LR, (LR_size, LR_size), interpolation=cv2.INTER_LINEAR)

            # print(" after crop {}".format(img_LR.shape))
            # print(" after crop {}".format(img_GT.shape))

            # augmentation - flip, rotate
            img_LR, img_GT =  augment([img_LR, img_GT], self.opt['use_flip'], self.opt['use_rot'])

        # change color space if necessary
        if self.opt['color']:
            img_LR =  channel_convert(C, self.opt['color'], [img_LR])[0]

        # BGR to RGB, HWC to CHW, numpy to tensor
        if img_GT.shape[2] == 3:
            img_GT = img_GT[:, :, [2, 1, 0]]
            img_LR = img_LR[:, :, [2, 1, 0]]
        img_GT = torch.from_numpy(np.ascontiguousarray(np.transpose(img_GT, (2, 0, 1)))).float()
        img_LR = torch.from_numpy(np.ascontiguousarray(np.transpose(img_LR, (2, 0, 1)))).float()

        if LR_path is None:
            LR_path = GT_path
        # print("xxxxxxxxxxxxxxxxxxxx")
        # print(img_LR.shape)
        # print(img_GT.shape)
        # print(LR_path)
        # print(GT_path)
        # print("xxxxxxxxxxxxxxxxxxxx")
        # print(6666666666666666)

        # addBlur_and_addNoise()

        # if self.opt['phase'] == 'train':
        #     img_LR = img_LR.unsqueeze(0)
        #     img_LR = addBlur_and_addNoise(img_LR, 10, 10)
        #     img_LR = img_LR.squeeze(0)

        # print(img_LR.size())
        # # print(img_LR.type())
        # print("----------------")
        # print(img_LR.shape)
        # print(img_GT.shape)

        return {'LR': img_LR, 'GT': img_GT, 'LR_path': LR_path, 'GT_path': GT_path}

    def __len__(self):
        return len(self.paths_GT)
