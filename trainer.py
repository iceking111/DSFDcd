import numpy as np
import torch.optim as optim
from dataset import load_img
import os
import torch.nn.functional as F
from models.model import *
from misc.metric_tool import ConfuseMatrixMeter
from models.myloss import CustomCrossEntropyLoss,L2NormLossForBCHW,mse_loss,L1_loss,L2_loss,CosineSimilarityLoss,L2_regularize,compute_kl_loss,compute_kl_loss1
import time
from models.myloss1 import CombinedLoss,BCE_loss,MyEntropyLoss
from models.myloss3 import calculate_ssim_loss
from misc.logger_tool import Logger, Timer


class CDTrainer():

    def __init__(self, train_dataloader, val_dataloader, batch_size):

        self.train_dataloader = train_dataloader  # 训练集
        self.val_dataloader = val_dataloader   # 验证集

        self.n_class = 2 # 最后是变与不变两个类别
        # define G
        self.net_G = Net() # 网络传入

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.net_G = self.net_G.to(self.device)

        print(self.device)# 输出设备名称   cuda就是正确的利用到了GPU


        # 设置一个随机梯度下降优化器   参数有
        # lr 学习率 即每次更新参数的步长
        # momentum 设置一个动量  有助于加速收敛并减少震荡
        # weight_decay 设置了权重衰减 用于正则化以防止过拟合
        self.optimizer_G = optim.SGD(self.net_G.parameters(), lr=0.01,momentum=0.9,weight_decay=5e-4)
        # 创建了一个学习率调度器 会在每个训练周期结束后调整学习率  step_size 表示每过10个epoch 学习率调整一次  gamma表示每个学习率调整的因子，即学习率会乘0.7
        self.exp_lr_scheduler_G = optim.lr_scheduler.StepLR(self.optimizer_G, step_size=20, gamma=0.7)
        # 自定义的性能度量工具 用于计算混淆矩阵以及其他指标  暂时可以认为他就是混淆矩阵这个类
        self.running_metric = ConfuseMatrixMeter(n_class=2)
        # 定义了检查点的路径
        self.checkpoint_dir = "11111111/"
        # 如果检查点路径不存在 就创建一个
        if os.path.exists(self.checkpoint_dir) is False:
            os.mkdir(self.checkpoint_dir)
        # 定义日志文件的路径，并创建一个 Logger 对象用于记录训练和验证过程中的信息。
        logger_path = os.path.join(self.checkpoint_dir, 'log.txt')
        self.logger = Logger(logger_path)
        # 创建一个 Timer 对象，用于跟踪训练和验证过程中的时间。
        self.timer = Timer()
        # 初始化批次大小
        self.batch_size = batch_size
        # 记录训练开始的绝对时间
        self.begin_total = time.time()  # 开始时间
        self.end_total = 0
        # 初始化训练周期准确率、最佳验证准确率和最佳训练周期 ID。
        self.epoch_mf1 = 0
        self.best_val_f1 = 0.0
        self.best_epoch_id = 0
        # 初始化从哪个训练周期开始、最大训练周期数、全局步数、每个训练周期的步数和总步数。
        self.epoch_to_start = 0
        self.max_num_epochs = 150
        self.global_step = 0
        # 每个训练周期的步数  数据集个数/epoch_size
        self.steps_per_epoch = len(self.train_dataloader)
        self.total_steps = (self.max_num_epochs - self.epoch_to_start)*self.steps_per_epoch
        # 初始化模型预测、可视化预测、当前批次数据、总损失和各个损失项。
        self.G_pred = None
        self.pred_vis = None
        self.batch = None
        self.G_loss = 0
        self.G_loss1 = 0
        self.G_loss2 = 0
        self.G_loss3 = 0
        self.G_loss4 = 0
        # 初始化是否处于训练状态、当前批次 ID 和当前训练周期 ID。
        self.is_training = False
        self.batch_id = 0
        self.epoch_id = 0
        # 初始化不同的损失函数实例，这些损失函数将在训练过程中用于计算损失。
        # self.loss1 = CustomCrossEntropyLoss()# 创建交叉熵损失函数实例 用于计算变化图和标签的损失
        self.loss1 = CosineSimilarityLoss()
        self.loss2 = CombinedLoss()
        self._pxl_loss = CombinedLoss()
        self.my_loss = MyEntropyLoss()

    # 定义了一个方法，作用是加载模型的检查点checkpoint
    def _load_checkpoint(self, ckpt_name='last.pt'): # 默认加载last.pt的检查点文件
        # 如果检查点文件存在 就用torch.load加载检查点文件c
        if os.path.exists(os.path.join(self.checkpoint_dir, ckpt_name)):
            self.logger.write('加载最近的检查点...\n')
            checkpoint = torch.load(os.path.join(self.checkpoint_dir, ckpt_name),map_location=self.device)
            # 使用 load_state_dict 方法将检查点中的模型参数加载到 self.net_G 模型中。
            self.net_G.load_state_dict(checkpoint['model_G_state_dict'])
            # 加载检查点中的优化器参数到 self.optimizer_G 优化器中。
            self.optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
            # 加载检查点中的学习率调度器参数到 self.exp_lr_scheduler_G 学习率调度器中。
            self.exp_lr_scheduler_G.load_state_dict(checkpoint['exp_lr_scheduler_G_state_dict'])
            self.net_G.to(self.device)
            # 更新从哪个训练周期开始、最佳验证准确率和最佳训练周期 ID。
            self.epoch_to_start = checkpoint['epoch_id'] + 1
            self.best_val_f1 = checkpoint['best_val_f1' ]
            self.best_epoch_id = checkpoint['best_epoch_id']
            # 根据新的训练周期开始点重新计算总步数。
            self.total_steps = (self.max_num_epochs - self.epoch_to_start)*self.steps_per_epoch
            # 使用 self.logger 记录加载的检查点信息和训练将从哪个周期开始。
            self.logger.write('从批次%d开始, 最好f1 = %.4f (at epoch %d)\n' %
                  (self.epoch_to_start, self.best_val_f1, self.best_epoch_id))
            self.logger.write('\n')
        # 如果检查点文件不存在，则打印消息表示将从头开始训练。
        else:
            print('从头开始训练')
    # 这个方法用来更新训练进度，并估算剩余训练时间。
    def _timer_update(self):
        # self.global_step 计算从训练开始到当前的总步数。这包括完成的周期数和当前周期中的批次数。
        self.global_step = (self.epoch_id-self.epoch_to_start) * self.steps_per_epoch + self.batch_id
        # 更新进度条，显示当前训练的完成百分比。
        self.timer.update_progress((self.global_step + 1) / self.total_steps)
        # est = self.timer.estimated_remaining() 估算剩余训练时间。
        est = self.timer.estimated_remaining()
        #  计算到目前为止的平均实例每秒处理数。这有助于估计在当前速度下完成剩余训练所需的时间。
        imps = (self.global_step + 1) * self.batch_size / self.timer.get_stage_elapsed()
        return imps, est
    # 这个方法用来保存当前训练状态的检查点，包括模型参数、优化器状态和学习率调度器状态。
    def _save_checkpoint(self, ckpt_name):
        # 使用一个字典定义了需要保存的内容，包括当前的训练周期 ID、最佳验证准确率、最佳训练周期 ID、模型的状态字典、优化器的状态字典和学习率调度器的状态字典。
        torch.save({
            'epoch_id': self.epoch_id,
            'best_val_f1': self.best_val_f1,
            'best_epoch_id': self.best_epoch_id,
            'model_G_state_dict': self.net_G.state_dict(),
            'optimizer_G_state_dict': self.optimizer_G.state_dict(),
            'exp_lr_scheduler_G_state_dict': self.exp_lr_scheduler_G.state_dict(),
        }, os.path.join(self.checkpoint_dir, ckpt_name))
    # 这个方法用来更新学习率调度器，通常在每个训练周期结束时调用。
    def _update_lr_schedulers(self):
        self.exp_lr_scheduler_G.step()
    # 这个方法用来更新性能度量，计算预测结果和真实标签之间的一致性。
    def _update_metric(self, labels):
        """
        update metric
        """
        target = labels.to(self.device).detach()# .detach()是pytorch得一个方法，用于从当前计算图中分离出张量。，从而阻止梯度计算。

        G_pred = self.G_pred.detach()
        # 使用 torch.argmax 函数找到预测结果中概率最高的类别索引。dim=1 表示沿着一个维度（通常是特征维度）进行操作。

        G_pred = torch.argmax(G_pred, dim=1) # 输出的形状是B 256 256
        # 调用 self.running_metric.update_cm 方法更新评估指标
        current_score = self.running_metric.update_cm(pr=G_pred.cpu().numpy(), gt=target.cpu().numpy())
        # 返回当前F1指标
        return current_score

    # 这个方法用来收集和记录当前批次的状态，例如准确率、实例每秒（IMPS）、剩余时间等。
    def _collect_running_batch_states(self, labels):
        # 调用 _update_metric 方法计算当前批次的F1。
        running_f1 = self._update_metric(labels)
        # 获取训练数据加载器的长度，如果是验证模式，则获取验证数据加载器的长度。
        m = len(self.train_dataloader)
        if self.is_training is False:
            m = len(self.val_dataloader)
        # 调用 _timer_update 方法获取实例每秒和剩余时间。
        imps, est = self._timer_update()
        # 每100个批次记录一次状态。
        if np.mod(self.batch_id, 100) == 1:
            # 构建一个字符串消息，包含训练或验证模式、当前周期和总周期、当前批次和总批次、实例每秒、剩余时间、总损失和运行准确率。
            message = '是否是训练: %s. [%d,%d][%d,%d], 每秒实例: %.2f, 剩余时间: %.2fh, 总损失: %.5f, running_mf1: %.5f\n' %\
                      (self.is_training, self.epoch_id, self.max_num_epochs-1, self.batch_id, m,
                        imps*self.batch_size, est,self.G_loss.item(), running_f1)
            self.logger.write(message)

    # 这个方法用来收集一个训练周期结束时的各种性能指标。
    def _collect_epoch_states(self):
        # 调用性能度量工具的 get_scores 方法获取当前周期的所有性能指标。
        scores = self.running_metric.get_scores()
        # 将度量结果中的 mf1（多类别 F1 分数）作为当前周期的准确率。
        self.epoch_mf1 = scores['mf1']
        # 将训练模式、周期编号、总周期数和当前周期的 F1 分数记录到日志中。
        self.logger.write('是否训练: %s. 批次 %d / %d, epoch_mF1= %.5f\n' %
              (self.is_training, self.epoch_id, self.max_num_epochs-1, self.epoch_mf1))
        # 初始化一个字符串，用于存储所有性能指标的文本表示。
        message = ''
        # 遍历所有性能指标，将它们添加到 message 字符串中。
        for k, v in scores.items():
            message += '%s: %.5f ' % (k, v)
        # 将所有性能指标的文本表示写入日志。
        self.logger.write(message+'\n')
        self.logger.write('\n')
    # 这个方法用来在每个训练周期结束时保存检查点，并在性能提升时更新最佳模型。
    def _update_checkpoints(self):
        # 保存当前周期的检查点，文件名为 last.pt。
        self._save_checkpoint(ckpt_name='last.pt')
        # ：记录日志，说明最新模型已更新，并显示当前周期的准确率和历史最佳准确率。
        self.logger.write('最近一次模型更新. Epoch_mf1=%.4f, Historical_best_f1=%.4f (at epoch %d)\n'
              % (self.epoch_mf1, self.best_val_f1, self.best_epoch_id))
        self.logger.write('\n')
        # 如果当前周期的准确率超过了历史最佳准确率，则更新历史最佳准确率和最佳周期编号。
        if self.epoch_mf1 > self.best_val_f1:
            self.best_val_f1 = self.epoch_mf1
            self.best_epoch_id = self.epoch_id
            self._save_checkpoint(ckpt_name='best.pt')
            self.logger.write('*' * 20 + '最好的模型更新啦！！！！！！！！！！！！！！！！！！！！！！！\n' + '*' * 20)
            self.logger.write('\n')
    # 这个方法用来清除性能度量工具中的缓存数据，准备开始下一个周期的度量。
    def _clear_cache(self):
        # 清除性能度量工具中的所有缓存数据，以便在下一个训练周期开始时重新收集数据。
        self.running_metric.clear()

    # 前向传播 这个方法负责将输入数据传递到模型中进行前向传播。
    def _forward_pass(self, imageA, imageB):

        img_in1 = imageA.to(self.device)
        img_in2 = imageB.to(self.device)
        self.img1,self.img2,self.img3,self.img4,self.G_pred = self.net_G(img_in1, img_in2)
        # self.img1,self.img2,self.img3,self.img4,self.map1,self.map2,self.map3,self.map4,self.G_pred = self.net_G(img_in1, img_in2)
        #self.G1_12,self.G2_12,self.G1_22,self.G2_22,self.G1_32,self.G2_32,self.G1_42,self.G2_42,self.G1_11,self.G2_11,self.G1_21,self.G2_21,self.G1_31,self.G2_31,self.G1_41,self.G2_41,self.img1,self.img2,self.img3,self.img4,self.G_pred = self.net_G(img_in1, img_in2)
        #self.G_pred = self.net_G(img_in1, img_in2)

    # 这个方法负责计算损失并进行反向传播。
    def _backward_G(self,imageA,imageB,labels):

        gt = labels.to(self.device)
        self.G_loss3 = L1_loss(imageA, self.img1) + L1_loss(imageA, self.img2) + L1_loss(imageB, self.img3) + L1_loss(imageB, self.img4)
        self.G_loss4 = compute_kl_loss1(imageA, self.img1) + compute_kl_loss1(imageA, self.img2) + compute_kl_loss1(imageB, self.img3) + compute_kl_loss1(imageB, self.img4)
        self.G_loss1 = 10 * self._pxl_loss(self.G_pred, gt) + self.my_loss(self.G_pred, gt)
        # *10之后 25.089   10.280   4.191   6.169
        self.G_loss = 0 * self.G_loss3 + self.G_loss1
        #self.G_loss = self.G_loss4 + self.G_loss5

        # 计算反向传播的损失，为之后的优化计算梯度
        self.G_loss.backward()
    # 它是训练模型的主要入口。
    def train_models(self):
        # 调用 _load_checkpoint 方法加载最后保存的检查点，以便从上次训练中断的地方继续训练。
        print(self.steps_per_epoch)
        print(self.total_steps)

        self._load_checkpoint()

        # loop over the dataset multiple times
        # 循环遍历从 epoch_to_start 到 max_num_epochs 的每个训练周期。
        for self.epoch_id in range(self.epoch_to_start, self.max_num_epochs):

            ################## train #################
            ##########################################
            # 记录当前训练周期的编号和从训练开始到当前周期的总时间。
            self.logger.write('\nEpoch: %d' % (self.epoch_id + 1))
            end = time.time()  # 每个epoch开始时间
            hua = end - self.begin_total
            self.logger.write(f'\nafter：{hua // 3600} Hour {(hua % 3600) // 60} Minute {int(hua % 60)} second ')  # 每个epoch开始的绝对时间
            # 清除性能度量工具的缓存。设置 is_training 为 True，表示当前是训练模式。将模型设置为训练模式。记录当前的学习率。
            self._clear_cache()
            self.is_training = True
            self.net_G.train()
            self.logger.write('\nlr: %0.7f\n' % self.optimizer_G.param_groups[0]['lr'])
            # 循环遍历训练数据加载器中的每个批次。
            for self.batch_id, (imageA, imageB, labels) in enumerate(self.train_dataloader, 0):
                # 调用 _forward_pass 方法进行前向传播。
                self._forward_pass(imageA, imageB)
                # update G
                # 清除梯度，调用 _backward_G 方法进行反向传播，并更新模型参数。
                self.optimizer_G.zero_grad()
                self._backward_G(imageA,imageB,labels)
                self.optimizer_G.step()
                # 收集当前批次的状态并更新进度条。
                self._collect_running_batch_states(labels)
            # 调用 _collect_epoch_states 方法收集当前训练周期的性能指标。
            self._collect_epoch_states()


            # 调用 _update_lr_schedulers 方法更新学习率调度器。
            self._update_lr_schedulers()

            ################## Eval ##################
            ##########################################
            # 记录开始评估的日志信息。
            self.logger.write('评估开始...\n')
            # 清除性能度量工具的缓存，设置为评估模式。
            self._clear_cache()
            self.is_training = False
            self.net_G.eval()

            # Iterate over data.
            # 循环遍历验证数据加载器中的每个批次，进行前向传播并收集状态。
            for self.batch_id, (imageA, imageB, labels) in enumerate(self.val_dataloader, 0):
                with torch.no_grad():
                    self._forward_pass(imageA, imageB)# 为什么不需要[8] 反向传播又是怎么进行的
                self._collect_running_batch_states(labels)
            # 收集评估周期的性能指标。
            self._collect_epoch_states()
            self._timer_update()
            ########### Update_Checkpoints ###########
            ##########################################
            # 调用 _update_checkpoints 方法保存当前周期的检查点，并更新最佳模型。
            self._update_checkpoints()
            # 记录当前训练周期结束的时间，并计算该周期所需的总时间。
            end_these = time.time()  # 每个epoch结束时间
            hua1 = end_these - end
            self.logger.write(
                f'\nthis epoch time：{hua1 // 3600} hour {(hua1 % 3600) // 60} minute {int(hua1 % 60)} second\n')  # 每个epoch需要的时间
        # 记录训练结束的总时间，并计算从开始训练到结束所需的总时间。
        self.end_total = time.time()  # 总训练完的绝对时间
        hua2 = self.end_total - self.begin_total
        self.logger.write(f'\ntotal time：{hua2 // 3600} hour {(hua2 % 3600) // 60} minute {int(hua2 % 60)} second\n')  # 总训练完的相对时间



if __name__ == '__main__':
    train_dataloader, test_dataloader, val_dataloader, batch_size = load_img('E:/dataset/TempLevircd_FC/')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CDTrainer(train_dataloader, val_dataloader, batch_size)
    model.train_models()

