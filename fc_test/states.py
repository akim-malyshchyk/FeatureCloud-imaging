import os
import torch
from FeatureCloud.app.engine.app import AppState, app_state
from FeatureCloud.app.engine.app import Role
import numpy as np
from enum import Enum
from torchvision.transforms import transforms
import torch.optim as optim
from model import VGG16
from utils import ChestMNIST, train, val, test
import torch.nn as nn


class States(Enum):
    initial = 'initial'
    distribute_data = 'distribute_data'
    receive_data = 'receive_data'
    compute = 'compute'
    agg_results = 'agg_results'
    receive_aggregation = 'receive_aggregation'
    terminal = 'terminal'


@app_state(States.initial.value, Role.BOTH)
class InitialState(AppState):
    def register(self):
        self.register_transition(States.distribute_data.value, role=Role.COORDINATOR)
        self.register_transition(States.receive_data.value, role=Role.PARTICIPANT)

    def run(self):
        self.log(f'Starting Initialization for node {self.id} ...')
        if self.is_coordinator:
            return States.distribute_data.value
        else:
            return States.receive_data.value


@app_state(States.distribute_data.value, Role.COORDINATOR)
class DistributeDataState(AppState):
    def register(self):
        self.register_transition(States.receive_data.value, role=Role.COORDINATOR)

    def run(self):
        self.log(f'{self.id} Split data for {len(self.clients)} clients ...')

        self.log('Distribute initial model parameters ...')
        self.broadcast_data((0, np.random.random(10)))

        return States.receive_data.value


@app_state(States.receive_data.value, Role.BOTH)
class ReceiveDataState(AppState):
    def register(self):
        self.register_transition(States.compute.value, role=Role.BOTH)

    def run(self):
        self.log(f'{self.id} Loading data ...')
        # Load the client data and split it into training and test set
        sorted_array = sorted(self.clients)
        client_number = sorted_array.index(self.id)

        input_root = f'app/data/client_{client_number}/data.npz'

        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        train_dataset = ChestMNIST(root=input_root, split='train', transform=transform)
        val_dataset = ChestMNIST(root=input_root, split='val', transform=transform)
        test_dataset = ChestMNIST(root=input_root, split='test', transform=transform)

        self.store('train', train_dataset)
        self.store('val', val_dataset)
        self.store('test', test_dataset)

        self.log(f'{self.id} loaded data ...')

        return States.compute.value


@app_state(States.compute.value, Role.BOTH)
class ComputeState(AppState):
    def register(self):
        self.register_transition(States.agg_results.value, role=Role.COORDINATOR)
        self.register_transition(States.compute.value, role=Role.PARTICIPANT)
        self.register_transition(States.terminal.value, role=Role.BOTH)

    def run(self):
        self.log(f'{self.id} performing compute ...')

        output_root = '/mnt/output'
        if not os.path.exists(output_root):
            os.mkdir(output_root)

        train_dataset = self.load('train')
        val_dataset = self.load('val')
        test_dataset = self.load('test')

        self.log(f'Received as {self.id} the data...')

        start_epoch = 0
        end_epoch = 2
        batch_size = 128
        val_auc_list = []

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

        self.log('Train model...')
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.log(f'Device "{device.type}" will be used for {self.id}...')

        iteration_number, states = self.await_data()
        self.log(f'Iter[{iteration_number}] Received newest coefficients...')
        self.log(f'{self.id}[{"Coordinator" if self.is_coordinator else "Participants"}]: training model ...')

        # Create a model with the latest model parameters
        model = VGG16(in_channels=1, num_classes=14).to(device)
        if iteration_number:
            model.load_state_dict(states)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.BCEWithLogitsLoss()

        if iteration_number == -1:
            self.log('Final Test model...')

            auc, acc = test(model, train_loader, device)
            self.log(f'Train data. AUC: {auc:.5f} ACC: {acc:.5f}')
            auc, acc = test(model, val_loader, device)
            self.log(f'Val data.   AUC: {auc:.5f} ACC: {acc:.5f}')
            auc, acc = test(model, test_loader, device)
            self.log(f'Test data.  AUC: {auc:.5f} ACC: {acc:.5f}')

            return States.terminal.value
        else:
            self.log(f'Training local model one more time ...')

            for epoch in range(start_epoch, end_epoch):
                train(model, optimizer, criterion, train_loader, device)
                auc = val(model, val_loader, device, val_auc_list, output_root, epoch)
                self.log(f'epoch {epoch + 1}. AUC: {auc:.5f}')

            auc_list = np.array(val_auc_list)
            index = auc_list.argmax()
            self.log(f'epoch {index} is the best model')

            restore_model_path = os.path.join(output_root, f'ckpt_{index}_auc_{auc_list[index]:.5f}.pth')
            model.load_state_dict(torch.load(restore_model_path)['net'])

            self.log('Test model...')

            auc, acc = test(model, train_loader, device)
            self.log(f'AUC: {auc:.5f} ACC: {acc:.5f}')
            auc, acc = test(model, val_loader, device)
            self.log(f'AUC: {auc:.5f} ACC: {acc:.5f}')
            auc, acc = test(model, test_loader, device)
            self.log(f'AUC: {auc:.5f} ACC: {acc:.5f}')

            self.log(f'Send to coordinator updated coefficients...')
            self.send_data_to_coordinator((torch.load(restore_model_path)['net'],))

        if self.is_coordinator:
            return States.agg_results.value
        else:
            return States.compute.value


@app_state(States.agg_results.value, Role.COORDINATOR)
class AggregationState(AppState):
    def register(self):
        self.register_transition(States.compute.value, role=Role.COORDINATOR)
        self.iteration = 0
        self.store('iteration', self.iteration)

    def run(self):
        self.iteration = self.load('iteration')
        self.log(f'{self.id}[{"Coordinator" if self.is_coordinator else "Participants"}, Iteration {self.iteration}')

        data = self.await_data(len(self.clients), is_json=False)
        self.log(f'Aggregation data received...')

        agg_data = np.mean(data, axis=0)
        self.log(f'Data aggregated successfully...')

        self.iteration += 1
        self.store('iteration', self.iteration)

        # Stop the process after 1 iteration, but we can fit the models more than once
        if self.iteration >= 1:
            self.iteration = -1

        self.log(f'{self.id} send results back ...')
        self.broadcast_data((self.iteration, agg_data[0]))
        return States.compute.value


@app_state(States.receive_aggregation.value, Role.BOTH)
class RecvAggregationState(AppState):
    def register(self):
        self.register_transition(States.terminal.value, role=Role.BOTH)

    def run(self):
        data = self.await_data(1)
        self.log(f'Received results {data}')
        return States.terminal.value
