"""fitrec_attn.py

This code provides the FitRec-Attn model introduced by Ni, et al [2019 WWW].

The model is trained to predict the next observed heart rate reading given 
a set of user attributes and previous sequences of current and past workouts.

The vast majority of this code is derived from the original author's repo:
https://github.com/nijianmo/fit-rec/ which we have updated and extended to 
perform different tasks:
1) Predicting a window of K steps ahead given the current workout
2) Predicting the current workout type (running, walking, biking, etc.)

--------------------------------------------------------------------------------
Jonas Guan, Natalie Dullerud and Taylor Killian
University of Toronto, CSC2515 Final Project -- Fall 2020

################################################################################
Notes:
TWK - Currently, this is just the base implementation of the model to get a feel
    for how it runs and trains. Once I get it to train in a stable manner, I will
    update the batch sampling code and the training scripts to support the tasks
    we're interested in.

"""

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F

from data_interpolate import dataInterpreter, metaDataEndomondo

import matplotlib
import argparse
import datetime as dt
import itertools
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
import os

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class contextEncoder(nn.Module):
    """Encodes a sequence and a scalar contextual variable using LSTMs, the hidden states of each
    LSTM are then combined and projected into a common embedding as the `context` for the user"""
    def __init__(self, input_size, hidden_size, output_size):
        super(contextEncoder, self).__init__()
        self.input_size = input_size
        self.hidden_dim = hidden_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.context_dim = self.output_size
        self.context_layer_1 = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_dim, batch_first=True)
        self.context_layer_2 = nn.LSTM(input_size=1, hidden_size=self.hidden_dim, batch_first=True)
        self.dropout_rate = 0.2
        print('context encoder dropout: {}'.format(self.dropout_rate))
        self.dropout = nn.Dropout(self.dropout_rate)
        self.project = nn.Linear(self.hidden_dim * 2, self.context_dim)

    def forward(self, context_input_1, context_input_2):
        context_input_1 = self.dropout(context_input_1)
        context_input_2 = self.dropout(context_input_2)

        hidden_1 = torch.zeros(1, context_input_1.size(0), self.hidden_size).to(device)
        cell_1 = torch.zeros_like(hidden_1).to(device)
        hidden_2 = torch.zeros(1, context_input_2.size(0), self.hidden_size).to(device)
        cell_2 = torch.zeros_like(hidden_2).to(device)

        self.context_layer_1.flatten_parameters()
        outputs_1, lstm_states_1 = self.context_layer_1(context_input_1, (hidden_1, cell_1))
        context_embedding_1 = outputs_1
        self.context_layer_2.flatten_parameters()
        outputs_2, lstm_states_2 = self.context_layer_2(context_input_2, (hidden_2, cell_2))
        context_embedding_2 = outputs_2

        context_embedding = self.project(torch.cat([context_embedding_1, context_embedding_2],dim=-1))

        return context_embedding

class encoder(nn.Module):
    """Encoder module"""
    def __init__(self, input_size, hidden_size, T, attr_embeddings, dropout=0.1):
        # input size: number of underlying factors (81)
        # T: number of time steps (10)
        # hidden_size: dimension of the hidden state
        super(encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.T = T
        self.user_embedding  = attr_embeddings[0]
        self.sport_embedding = attr_embeddings[1]
        self.dropout_rate = dropout
        self.dropout = nn.Dropout(dropout)
        print("encoder dropout: {}".format(self.dropout_rate))

        self.lstm_layer = nn.LSTM(input_size=input_size, hidden_size=hidden_size)
        self.attn_linear = nn.Linear(in_features = 2*hidden_size+T, out_features=1)

    def forward(self, attr_inputs, context_embedding, input_variable):
        for attr in attr_inputs:
            attr_input = attr_inputs[attr]
            if attr == "user_input":
                attr_embed = self.user_embedding(attr_input)
            if attr == "sport_input":
                attr_embed = self.sport_embedding(attr_input)
            input_variable = torch.cat([attr_embed, input_variable], dim=-1)

        input_variable = torch.cat([context_embedding, input_variable], dim=-1)

        input_data = input_variable

        input_weighted = torch.zeros(input_data.size(0), self.T, self.input_size).to(device)
        input_encoded = torch.zeros(input_data.size(0), self.T, self.hidden_size).to(device)
        hidden = torch.zeros(1, input_data.size(0), self.hidden_size).to(device)
        cell = torch.zeros_like(hidden).to(device)

        for t in range(self.T):
            # Eqn 8: concatenate the hidden states with each predictor
            x = torch.cat((hidden.repeat(self.input_size, 1, 1).permute(1, 0, 2),
                           cell.repeat(self.input_size, 1, 1).permute(1, 0, 2),
                           input_data.permute(0, 2, 1)), dim=2)  # batch_size * input_size * (2*hidden_size + T)
            # Eqn 9: Get attention weights
            x = self.attn_linear(x.view(-1, 2*self.hidden_size + self.T))  # (batch_size * input_size) * 1
            attn_weights = F.softmax(x.view(-1, self.input_size), dim=-1)  # batch_size * input_size, attn weights sum up to 1
            # Eqn 10: LSTM
            weighted_input = torch.mul(attn_weights, input_data[:, t, :])  # batch_size * input_size

            self.lstm_layer.flatten_parameters()
            _, lstm_states = self.lstm_layer(weighted_input.unsqueeze(0), (hidden, cell))
            hidden = lstm_states[0]
            cell = lstm_states[1]
            # Save output
            input_weighted[:, t, :] = weighted_input
            input_encoded[:, t, :] = hidden
        return input_weighted, input_encoded


class decoder(nn.Module):
    """Decode the embedded hidden state to predict the next time window"""
    def __init__(self, encoder_hidden_size, decoder_hidden_size, T):
        super(decoder, self).__init__()

        self.T = T
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size

        self.attn_layer = nn.Sequential(nn.Linear(2*decoder_hidden_size + encoder_hidden_size, encoder_hidden_size),
                                        nn.Tanh(),
                                        nn.Linear(encoder_hidden_size, 1))  # Here the last layer is trying to predict the next time step
        self.lstm_layer = nn.LSTM(input_size=1, hidden_size=decoder_hidden_size)
        self.fc = nn.Linear(encoder_hidden_size+1, 1)
        self.fc_final = nn.Linear(decoder_hidden_size + encoder_hidden_size, T)  # Final prediction of the next time step

        self.fc.weight.data.normal_()

    def forward(self, input_encoded, y_history):
        # input_encoded: batch_size * T * encoder_hidden_size
        # y_history: batch_size * (T-1)
        hidden = torch.zeros(1, input_encoded.size(0), self.decoder_hidden_size).to(device)
        cell = torch.zeros_like(hidden).to(device)

        for t in range(self.T):
            # Eqn 12-13: compute attention weights
            ## batch_size * T * (2*decoder_hidden_size + encoder_hidden_size)
            x = torch.cat((hidden.repeat(self.T, 1, 1).permute(1, 0, 2),
                           cell.repeat(self.T, 1, 1).permute(1, 0, 2), 
                           input_encoded), dim=2)
            x = F.softmax(self.attn_layer(x.view(-1, 2*self.decoder_hidden_size+self.encoder_hidden_size)).view(-1, self.T), dim=-1)

            # Eqn 14: compute context vector
            context = torch.bmm(x.unsqueeze(1), input_encoded)[:, 0, :]  # batch_size * encoder_hidden_size

            if t < self.T - 1:
                # Eqn 15
                y_tilde = self.fc(torch.cat((context, y_history[:, t].unsqueeze(1)), dim=1))  # batch_size * 1

                # Eqn 16: LSTM
                self.lstm_layer.flatten_parameters()
                _, lstm_output = self.lstm_layer(y_tilde.unsqueeze(0), (hidden, cell))
                hidden = lstm_output[0]  # 1 * batch_size * decoder_hidden_size
                cell = lstm_output[1]  # 1 * batch_size * decoder_hidden_size

        # Eqn 22: final output (hallelujah!)
        y_pred = self.fc_final(torch.cat((hidden[0], context), dim=1))

        # return y_pred.view(y_pred.size(0))
        return y_pred



class sport_decoder(nn.Module):
    """Using the embedded hidden state, predict the sport"""
    def __init__(self, encoder_hidden_size):
        super(sport_decoder, self).__init__()

        self.encoder_hidden_size = encoder_hidden_size
        self.interim_hidden_size = 8
        
        self.fc1 = nn.Linear(self.encoder_hidden_size, self.interim_hidden_size)
        self.final_fc = nn.Linear(self.interim_hidden_size, 1)
        self.m = nn.ReLU()

    def forward(self, input_encoded):

        interim = self.m(self.fc1(input_encoded))
        output = F.softmax(self.final_fc(interim), dim=-1)

        return output


class zone_decoder(nn.Module):
    """Using the embedded hidden state, predict whether user hits their target HR zone"""
    def __init__(self, encoder_hidden_size):
        super(zone_decoder, self).__init__()

        self.encoder_hidden_size = encoder_hidden_size
        self.interim_hidden_size = 8

        self.fc1 = nn.Linear(self.encoder_hidden_size, self.interim_hidden_size)
        self.final_fc = nn.Linear(self.interim_hidden_size, 1)
        self.m = nn.ReLU()

    def forward(self, input_encoded):

        interim = self.m(self.fc1(input_encoded))
        output = F.softmax(self.final_fc(interim), dim=-1)

        return output


## PUT ALL THESE PIECES TOGETHER TO FORM THE FITREC-ATTN MODEL
class da_rnn:
    def __init__(self, encoder_hidden_size=64, decoder_hidden_size=64, 
                T=10, learning_rate = 0.01, batch_size=5120, parallel=True, debug=False, test_model_path=None, predict_sport=False, predict_zone=False):

        self.T = T
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        path = "../data"
        self.model_save_location = "/scratch/gobi2/tkillian/endomondo/model_states/"
        self.summaries_dir = path + "/fitrec-attn/logs/"
        self.data_path = "endomondoHR_proper.json"
        self.patience = 3  # [3, 5, 10]
        self.max_epochs = 50
        self.zMultiple = 5

        self.predict_sport = predict_sport
        self.predict_zone = predict_zone

        self.pretrain, self.includeUser, self.includeSport, self.includeTemporal = False, True, False, True  # Defaults...
        print("include pretrain/user/sport/temporal = {}/{}/{}/{}".format(self.pretrain, self.includeUser, self.includeSport, self.includeTemporal))

        self.model_file_name = []
        if self.includeUser:
            self.model_file_name.append("userId")
        if self.includeSport:
            self.model_file_name.append("sport")
        if self.includeTemporal:
            self.model_file_name.append("context")
        if self.predict_sport:
            self.model_file_name.append("predict_sport")
        if self.predict_zone:
            self.model_file_name.append("predict_zone")
        print(self.model_file_name)

        self.user_dim = 5
        self.sport_dim = 5

        self.trainValidTestSplit = [0.8, 0.1, 0.1]
        self.targetAtts = ['heart_rate']
        self.heartRateTarget = 0.84
        self.medianAge = 35
        self.targetDuration = 12 * 60  # 12 minutes, in seconds
        self.inputAtts = ['derived_speed', 'altitude']

        self.trimmed_workout_len = 300
        self.num_steps = self.trimmed_workout_len

        # Should the data values be scaled to their z_scores with the z-multiple?
        self.scale_toggle = True
        self.scaleTargets = False

        self.trainValidTestFN = path +'/'+ self.data_path.split(".")[0] + "_temporal_dataset.pkl"

        
        # Prepare the data reader -- Preprocess the data for use
        self.endo_reader = dataInterpreter(self.T, self.inputAtts, self.includeUser, self.includeSport,
                                           self.includeTemporal, self.targetAtts, self.heartRateTarget, self.medianAge, self.targetDuration,
                                           fn=self.data_path, scaleVals=self.scale_toggle, trimmed_workout_len=self.trimmed_workout_len,
                                           scaleTargets=self.scaleTargets, trainValidTestSplit=self.trainValidTestSplit,
                                           zMultiple=self.zMultiple, trainValidTestFN=self.trainValidTestFN)

        self.endo_reader.preprocess_data()

        self.input_dim = self.endo_reader.input_dim
        self.output_dim = self.endo_reader.output_dim

        self.train_size = len(self.endo_reader.trainingSet)
        self.valid_size = len(self.endo_reader.validationSet)
        self.test_size = len(self.endo_reader.testSet)

        modelRunIdentifier = dt.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
        self.model_file_name.append(modelRunIdentifier)  # Append a unique identifier to the filenames
        self.model_file_name = "_".join(self.model_file_name)

        self.model_save_location += self.model_file_name + "/"
        print(self.model_save_location)

        # Build model
        # model
        self.num_users = len(self.endo_reader.oneHotMap['userId'])
        self.num_sports = len(self.endo_reader.oneHotMap['sport'])
        self.num_genders = len(self.endo_reader.oneHotMap['gender'])

        self.input_size = self.input_dim
        self.attr_num = 0
        self.attr_embeddings = []
        user_embedding = nn.Embedding(self.num_users, self.user_dim)
        torch.nn.init.xavier_uniform(user_embedding.weight.data)
        self.attr_embeddings.append(user_embedding)
        sport_embedding = nn.Embedding(self.num_sports, self.sport_dim)
        self.attr_embeddings.append(sport_embedding)

        if self.includeUser:
            self.attr_num += 1
            self.input_size += self.user_dim
        if self.includeSport:
            self.attr_num += 1
            self.input_size += self.sport_dim

        if self.includeTemporal:
            self.context_dim = int(encoder_hidden_size / 2)
            self.input_size += self.context_dim
            self.context_encoder = contextEncoder(input_size=self.input_dim+1, hidden_size=encoder_hidden_size,
                                        output_size=self.context_dim).to(device)
        
        for attr_embedding in self.attr_embeddings:
                attr_embedding = attr_embedding.to(device)

        self.encoder = encoder(input_size=self.input_size, hidden_size=encoder_hidden_size, T=T,
                               attr_embeddings=self.attr_embeddings).to(device)
        self.decoder = decoder(encoder_hidden_size=encoder_hidden_size,
                               decoder_hidden_size=decoder_hidden_size, T=T).to(device)
        if self.predict_sport:
            self.decoder_sport = sport_decoder(encoder_hidden_size=encoder_hidden_size).to(device)
        if self.predict_zone:
            self.decoder_meetHRzone = zone_decoder(encoder_hidden_size=encoder_hidden_size).to(device)
        
        if parallel:
            self.encoder = nn.DataParallel(self.encoder)
            self.context_encoder = nn.DataParallel(self.context_encoder)
            self.decoder = nn.DataParallel(self.decoder)
            if self.predict_sport:
                self.decoder_sport = nn.DataParallel(self.decoder_sport)
            if self.predict_zone:
                self.decoder_meetHRzone = nn.DataParallel(self.decoder_meetHRzone)


        wd1 = 0.002
        wd2 = 0.005
        if self.includeUser:
            print("user weight decay: {}".format(wd1))
        if self.includeSport:
            print("sport weight decay: {}".format(wd2))


        self.encoder_optimizer = optim.Adam([
                {'params': [param for name, param in self.encoder.named_parameters() if 'user_embedding' in name], 'weight_decay':wd1},
                {'params': [param for name, param in self.encoder.named_parameters() if 'sport_embedding' in name], 'weight_decay':wd2},
                {'params': [param for name, param in self.encoder.named_parameters() if 'embedding' not in name]}
            ], lr=learning_rate)
        
        self.context_encoder_optimizer = optim.Adam(params=filter(lambda p: p.requires_grad, self.decoder.parameters()), lr = learning_rate)
        self.decoder_optimizer = optim.Adam(params=filter(lambda p: p.requires_grad, self.decoder.parameters()), lr = learning_rate)
        self.loss_func = nn.MSELoss(size_average=True)
        if self.predict_sport:
            self.decoder_sport_optimizer = optim.Adam(params=filter(lambda p: p.requires_grad, self.decoder_sport.parameters()), lr = learning_rate)
            self.loss_func_sport = nn.CrossEntropyLoss()
        if self.predict_zone:
            self.decoder_meetHRzone_optimizer = optim.Adam(params=filter(lambda p: p.requires_grad, self.decoder_meetHRzone.parameters()), lr = learning_rate)
            self.loss_func_meetHRzone = nn.CrossEntropyLoss()

        if test_model_path:
            checkpoint = torch.load(test_model_path)
            self.encoder.load_state_dict(checkpoint['en'])
            self.context_encoder.load_state_dict(checkpoint['context_en'])
            self.decoder.load_state_dict(checkpoint['de'])
            if predict_sport:
                self.decoder_sport.load_state_dict(checkpoint['de_sport'])
            if predict_zone:
                self.decoder_meetHRzone.load_state_dict(checkpoint['de_zone'])
            print("test model: {}".format(test_model_path))

    # Prepare a provided batch from the data for use in training the FitRec-Attn Model
    def get_batch(self, batch):
        attr_inputs = {}
        if self.includeUser:
            user_input = batch[0]['user_input']
            attr_inputs['user_input'] = user_input
        if self.includeSport:
            sport_input = batch[0]['sport_input']
            attr_inputs['sport_input'] = sport_input

        for attr in attr_inputs:
            attr_input = attr_inputs[attr]
            attr_input = torch.from_numpy(attr_input).long().to(device)

            attr_inputs[attr] = attr_input

        context_input_1 = torch.from_numpy(batch[0]['context_input_1'][:,-self.T:,:]).float().to(device)
        context_input_2 = torch.from_numpy(batch[0]['context_input_2'][:,-self.T:,:]).float().to(device)

        input_variable = torch.from_numpy(batch[0]['input']).float().to(device)
        target_ts_variable = torch.from_numpy(batch[1]).float().to(device)
        target_sport_variable = torch.where(torch.from_numpy(batch[2]) == 13, torch.ones(batch[2].shape), torch.zeros(batch[2].shape)).long().to(device)
        target_meetHRZone_variable = torch.from_numpy(batch[3]).long().to(device)

        # TODO(TWK): This is where we can adjust how we predict the next K 
        # timesteps by adjusting the target variable. What needs to be done 
        # is to adapt how the batches passed to this method are formed so that 
        # the time sequences are longer (T-1)+K...... We can also provide the 
        # sport as target for that secondary task.
        y_history = target_ts_variable[:, :self.T, :].squeeze(-1)
        y_target = target_ts_variable[:, -self.T:, :].squeeze(-1)  # Convert to just pulling the the self.T steps into the "future"

        return attr_inputs, context_input_1, context_input_2, input_variable, y_history, y_target, target_sport_variable, target_meetHRZone_variable


    def train(self, n_epochs=30, print_every=400):

        # Initialize
        print('Initializing...')
        start_epoch = 0
        best_val_loss = None
        best_epoch_path = None
        best_valid_score = 9999999999
        best_epoch = 0

        for iteration in range(n_epochs):
            print('\n')
            print('-'*50)
            print('Iteration', iteration)

            epoch_start_time = time.time()
            start_time = time.time()

            # Train
            trainDataGen = self.endo_reader.generator_for_autotrain(self.batch_size, self.num_steps, "train")
            print_loss = 0
            hr_losses = 0
            sport_losses = 0
            zone_losses = 0
            for batch, training_batch in enumerate(trainDataGen):
                attr_inputs, context_input_1, context_input_2, input_variable, y_history, y_target, sport_target, zone_target = self.get_batch(training_batch)
                losses = self.train_iteration(attr_inputs, context_input_1, context_input_2, input_variable, y_history, y_target, sport_target, zone_target)
                if self.predict_sport and self.predict_zone:
                    loss, hr_loss, sport_loss, zone_loss = (*losses,)
                    sport_losses += sport_loss
                    zone_losses += zone_loss
                elif self.predict_sport:
                    loss, hr_loss, sport_loss = (*losses,)
                    sport_losses += sport_loss
                elif self.predict_zone:
                    loss, hr_loss, zone_loss = (*losses,)
                    zone_losses += zone_loss
                else:
                    loss, hr_loss = (*losses,)

                print_loss += loss
                hr_losses += hr_loss
                if batch % print_every == 0 and batch > 0:
                    cur_loss = print_loss / print_every
                    cur_hr_loss = hr_losses / print_every
                    elapsed = time.time() - start_time
                    print_string = '| epoch {:3d} | {:5d} batches | lr {:05.5f} | ms/batch {:5.2f} | loss {:5.3f} | hr_loss {:5.3f}'.format(
                          iteration, batch, self.learning_rate,
                          elapsed * 1000 / print_every, cur_loss, cur_hr_loss)
                    if self.predict_sport:
                        cur_sport_loss = sport_losses / print_every
                        print_string += ' | sport_loss {:5.3f}'.format(cur_sport_loss)
                    if self.predict_zone:
                        cur_zone_loss = zone_losses / print_every
                        print_string += ' | zone_loss {:5.3f}'.format(cur_zone_loss)

                    print(print_string)
                    
                    print_loss = 0
                    hr_losses = 0
                    sport_losses = 0
                    zone_losses = 0
                    start_time = time.time()

            # Evaluate
            validDataGen = self.endo_reader.generator_for_autotrain(self.batch_size, self.num_steps, "valid")
            val_loss = 0
            val_batch_num = 0
            for val_batch in validDataGen:
                val_batch_num += 1

                attr_inputs, context_input_1, context_input_2, input_variable, y_history, y_target, sport_target, zone_target = self.get_batch(val_batch)
                losses = self.evaluate(attr_inputs, context_input_1, context_input_2, input_variable, y_history, y_target, sport_target, zone_target)

                loss = losses[0]

                val_loss += loss
            val_loss /= val_batch_num
            print('-'*89)
            if not best_val_loss or val_loss <= best_val_loss:
                best_val_loss = val_loss
                best_epoch = iteration
                best_epoch_path = self.model_save_location+self.model_file_name+"_best"

                if not os.path.exists(self.model_save_location):
                    os.makedirs(self.model_save_location)

                save_dict = {
                    'epoch': iteration,
                    'en': self.encoder.state_dict(),
                    'context_en': self.context_encoder.state_dict(),
                    'de': self.decoder.state_dict(),
                    'en_opt': self.encoder_optimizer.state_dict(),
                    'context_en_opt': self.context_encoder_optimizer.state_dict(),
                    'de_opt': self.decoder_optimizer.state_dict(),
                    'loss': loss
                    }
                if self.predict_sport:
                    save_dict['de_sport'] = self.decoder_sport.state_dict()
                    save_dict['de_sport_opt'] = self.decoder_sport_optimizer.state_dict()
                if self.predict_zone:
                    save_dict['de_zone'] = self.decoder_meetHRzone.state_dict()
                    save_dict['de_zone_opt'] = self.decoder_meetHRzone_optimizer.state_dict()
                torch.save(save_dict, best_epoch_path)

            elif (iteration - best_epoch < self.patience):
                pass
            else:
                print("Stopped early at epoch: " + str(iteration))
                break

        # Load the best model to test
        if best_epoch_path:
            checkpoint = torch.load(best_epoch_path)
            self.encoder.load_state_dict(checkpoint['en'])
            self.context_encoder.load_state_dict(checkpoint['context_en'])
            self.decoder.load_state_dict(checkpoint['de'])
            if predict_sport:
                self.decoder_sport.load_state_dict(checkpoint['de_sport'])
            if predict_zone:
                self.decoder_meetHRzone.load_state_dict(checkpoint['de_zone'])
        print("best model: {}".format(best_epoch_path))

        # test
        testDataGen = self.endo_reader.generator_for_autotrain(self.batch_size, self.num_steps, "test")
        test_loss = 0
        test_hr_loss = 0
        test_sport_loss = 0
        test_zone_loss = 0
        test_batch_num = 0

        for test_batch in testDataGen:
            test_batch_num += 1
            
            attr_inputs, context_input_1, context_input_2, input_variable, y_history, y_target, sport_target, zone_target = self.get_batch(test_batch)
            losses = self.evaluate(attr_inputs, context_input_1, context_input_2, input_variable, y_history, y_target, sport_target, zone_target)
            if self.predict_sport and self.predict_zone:
                loss, hr_loss, sport_loss, zone_loss = (*losses,)
                test_sport_loss += sport_loss
                test_zone_loss += zone_loss
            elif self.predict_sport:
                loss, hr_loss, sport_loss = (*losses,)
                test_sport_loss += sport_loss
            elif self.predict_zone:
                loss, hr_loss, zone_loss = (*losses,)
                test_zone_loss += zone_loss
            else:
                loss, hr_loss = (*losses,)
            test_loss += loss
            test_hr_loss += hr_loss
        test_loss /= test_batch_num
        test_hr_loss /= test_batch_num
        test_sport_loss /= test_sport_loss
        test_zone_loss /= test_zone_loss
        print_string = '| test loss {:5.3f} | test hr_loss {:5.3f}'.format(test_loss, test_hr_loss)
        if self.predict_sport:
            print_string += ' | test sport_loss {:5.3f}'.format(test_sport_loss)
        if self.predict_zone:
            print_string += ' | test zone_loss {:5.3f}'.format(test_zone_loss)
        print('-'*89)
        print(print_string)
        print('-'*89)

    #################################################################
    ###  FULL MODEL TRAINING, EVALUATION AND PREDICTION METHODS
    #################################################################
    def train_iteration(self, attr_inputs, context_input_1, context_input_2, input_variable, y_history, y_target, target_sport, target_zone):
        '''TODO: UPDATE AND AUGMENT THIS TO EITHER 1) PREDICT K steps forward OR 2) PREDICT Sport category'''
        self.encoder.train()
        self.context_encoder.train()
        self.decoder.train()
        if self.predict_sport:
            self.decoder_sport.train()
        if self.predict_zone:
            self.decoder_meetHRzone.train()

        self.encoder_optimizer.zero_grad()
        self.context_encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()
        if self.predict_sport:
            self.decoder_sport_optimizer.zero_grad()
        if self.predict_zone:
            self.decoder_meetHRzone_optimizer.zero_grad()

        loss = 0


        context_embedding = self.context_encoder(context_input_1, context_input_2)
        input_weighted, input_encoded = self.encoder(attr_inputs, context_embedding, input_variable)
        y_pred = self.decoder(input_encoded, y_history)
        hr_loss = self.loss_func(y_pred, y_target)
        loss += hr_loss

        if self.predict_sport:
            pred_sport = self.decoder_sport(input_encoded)
            sport_loss = self.loss_func_sport(pred_sport, target_sport)
            loss += sport_loss

        if self.predict_zone:
            pred_zone = self.decoder_meetHRzone(input_encoded)
            zone_loss = self.loss_func_meetHRzone(pred_zone, target_zone)
            loss += zone_loss

        loss.backward()

        self.encoder_optimizer.step()
        self.context_encoder_optimizer.step()
        self.decoder_optimizer.step()
        if self.predict_sport:
            self.decoder_sport_optimizer.step()
        if self.predict_zone:
            self.decoder_meetHRzone_optimizer.step()

        return_losses = [loss.detach().item(), hr_loss.detach().item()]
        if self.predict_sport:
            return_losses += [sport_loss.detach().item()]
        if self.predict_zone:
            return_losses += [zone_loss.detach().item()]

        return return_losses

    def evaluate(self, attr_inputs, context_input_1, context_input_2, input_variable, y_history, y_target):
        self.encoder.eval()
        self.context_encoder.eval()
        self.decoder.eval()
        if self.predict_sport:
            self.decoder_sport.eval()
        if self.predict_zone:
            self.decoder_meetHRzone.eval()

        loss = 0

        context_embedding = self.context_encoder(context_input_1, context_input_2)
        input_weighted, input_encoded = self.encoder(attr_inputs, context_embedding, input_variable)
        y_pred = self.decoder(input_encoded, y_history)
        hr_loss = self.loss_func(y_pred, y_target)
        loss += hr_loss

        if self.predict_sport:
            pred_sport = self.decoder_sport(input_encoded)
            sport_loss = self.loss_func_sport(pred_sport, target_sport)
            loss += sport_loss

        if self.predict_zone:
            pred_zone = self.decoder_meetHRzone(input_encoded)
            zone_loss = self.loss_func_meetHRzone(pred_zone, target_zone)
            loss += zone_loss

        return_losses = [loss.detach().item(), hr_loss.detach().item()]
        if self.predict_sport:
            return_losses += [sport_loss.detach().item()]
        if self.predict_zone:
            return_losses += [zone_loss.detach().item()]

        return return_losses


def main(predict_sport=False, predict_zone=False):
    learning_rate = 0.005
    batch_size = 5120
    # batch_size = 10240
    # batch_size = 12800 # Gigantic batch size... Needs to be farmed across GPUs...
    hidden_size = 64
    # T = 20
    T=10
    print("learning_rate = {}, batch_size = {}, hidden_size = {}, T = {}".format(learning_rate, batch_size, hidden_size, T))
    model = da_rnn(parallel=True, T=T, encoder_hidden_size=hidden_size, decoder_hidden_size=hidden_size, learning_rate=learning_rate, batch_size=batch_size, predict_sport=predict_sport, predict_zone=predict_zone)

    model.train(n_epochs=50)

    '''
    plt.figure()
    plt.semilogy(range(len(model.iter_losses)), model.iter_losses)
    plt.show()

    plt.figure()
    plt.semilogy(range(len(model.epoch_losses)), model.epoch_losses)
    plt.show()

    plt.figure()
    plt.plot(y_pred, label='Predicted')
    plt.plot(model.y[model.train_size:], label="True")
    plt.legend(loc='upper left')
    plt.show()
    '''

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--predict_sport', dest='predict_sport', action='store_true')
    parser.add_argument('--predict_zone', dest='predict_zone', action='store_true')

    args = parser.parse_args()
    main(predict_sport=args.predict_sport, predict_zone=args.predict_zone)
