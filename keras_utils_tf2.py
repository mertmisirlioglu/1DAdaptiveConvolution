from __future__ import print_function

import numpy as np
from tensorflow.keras.callbacks import Callback


class PrintLayerVariableStats(Callback):
    def __init__(self,name,var,stat_functions,stat_names,not_trainable=False):
        
        self.layername = name
        self.varname = var
        self.stat_list = stat_functions
        self.stat_names = stat_names
        self.not_trainable = not_trainable

    def setVariableName(self,name, var):
        self.layername = name
        self.varname = var
    def on_train_begin(self, logs={}):
        all_params = self.model.get_layer(self.layername).weights
        all_weights = self.model.get_layer(self.layername).get_weights()

        for i,p in enumerate(all_params):
            #print(p.name)
            if (p.name.find(self.varname)>=0):
                stat_str = [n+str(s(all_weights[i])) for s,n in zip(self.stat_list,self.stat_names)]
                print("Stats for", p.name, stat_str)

        #def on_batch_end(self, batch, logs={}):
        #    self.record.append(logs.get('loss'))

    def on_epoch_end(self, epoch, logs={}):
        all_params = self.model.get_layer(self.layername).weights
        
        all_weights = self.model.get_layer(self.layername).get_weights()
            
        for i,p in enumerate(all_params):
            # print(p.name)
            if (p.name.find(self.varname)>=0):
                stat_str = [n+str(s(all_weights[i])) for s,n in zip(self.stat_list,self.stat_names)]
                print("\n Stats for", p.name, stat_str)


class ClipCallback(Callback):
    def __init__(self, varname, clips=[]):
        self.varname = varname
        self.clips = clips
        #self.model = model
        
    def on_batch_end(self, batch, logs={}):
        all_weights = self.model.trainable_weights
            
        for i,p in enumerate(all_weights):
            #print(p.name)
            if (p.name.find(self.varname)>=0):
                pval = p.numpy()
                clipped = np.clip(pval,self.clips[0],self.clips[1])
                p.assign(clipped)
                #K.set_value(p,K.clip(p,self.clips[0],self.clips[1]))
                #tf.print("Clipped", p.name, output_stream=sys.stdout)


class RecordWeights(Callback):
    def __init__(self, name, var):
        self.layername = name
        self.varname = var

    def setVariableName(self, name, var):
        self.layername = name
        self.varname = var

    def on_train_begin(self, logs={}):
        self.record = []
        all_params = self.model.get_layer(self.layername)._trainable_weights
        all_weights = self.model.get_layer(self.layername).get_weights()

        for i, p in enumerate(all_params):
            # print(p.name)
            if (p.name.find(self.varname) >= 0):
                # print("recording", p.name)
                self.record.append(all_weights[i])

        # def on_batch_end(self, batch, logs={}):
        #    self.record.append(logs.get('loss'))

    def on_epoch_end(self, epoch, logs={}):
        all_params = self.model.get_layer(self.layername)._trainable_weights
        all_weights = self.model.get_layer(self.layername).get_weights()

        for i, p in enumerate(all_params):
            # print(p.name)
            if (p.name.find(self.varname) >= 0):
                # print("recording", p.name)
                self.record.append(all_weights[i])
