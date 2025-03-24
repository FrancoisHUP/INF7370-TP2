
We have this kind of output: 
600/600 ━━━━━━━━━━━━━━━━━━━━ 391s 647ms/step - accuracy: 0.3214 - loss: 1.5714 - val_accuracy: 0.5392 - val_loss: 1.1920

Here we can see we have 600 steps per epoch. That number comes from the total number of traning data (24000*0.8 = 19200) divided by the batch size (19200/32 = 600). One step correspond to the passing of the batch size date, so that would be 32 sample of data with their label. One we pass the examples through the model, then kera use the "categorical_crossentropy" loss and "adam" optimizer to handler the learning rate to update the weight of the model. In fact 600 steps means 600 update of the model. In total every example with label (of every class) will pass through the model in one epoch.    