import tensorflow as tf
from keras.models import load_model
# global graph, model, output_list

#graph = tf.get_default_graph()
graph=tf.compat.v1.get_default_graph()
#print (graph)
model = tf.keras.models.load_model('models1/newNeu_model50e.h5')

output_dict = {'Low[9-10]': 0,
               'Moderate[11-13]': 1,
               'Good[13-15]':2,
               'High[16-18]':3,
                  
              
               }


output_list = list(output_dict.keys())
































def level(a):
    import random
    if(a==0):
        

        return(round(random.uniform(9.0,10.0),1))
    elif(a==1):
        return(round(random.uniform(11.0,13.0),1))
    elif(a==2):
        return(round(random.uniform(14.0,15.0),1))
        
    else:
        return(round(random.uniform(16.0,18.0),1))
    
