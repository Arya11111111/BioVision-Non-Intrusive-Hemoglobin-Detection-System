
import os
import tensorflow as tf
from keras.preprocessing import image
import numpy as np
from deeplearning import graph, model, output_list,level

import cv2
import pymysql as mdb
#model.call = tf.function(model.call)
import time
def upd(disease, ids):
    
    mydb = mdb.connect(
      host="127.0.0.1",
      user="root",
      passwd="",
      database="tmss"
    )

    mycursor = mydb.cursor()

   

    sql = "UPDATE disease SET disease = %s WHERE ids = %s"
    val = (disease, ids)

    mycursor.execute(sql, val)

    mydb.commit()
    mydb.close()

while True:

    mydb = mdb.connect(
          host="127.0.0.1",
          user="root",
          passwd="",
          database="tmss"
        )

    mycursor = mydb.cursor()

    sql = "SELECT ids,pic,disease FROM disease"


    mycursor.execute(sql)
    fine=0
    myresult = mycursor.fetchall()
    mydb.close()
    #print (myresult)
    for x in myresult:
      ids= str(x[0])
      pic=str(x[1])
      fl=str(x[2])
      print (ids,pic,fl)
      if fl=='1':          
        pic='images/'+pic
        #pic='dataset/0/16348281232354043842727266562755 - Pranav Upadhyay.jpg'
        myfile = pic
        time.sleep(2)


        img = image.load_img(myfile, target_size=(150,150))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = img/255

        #with graph.as_default():
        prediction = model.predict(img)
            
            

        prediction_flatten = prediction.flatten()
        var= (prediction_flatten[0])


        max_val_index = np.argmax(prediction_flatten)
        print (max_val_index)
        print(level(max_val_index))
        le=str(level(max_val_index))
        result = output_list[max_val_index]
       
        result= (result+':'+le)
        
        
        upd(result, ids)    
        

          
