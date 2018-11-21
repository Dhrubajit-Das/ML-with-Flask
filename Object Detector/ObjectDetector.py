
import cv2 as cv
import numpy

#specify all class names you want to include...
classNames = {1: 'person',
              2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane', 6: 'bus',
              7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light', 11: 'fire hydrant',
              13: 'stop sign', 14: 'parking meter', 15: 'bench', 16: 'bird', 17: 'cat',
              18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow', 22: 'elephant', 23: 'bear',
              24: 'zebra', 25: 'giraffe', 27: 'backpack', 28: 'umbrella', 31: 'handbag',
              32: 'tie', 33: 'suitcase', 34: 'frisbee', 35: 'skis', 36: 'snowboard',
              37: 'sports ball', 38: 'kite', 39: 'baseball bat', 40: 'baseball glove',
              41: 'skateboard', 42: 'surfboard', 43: 'tennis racket', 44: 'bottle',
              46: 'wine glass', 47: 'cup', 48: 'fork', 49: 'knife', 50: 'spoon',
              51: 'bowl', 52: 'banana', 53: 'apple', 54: 'sandwich', 55: 'orange',
              56: 'broccoli', 57: 'carrot', 58: 'hot dog', 59: 'pizza', 60: 'donut',
              61: 'cake', 62: 'chair', 63: 'couch', 64: 'potted plant', 65: 'bed',
              67: 'dining table', 70: 'toilet', 72: 'tv', 73: 'laptop', 74: 'mouse',
              75: 'remote', 76: 'keyboard', 77: 'cell phone', 78: 'microwave', 79: 'oven',
              80: 'toaster', 81: 'sink', 82: 'refrigerator', 84: 'book', 85: 'clock',
              86: 'vase', 87: 'scissors', 88: 'teddy bear', 89: 'hair drier', 90: 'toothbrush' }


class Detector:
    def __init__(self):
        global NeuralNet    #declaring a global variable

        #loading the pre-trained models...
        NeuralNet = cv.dnn.readNetFromTensorflow('model/frozen_inference_graph.pb',
                                             'model/ssd_mobilenet_v1_coco_2017_11_17.pbtxt')



    def object_detection (self, image):
        img = cv.cvtColor(numpy.array(image), cv.COLOR_BGR2RGB)  #changing the image color from BGR TO RGB
        NeuralNet.setInput(cv.dnn.blobFromImage(img, 0.008, (300, 300), (127.5, 127.5, 127.5), swapRB=True, crop=False))
        detections = NeuralNet.forward()  #used to forward-propagate our image and obtain the actual classification
        
        #width and height of the image...
        w = img.shape[1]
        h = img.shape[0]
        

        for i in range(detections.shape[2]):
            probability = detections[0, 0, i, 2]
            if probability > 0.5:
                class_id = int(detections[0, 0, i, 1])
                
                #determining the start-end coordinate of width & height
                w_start = int(detections[0, 0, i, 3] * w)
                h_start = int(detections[0, 0, i, 4] * h)
                w_end = int(detections[0, 0, i, 5] * w)
                h_end = int(detections[0, 0, i, 6] * h)
                
 
                #creating a rectangle with the co-ordinates..
                cv.rectangle(img, (w_start, h_start), (w_end, h_end), (0, 0, 255))

                #checking if the class_id isin class names provided...
                if class_id in classNames:
                    label = classNames[class_id] + ": " + str(probability)  #text to print with the probability
                    label1 = w_start,h_start, w_end, h_end
                    cv.putText(img, label, (w_start+5, h_start+10), cv.FONT_HERSHEY_DUPLEX, 0.5, (255,255,255)) #printing the text
                    
                    print("Confidence of {} class".format(classNames[class_id]))
                    print("Coordinates")
                    print("Top Left    :{0},{1}".format(w_start, h_start))
                    print("Bottom Right:{0},{1}".format(w_end, h_end))
                    print("")
                    print("")

        img = cv.imencode('.jpg', img)[1].tobytes()
        return img