import os
import cv2 as cv
import numpy as np
import torch
import torchvision.transforms as T


CLASS_NAMES = ['BG', 'person']
ROOT = os.path.dirname(os.path.abspath(__file__))
vid_in_path = os.path.join(ROOT, 'data', 'videos', 'telaviv.mp4')
vid_out_path = os.path.join(ROOT, 'data', 'videos', 'telaviv_out.avi')

def plot_prediction_on_frame(frame, detections):
    # plot prediction on frame
    boxes = detections[0]['boxes'].cpu().detach().numpy()
    labels = detections[0]['labels'].cpu().detach().numpy()
    scores = detections[0]['scores'].cpu().detach().numpy()
    for box, label, score in zip(boxes, labels, scores):
        if score > 0.65:
            x1, y1, x2, y2 = np.array(box, dtype=np.int32)
            cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv.putText(frame, f"Person Walking: {score:.2f}", (x1, y1), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return frame

#########
# Model #
#########

walking_person_detector = torch.load(os.path.join(ROOT, 'mask_rcnn_pedestrians2.pt'))

##########
i_stream = cv.VideoCapture(vid_in_path)
# W / H
work_size = (int(i_stream.get(cv.CAP_PROP_FRAME_WIDTH)), int(i_stream.get(cv.CAP_PROP_FRAME_HEIGHT)))
o_stream = cv.VideoWriter(vid_out_path,
                          cv.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                          i_stream.get(cv.CAP_PROP_FPS),
                          work_size)
counter = 0
while i_stream.isOpened():
    returned_a_value, frame = i_stream.read()
    if returned_a_value:
        # use model to predict frame
        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        T_frame = T.ToTensor()(frame)
        detections = walking_person_detector([T_frame])
        # plot prediction on frame
        plotted_frame = plot_prediction_on_frame(frame, detections)
        # convert back to BGR
        frame = cv.cvtColor(plotted_frame, cv.COLOR_RGB2BGR)

        if counter % 100 == 0 : print(f"frame number {counter+1} was processed!")
        # write frame to output stream
        cv.imshow('frame', frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

        o_stream.write(frame)
    else:
        break
    counter += 1

o_stream.release()
i_stream.release()
cv.DestroyAllWindows()

